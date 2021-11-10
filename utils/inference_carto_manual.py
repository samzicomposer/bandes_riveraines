import numpy as np
import os
import csv

from collections import defaultdict
import gdal
from PIL import Image
from utils.valid_dataset_rgbnir import TestDataset
from scipy import stats

import torch
import torchvision

from torch.utils.data import Dataset
from models.mvdcnn import MVDCNN
from models.pytorch_resnet18 import resnet18

from utils.augmentation import compose_transforms

from sklearn.metrics import confusion_matrix, classification_report
from utils.plot_a_confusion_matrix import plot_confusion_matrix

import random

use_cuda = torch.cuda.is_available()

# Seeds déterministes
seed = 99
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

if not use_cuda:
    print("WARNING: PYTORCH COULD NOT LOCATE ANY AVAILABLE CUDA DEVICE.\n")
else:
    device = torch.device("cuda:0")
    print("All good, a GPU is available.")


def random_pred(classes, prob):
    pred = np.random.choice(classes, p=prob)
    return pred

def CenterCrop(img, dim):
    width, height = img.shape[2], img.shape[1]
    crop_size= dim
    mid_x, mid_y = int(width / 2), int(height / 2)
    cw2, ch2 = int(crop_size / 2), int(crop_size / 2)
    crop_img = img[:, mid_y - ch2:mid_y + ch2, mid_x - cw2:mid_x + cw2]
    return crop_img

# for confusion matrix

classes = ['0', '1', '2', '3', '4']

# rgb
# stds_t = [0.0598, 0.0386, 0.0252] # nouveau jeu de données
# means_t =[0.2062, 0.2971,0.3408]
stds = {1:5.8037, 2: 3.8330, 3:5.0484, 4:9.3750}
means = {1:49.0501, 2:74.7162, 3:86.9113, 4:109.2907}


class PleiadesDataset(torch.utils.data.Dataset):

    def __init__(self, root, views, bands=None,indices=None, transform=None, expand=False):
        assert os.path.isdir(root)
        self.indices = indices
        self.expand = expand
        self.transform = transform
        self.views = views
        self.class_names = ['0', '1', '2', '3', '4']
        self.samples_vfs = []
        self.target_bands = bands

        for root, dir, samp in os.walk(root):
            for name in samp:
                self.samples_vfs.append((os.path.join(root, name)))


        # dictionnaire classant les images par segment d'iqbr
        image_dict1 = defaultdict(list)
        for img_path in self.samples_vfs:
            img = os.path.basename(img_path)
            key = str(img.split('_')[0])
            image_dict1[key].append(img_path)

        del self.samples_vfs

        # nécessaire encore pour avoir des indices en pas de 1
        image_dict = defaultdict(list)
        num = 0
        for key, value in image_dict1.items():
            image_dict[num] = value
            num += 1

        # samples that will be used
        self.samples = image_dict

        # version avec les vues assemblées d'avance
        samples_list = []
        if self.expand:

            for key, samp in self.samples.items():
                samples_list.append(samp)
                # random.shuffle(samp)
                # samples_list.append(random.choices(samp, k=self.views))

            self.samples = samples_list

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.expand:
            sample = self.samples[idx]  # returns [[path, classe]]
            random.shuffle(sample)
        else:
            sample = self.samples[idx]
        segment = sample[0].split('\\')[-1].split('_')[0]
        # if iqbr < 8.0:
        #     iqbr = iqbr**(1+((iqbr-8)/60))
        # iqbr = np.float32(sample[0][1])  # any second element
        # if iqbr == 0:
        #     iqbr = 0.8

        # image = Image.open(sample[0][0])
        image_stack = []
        for samp in sample:

            raster_ds = gdal.Open(samp, gdal.GA_ReadOnly)
            image = []
            for channel in self.target_bands:
                image_arr = raster_ds.GetRasterBand(channel).ReadAsArray()
                assert image_arr is not None, f"Band not found: {channel}"

                image_arr = ((image_arr.astype(np.float32) - means[channel]) / stds[channel]).astype(np.float32)

                image.append(image_arr)
            image = np.dstack(image)

            if self.transform:
                image = self.transform(image)
            image = torchvision.transforms.functional.to_tensor(image)
            image = CenterCrop(image, 46)

            image_stack.append(image)

        # image = torchvision.transforms.functional.to_tensor(np.float32(image))
        # on empile les images/vues
        image_stack = torch.stack(image_stack)
        return image_stack, segment

# views = 16
batch_size = 1
crop_size = 45
views = 16

pre_model = resnet18()
model = MVDCNN(pre_model,'RESNET', 1)


###### test du modèle ######

checkpoints = [

    'MVDCNN_2021-04-22-12_16_53'
]

# si on calcule le mode
y_pred = defaultdict(list)

for c in checkpoints:
    print(c)
    # checkpoint = torch.load(os.path.join(r'I:/annotation/checkpoints/', 'MVDCNN_2021-02-09-14_02_55', c, c[:-3]+'.pth'))
    checkpoint = torch.load(os.path.join(r'I:/annotation/checkpoints/', c, c + '.pth'))

    model.load_state_dict(checkpoint['model_state_dict'])
    if use_cuda:
        model = model.cuda()
    model.eval()

    for i in range(1):

        test_dataset = PleiadesDataset(
            root=r"K:\deep_learning\samples\misc\occ_sol_test",
            views=views, bands=[1,2,3], expand=True)

        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                                  num_workers=0)
        # enlever si on calcule le mode
        # y_pred = defaultdict(list)

        test_correct, test_total = 0, 0
        y_true = []
        segment_ids = []
        for minibatch in test_loader:

            images = minibatch[0]  # rappel: en format BxCxHxW
            segment_id = minibatch[1]
            # images = torch.stack([item for sublist in images for item in sublist])

            if use_cuda:
                images = images.to(device)
            with torch.no_grad():
                preds = model(images)

            # pour les predictions random
            # preds = torch.tensor([random_pred(classes_int, prob) for i in range(len(labels))]).view(-1)
            # top_preds = preds

            # top_preds = preds.topk(k=1, dim=1)[1].view(-1)
            preds = [preds[0].cpu()]

            for p, s in zip(preds, segment_id):
                segment_ids.append(s)
                y_pred[str(i) + str(views) + c].append(p)

            # test_total += top_preds.numel()

        print(f"\nfinal test {i +1}")

##################### confusion matrix #####################

# conversion des prediction en matrice numpy
y_pred_np = []
for k in y_pred.keys():
    y_pred_np.append(np.array([i.numpy() for i in y_pred[k]]))
y_pred_np = np.array(y_pred_np)[0]

# write predictions to csv
fields = ['Segment_id', 'Prediction']
segment_predictions =  [[segment_ids[x]] + [str(y_pred_np[x][0])] for x in range(len(segment_ids))]

with open('predictions/' + checkpoints[0] +'occ_sol_test' + '.csv', 'w', newline='') as f:
    write = csv.writer(f)
    write.writerow(fields)
    write.writerows(segment_predictions)

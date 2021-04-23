import numpy as np
import os
import csv

from collections import defaultdict
from PIL import Image
from scipy import stats

import torch
import torchvision
from torch.utils.data import Dataset
from models.mvdcnn import MVDCNN

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


# for confusion matrix

classes = ['0', '1', '2', '3', '4']
# stds = [0.0598, 0.0386, 0.1064]  # nouveau jeu de données juin et full
# means = [0.2062, 0.2971, 0.4101]  # nouveau jeu de données juin et full
# rgb
stds = [0.0598, 0.0386, 0.0252]  # nouveau jeu de données
means = [0.2062, 0.2971, 0.3408]  # nouveau jeu de données


class PleiadesDataset(torch.utils.data.Dataset):

    def __init__(self, root, views, indices=None, transform=compose_transforms, expand=False):
        assert os.path.isdir(root)
        self.indices = indices
        self.expand = expand
        self.transform = transform
        self.views = views
        self.class_names = ['0', '1', '2', '3', '4']
        self.samples_vfs = []

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


        segment = os.path.basename(sample[0]).split('_')[0]

        #
        # if len(sample) < self.views:
        #     # raster_paths = sample
        #     raster_paths = random.choices(sample, k=self.views)
        # else:
        #     # raster_paths =  sample
        #     raster_paths = random.sample(sample, self.views)

        # image = Image.open(sample[0][0])
        image_stack = []
        for path in sample:
            image = Image.open(path)

            if self.transform:
                image = self.transform(image)
            image_stack.append(image)

        # convert numpy array to torch tensor
        # image = torchvision.transforms.functional.to_tensor(np.float32(image))
        image_stack = torch.stack(image_stack)

        return image_stack, segment  # return tuple with class index as 2nd member


# views = 16
batch_size = 1
crop_size = 45
views = 16

base_transforms = torchvision.transforms.Compose([torchvision.transforms.RandomApply([
    torchvision.transforms.RandomRotation(45),
], p=0.9),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.RandomVerticalFlip(p=0.5),
    torchvision.transforms.ColorJitter(0.2, 0.2, 0.2),
    torchvision.transforms.CenterCrop(45),
    # torchvision.transforms.RandomCrop(crop_size),

    torchvision.transforms.ToTensor(),  # rescale de 0 à 1, de là les valeurs ci-dessous
    torchvision.transforms.Normalize(mean=(means[0], means[1], means[2]), std=(stds[0], stds[1], stds[2]))
])

pre_model = torchvision.models.resnet18(pretrained=True)
model = MVDCNN(pre_model, 1)

###### test du modèle ######

checkpoints = [

    'MVDCNN_2021-03-17-11_24_55'
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

    for i in range(5):

        test_dataset = PleiadesDataset(
            root=r"D:\deep_learning\samples\manual_br\rgb\\",
            # root=r"D:/deep_learning/samples/jeux_separes/test/all_values_v2_rgb/",
            views=views, transform=base_transforms, expand=True)

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
                preds = model(images).view(-1)

            # pour les predictions random
            # preds = torch.tensor([random_pred(classes_int, prob) for i in range(len(labels))]).view(-1)
            # top_preds = preds

            preds = [preds[0].cpu()]

            for p, s in zip(preds, segment_id):
                segment_ids.append(s)
                y_pred[str(i) + str(views) + c].append(p)

            test_total += 1

        print(f"\nfinal test {i +1}")

##################### confusion matrix #####################

# conversion des prediction en matrice numpy
y_pred_np = []
for k in y_pred.keys():
    y_pred_np.append(np.array([i.numpy() for i in y_pred[k]]))
y_pred_np = np.array(y_pred_np)
# print(diff)

# calcul de la moyenne des predictions
avg = np.average(y_pred_np, 0)
avg[avg<1.7] = 1.7
avg[avg>10.0] = 10.0
# classe ayant la plus haute moyenne de prediction
# rounded_pred = np.round(avg, 0)


# write predictions to csv
fields = ['Segment_id', 'Prediction']
segment_predictions =  [[segment_ids[x]] + [avg[x]] for x in range(len(segment_ids))]

with open('predictions/' + checkpoints[0] +'manual_set' + '.csv', 'w', newline='') as f:
    write = csv.writer(f)
    write.writerow(fields)
    write.writerows(segment_predictions)

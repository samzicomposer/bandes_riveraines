import numpy as np
import os
import time
from datetime import datetime

from collections import defaultdict
from PIL import Image
from scipy import stats

import torch
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset
from models.mvdcnn import MVDCNN

from utils.augmentation import compose_transforms
from utils.logger import logger

from sklearn.metrics import confusion_matrix, classification_report
from utils.plot_a_confusion_matrix import plot_confusion_matrix
import matplotlib.pyplot as plt
from scipy import ndimage


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
stds = [0.0598, 0.0386, 0.1064] # nouveau jeu de données juin et full
means =[0.2062, 0.2971,0.4101]  # nouveau jeu de données juin et full

class PleiadesDataset(torch.utils.data.Dataset):

    def __init__(self, root, views,indices=None, transform=compose_transforms, expand = False):
        assert os.path.isdir(root)
        self.indices = indices
        self.expand = expand
        self.transform = transform
        self.views = views
        self.class_names = ['0', '1', '2', '3', '4']
        self.samples_vfs = []

        for class_idx, class_set in enumerate(self.class_names):
            for class_name in class_set:
                class_dir = root + "/" + class_name
                if os.path.isdir(class_dir):
                    image_paths = [os.path.join(class_dir, p) for p in os.listdir(class_dir)]
                    for p in image_paths:
                        if p.endswith(".tif"):
                            self.samples_vfs.append((p, int(class_name)))

        # dictionnaire classant les images par segment d'iqbr
        image_dict1 = defaultdict(list)
        for img_path in self.samples_vfs:
            img = os.path.basename(img_path[0])
            key = str(img.split('_')[0])
            image_dict1[key].append(img_path)

        del self.samples_vfs
        # renomme chacune des clefs de 0 à x, nécessaire pour le get_item
        # image_dict = defaultdict(list)
        # num = 0
        # for key, value in image_dict1.items():
        #     image_dict[num] = value
        #     num+=1

        # genere une copie de la liste d'images pour chaque segment en fonction de leur nombre
        # si un segment a 24 images et quon a des vues à 3 images, 8 copies sont générées
        # image_views = defaultdict(list)
        # for key, samples in image_dict1.items():
        #     num_samples = len(samples) // self.views
        #     # random.shuffle(samples)
        #     for view in range(num_samples):
        #         image_views[key].append(samples)
        #         # del samples[0:self.views]

        # nécessaire encore pour avoir des indices en pas de 1
        image_dict = defaultdict(list)
        num = 0
        for key, value in image_dict1.items():
            image_dict[num] = value
            num += 1

        # samples that will be used

        self.samples = image_dict

        ### va chercher tous les exemples pour chacune des clefs
        ### (segments de rive) et les met en liste
        ### cela permet un assemblage aleatoire des vues tout en conservant chacun des segments dans un seul dataset (train,val,test)
        # if self.expand:
        #     samples_list = []
        #     for key, samp in self.samples.items():
        #         if key in self.indices:
        #             for s in samp:
        #                 samples_list.append(s)

        # version avec les vues assemblées d'avance
        samples_list = []
        if self.expand:

            for key, samp in self.samples.items():
                if key in self.indices:
                    random.shuffle(samp)

                    if len(samp)//self.views < 1:
                        samples_list.append(random.choices(samp, k=self.views))
                    else:
                        for i in range(len(samp)//self.views):
                            b = i*self.views
                            # samples_list.append(random.sample(s, k=self.views))
                            samples_list.append(samp[b:b+self.views])

            self.samples = samples_list


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.expand:
            sample = self.samples[idx] # returns [[path, classe]]
            random.shuffle(sample)
        else:
            sample = self.samples[idx]

        iqbr = sample[0][1]  # any second element

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
            image = Image.open(path[0])

            if self.transform:
                image = self.transform(image)
            image_stack.append(image)

        # convert numpy array to torch tensor
        # image = torchvision.transforms.functional.to_tensor(np.float32(image))
        image_stack = torch.stack(image_stack)
        return image_stack, iqbr  # return tuple with class index as 2nd member


views = 10
batch_size = 512
epochs = 15
crop_size = 45

base_transforms = torchvision.transforms.Compose([torchvision.transforms.RandomApply([
    torchvision.transforms.RandomRotation(45),
], p=0.9),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.RandomVerticalFlip(p=0.5),
    torchvision.transforms.ColorJitter(0.3,0.3,0.3),
    # torchvision.transforms.CenterCrop(45),
    torchvision.transforms.RandomCrop(crop_size),

    torchvision.transforms.ToTensor(),  # rescale de 0 à 1, de là les valeurs ci-dessous
    torchvision.transforms.Normalize(mean=(means[0], means[1], means[2]), std=(stds[0], stds[1], stds[2]))
])
test_transforms = torchvision.transforms.Compose([
    # torchvision.transforms.CenterCrop(45),
    torchvision.transforms.RandomCrop(crop_size),
    torchvision.transforms.ToTensor(),  # rescale de 0 à 1, de là les valeurs ci-dessous
    torchvision.transforms.Normalize(mean=(means[0], means[1], means[2]), std=((stds[0], stds[1], stds[2])))
])

temp_test_dataset = PleiadesDataset(root=r"D:/deep_learning/samples/jeux_separes/test/iqbr_cl_covabar_7mdist/", views=1, transform=base_transforms, indices = None)
test_indices = np.random.permutation(len(temp_test_dataset)).tolist()
test_sampler = torch.utils.data.sampler.SubsetRandomSampler(test_indices)
test_dataset = PleiadesDataset(root=r"D:/deep_learning/samples/jeux_separes/test/iqbr_cl_covabar_7mdist/", views=views, transform=base_transforms, indices = test_sampler.indices, expand=True)

test_rsampler = torch.utils.data.sampler.RandomSampler(test_dataset)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=0)

pre_model = torchvision.models.resnet18(pretrained=True)
model = MVDCNN(pre_model, len(classes))


###### test du modèle ######

checkpoints = [
                # 'MVDCNN_2020-12-01-15_00_08' # L4
                # 'MVDCNN_2020-12-03-08_09_59' # 15 epochs
                #  'MVDCNN_2020-12-01-22_10_42', # 15 EPOCHS, 95-5, 55%, tres bien
                # 'MVDCNN_2020-12-09-11_47_16' # 1.2 cl3, 99-1, 25
               # 'MVDCNN_2020-12-01-15_39_19' # excellent pour 2 premiere classes 54%
               'MVDCNN_2020-12-01-16_18_12' # excellente 4e classe, 95-5, arret à 12e epoch, tres bien!
               #  'MVDCNN_2020-12-09-11_05_10' # Last epoch 25,1.1 pour cl3, 99-1, 53,8%
               #  'MVDCNN_2020-12-11-15_11_38' # LAST EPOCH 20, 10 views
               #  'MVDCNN_2020-11-27-23_09_24' # classifier 512

               ]

# si on calcule le mode
y_pred = defaultdict(list)

prob = np.array([94,122,43,39,79])/377.0
classes_int = [0,1,2,3,4]

for c in checkpoints:
    print(c)
    checkpoint = torch.load(os.path.join(r'I:/annotation/checkpoints/', c, c + '.pth'))

    model.load_state_dict(checkpoint['model_state_dict'])
    if use_cuda:
        model = model.cuda()
    model.eval()

    views = [16]

    for v in views:
        print(f'views: {v}')
        for i in range(9):

            test_dataset = PleiadesDataset(root=r"D:/deep_learning/samples/jeux_separes/test/iqbr_cl_covabar_7mdist/",
                                           views=v, transform=base_transforms, indices=test_sampler.indices,
                                           expand=True)

            test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                                      num_workers=0)
            # enlever si on calcule le mode
            # y_pred = defaultdict(list)

            test_correct, test_total = 0, 0
            y_true = []
            for minibatch in test_loader:

                images = minibatch[0]  # rappel: en format BxCxHxW
                labels = minibatch[1]  # rappel: en format Bx1
                # images = torch.stack([item for sublist in images for item in sublist])

                if use_cuda:
                    images = images.to(device)
                    labels = labels.to(device)
                with torch.no_grad():
                    preds = model(images)

                # preds = torch.tensor([random_pred(classes_int, prob) for i in range(len(labels))]).view(-1)
                # top_preds = preds

                top_preds = preds.topk(k=1, dim=1)[1].view(-1)

                for p, l in zip(top_preds, labels):
                    y_true.append(l)
                    y_pred[str(i)+str(v)+c].append(p)
                test_correct += (top_preds == labels).nonzero().numel()
                test_total += labels.numel()

            test_accuracy = test_correct / test_total
            print(f"\nfinal test: accuracy={test_accuracy:0.4f}\n")



##################### confusion matrix #####################

y_pred_np = []
for k in y_pred.keys():
    y_pred_np.append(np.array([i.item() for i in y_pred[k]]))

y_pred_mode = np.array(stats.mode(np.array(y_pred_np),0)[0])[0]
# y_pred_std = np.std(y_pred_np,0)

# y_diff = y_pred_np-y_pred_avg
# filtre = np.where(abs(y_diff) <= abs(y_pred_std), y_pred_np, 100)
# y_pred_avg = np.mean(y_pred_np,0)

# faut trouver une façon d'enlever les valeurs abberantes dans le nouveau calcul de la moyenne
conf_class_map = {idx: name for idx, name in enumerate(test_dataset.class_names)}
y_true_final = [conf_class_map[idx.item()] for idx in y_true]
y_pred_final = [conf_class_map[idx.item()] for idx in y_pred_mode]

class_report = classification_report(y_true_final, y_pred_final, target_names=classes)
cm = confusion_matrix(y_true_final, y_pred_final)

print(class_report)

plot_confusion_matrix(cm, 'test_cf', normalize=True,
                      target_names=classes,
                      title="Confusion Matrix", show=True, save=False)



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
# rgb
stds = [0.0598, 0.0386, 0.0252]  # nouveau jeu de données
means = [0.2062, 0.2971, 0.3408]  # nouveau jeu de données


# rg-nir
# stds = [0.0598, 0.0386, 0.1064] # nouveau jeu de données juin et full
# means =[0.2062, 0.2971,0.4101]  # nouveau jeu de données juin et full

class PleiadesDataset(torch.utils.data.Dataset):

    def __init__(self, root, views, indices=None, transform=compose_transforms, expand=False):
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
                # if key in self.indices:
                # random.shuffle(samp)

                # if len(samp)//self.views < 1:
                samples_list.append(samp)
                # samples_list.append(random.choices(samp, k=self.views))
                # else:
                #     for i in range(len(samp)//self.views):
                #         b = i*self.views
                #         # samples_list.append(random.sample(s, k=self.views))
                #         samples_list.append(samp[b:b+self.views])

            self.samples = samples_list

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.expand:
            sample = self.samples[idx]  # returns [[path, classe]]
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


views = 4
batch_size = 1
crop_size = 45

base_transforms = torchvision.transforms.Compose([torchvision.transforms.RandomApply([
    torchvision.transforms.RandomRotation(45),
], p=0.9),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.RandomVerticalFlip(p=0.5),
    torchvision.transforms.ColorJitter(0.2,0.2,0.2),
    torchvision.transforms.CenterCrop(crop_size),
    # torchvision.transforms.RandomCrop(crop_size),

    torchvision.transforms.ToTensor(),  # rescale de 0 à 1, de là les valeurs ci-dessous
    torchvision.transforms.Normalize(mean=(means[0], means[1], means[2]), std=(stds[0], stds[1], stds[2]))
])
pre_model = torchvision.models.resnet18(pretrained=True)
model = MVDCNN(pre_model, len(classes))

###### test du modèle ######

checkpoints = [

    'MVDCNN_2021-02-24-13_08_30'
]

test_dataset = PleiadesDataset(
        root=r"D:/deep_learning/samples/jeux_separes/test/iqbr_cl_covabar_10mdist_obcfiltered3_rgb/",
        views=16, transform=base_transforms, expand=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,num_workers=0)

# si on calcule le mode
y_pred = defaultdict(list)

prob = np.array([94, 122, 43, 39, 79]) / 377.0
classes_int = [0, 1, 2, 3, 4]

###############################################################################
'''
On prend la moyenne des classes prédites pour chaque bande riveraine selon l'assemblage des vue. 
Exemple: bande riveraine de 16 images, 4 vues donc 4 prédictions. On fait la moyenne de ces 4 prédictions, et ce
un nombre x de fois. Une fois sorti de la boucle, on arrondi la moyenne des moyennes à la classe la plus près.
'''

for c in checkpoints:
    print(c)

    # checkpoint = torch.load(os.path.join(r'I:/annotation/checkpoints/','MVDCNN_2021-02-09-14_02_55', c, c[:-3] + '.pth'))
    checkpoint = torch.load(os.path.join(r'I:/annotation/checkpoints/', c, c + '.pth'))

    model.load_state_dict(checkpoint['model_state_dict'])
    if use_cuda:
        model = model.cuda()
    model.eval()

    for i in range(10):

        # enlever si on calcule le mode
        # y_pred = defaultdict(list)

        test_correct, test_total = 0, 0
        y_true = []
        for minibatch in test_loader:

            view_features = []
            if len(minibatch[0][0]) < views:
                view_features.append(minibatch[0][0][:])
            else:
                # on separe la bande en groupes selon le nombre de vues
                range_v = (len(minibatch[0][0])//views)
                for v in range(range_v):
                    # on assemble les images selon le nombre de vues
                    coord = v*views
                    view_features.append(minibatch[0][0][coord:coord+views])
                # if range_v % views != 0:
                #     view_features.append(minibatch[0][0][coord+views:])

            images = torch.stack(view_features)
            labels = minibatch[1]  # rappel: en format Bx1

            if use_cuda:
                images = images.to(device)
                labels = labels.to(device)
            with torch.no_grad():
                preds = model(images)

            # la meilleure prediction pour chacune des vues
            top_preds = preds.topk(k=1, dim=1)[1].view(-1)
            # la classe moyenne predite par les assemblages de vues (toute la bande)
            avg_prediction = [torch.mean(top_preds.double())]

            for p, l in zip(avg_prediction, labels):
                y_true.append(l)
                y_pred[str(i) + str(views) + c].append(p)
            # on arrondi la moyenne à la classe la plus près pour le test de classification
            test_correct += (torch.round(avg_prediction[0]) == labels).nonzero().numel()
            test_total += labels.numel()

        test_accuracy = test_correct / test_total
        print(f"\nfinal test {i}: accuracy={test_accuracy:0.4f}\n")

##################### confusion matrix #####################

# conversion des prediction en matrice numpy
y_pred_np = []
for k in y_pred.keys():
    y_pred_np.append(np.array([i.cpu().numpy() for i in y_pred[k]]))
y_pred_np = np.array(y_pred_np)

# calcul de la moyenne des predictions
avg = np.average(y_pred_np, 0)
# classe la plus proche (arrondie)
rounded_pred = np.round(avg, 0)

# y_pred_mode = np.array(stats.mode(np.array(y_pred_np), 0)[0])[0]
# mean = np.array(np.mean(np.array(y_pred_np), 0))
# vrai = [i.item() for i in y_true]
# diff = mean - vrai
# print(diff)

conf_class_map = {idx: name for idx, name in enumerate(test_dataset.class_names)}
y_true_final = [conf_class_map[idx.item()] for idx in y_true]
y_pred_final = [conf_class_map[idx.item()] for idx in rounded_pred]

# std = np.round(np.std(y_pred_mode - np.array(y_true).astype(int)), 3)
ecart = np.abs(rounded_pred - np.array(y_true).astype(int))
one_class_diff_miss_percent = np.round(np.count_nonzero((np.array(ecart) == 1)) / np.count_nonzero(ecart != 0), 3)
one_class_diff_percent = np.round(np.count_nonzero((np.array(ecart) <= 1)) / len(ecart), 3)
# print(f'std = {std},\n % des mal classés à une classe d\'écart = {one_class_diff_miss_percent}')
print(f'Prédictions à + ou - une classe d\'écart = {one_class_diff_percent}')

class_report = classification_report(y_true_final, y_pred_final, target_names=classes)
cm = confusion_matrix(y_true_final, y_pred_final)

print(class_report)

plot_confusion_matrix(cm, 'test_cf', normalize=True,
                      target_names=classes,
                      title="Confusion Matrix", show=True, save=False)



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
seed = 48
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

        # iqbr = sample[0][1]  # any second element
        iqbr = np.float32(sample[0][0].split('_')[-1][:-4])  # la valeur dans le nom en enlevant le .tiff à la fin
        # iqbr = round(iqbr, 1)

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


views = 1
batch_size = 1
crop_size = 45

base_transforms = torchvision.transforms.Compose([torchvision.transforms.RandomApply([
    torchvision.transforms.RandomRotation(45),
], p=0.9),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.RandomVerticalFlip(p=0.5),
    torchvision.transforms.ColorJitter(0.2,0.2,0.2),
    torchvision.transforms.CenterCrop(crop_size),
    # torchvision.transforms.Resize(64),

    torchvision.transforms.ToTensor(),  # rescale de 0 à 1, de là les valeurs ci-dessous
    torchvision.transforms.Normalize(mean=(means[0], means[1], means[2]), std=(stds[0], stds[1], stds[2]))
])
pre_model = torchvision.models.resnet18(pretrained=True)
model = MVDCNN(pre_model, 1)

###### test du modèle ######

checkpoints = [

    'MVDCNN_2021-03-17-11_24_55'
]

test_dataset = PleiadesDataset(
        root=r"D:/deep_learning/samples/jeux_separes/test/all_values_v2_rgb/",
        views=1, transform=base_transforms, expand=True)

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

    for i in range(5):

        # enlever si on calcule le mode
        # y_pred = defaultdict(list)

        test_ecart, test_total = [], 0
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
                preds = model(images).view(-1)

            # la meilleure prediction pour chacune des vues
            # top_preds = preds.topk(k=1, dim=1)[1].view(-1)
            # la classe moyenne predite par les assemblages de vues (toute la bande)
            avg_prediction = [torch.mean(preds)]
            avg_prediction= [avg_prediction[0].cpu()]

            for p, l in zip(avg_prediction, labels):
                y_true.append(l)
                y_pred[str(i) + str(views) + c].append(p)
                test_ecart.append(abs(p - l).item())
            # on arrondi la moyenne à la classe la plus près pour le test de classification

            test_total += labels.numel()
        test_ecart = np.array(test_ecart)*10
        test_accuracy = np.sqrt(np.mean(test_ecart**2))
        print(f"\nfinal test {i}: RMSE = {test_accuracy:0.4f}\n")

##################### confusion matrix #####################

# conversion des prediction en matrice numpy
y_pred_np = []
for k in y_pred.keys():
    y_pred_np.append(np.array([i.numpy() for i in y_pred[k]]))
y_pred_np = np.array(y_pred_np)

# calcul de la moyenne des predictions
y_true = np.array([x.item() for x in y_true])*10
avg = np.average(y_pred_np, 0)*10
# avg[avg<17] = 17
avg[avg>100] = 100
# y_true[y_true==40] = 80
avg_error = np.round(np.mean(abs(avg-y_true)),3)
# classe ayant la plus haute moyenne de prediction
rounded_pred = np.round(avg, 0)

mean = np.mean(abs(avg-y_true))
std = np.std(abs(avg-y_true))
rmse = np.sqrt(np.mean((avg-y_true)**2))

means = []
stds = []
for cl in set(y_true):
    masque = np.array(y_true)!=cl
    z = np.ma.array(avg,mask=masque)
    means.append(np.mean(z))
    stds.append(np.std(z - y_true))

stds = np.array(stds)
stds[stds==0.0] = 'NaN'
plt.figure(figsize=(8,5))

plt.scatter(y_true, avg, color = 'black', marker='.')
# plt.errorbar(list(set(y_true)), means, stds, linestyle = 'None',color='black', marker = '.', capsize = 3)
coef = np.polyfit(y_true, avg, 1)
slope, intercept, r_value, p_value, std_err = stats.linregress(y_true, avg)
plt.xlabel('IQBR terrain')
plt.ylabel('IQBR prédit')
plt.text(30, 95, f"r2: {round(r_value**2,2)}")
plt.text(30,90, f"Écart type des erreurs: {np.round(std,2)}")
plt.text(30,85, f"RMSE: {np.round(rmse,2)}")
plt.plot(y_true, intercept + (np.array(y_true) * slope))
plt.grid()
# plt.xticks(np.arange(min(y_true), max(y_true)+1, 0.5))
# plt.xlim(1.7,10.0)
plt.show()

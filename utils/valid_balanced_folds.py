"""
Script utilisé pour évaluer les résultats sur le jeu de validation, en fin d'entrainement,
sur une version balancée de ce jeu de validation.
Le jeu de données est chargé comme à l'entrainement (5-fold) mais un checkpoint est utilisé comme modèle.
"""

import numpy as np
import os
import time
import random
from datetime import datetime
from collections import defaultdict
from utils.valid_dataset_rgbnir import ValidDataset
from scipy import stats
import matplotlib.pyplot as plt
import gdal

import torch
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset
from models.mvdcnn import MVDCNN
from models.pytorch_resnet18 import resnet18

from utils.augmentation import compose_transforms
from utils.logger import logger


def CenterCrop(img, dim):
    width, height = img.shape[2], img.shape[1]
    crop_size= dim
    mid_x, mid_y = int(width / 2), int(height / 2)
    cw2, ch2 = int(crop_size / 2), int(crop_size / 2)
    crop_img = img[:, mid_y - ch2:mid_y + ch2, mid_x - cw2:mid_x + cw2]
    return crop_img

use_cuda = torch.cuda.is_available()

if not use_cuda:
    print("WARNING: PYTORCH COULD NOT LOCATE ANY AVAILABLE CUDA DEVICE.\n")
else:
    device = torch.device("cuda:0")
    print("All good, a GPU is available.")


seed = 1
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True


# Normalisation : matrice transformée en tenseur (entre 0 et 1) - ACADIE RGBNIR
# stds_t = [0.0598, 0.0386, 0.0252, 0.1064] # nouveau jeu de données
# means_t =[0.2062, 0.2971,0.3408, 0.4101]  # nouveau jeu de données

# Normalisation (0-255)
stds = {1:5.8037, 2: 3.8330, 3:5.0484, 4:9.3750, 5: 15.3279}
means = {1:49.0501, 2:74.7162, 3:86.9113, 4:109.2907, 5: 33.8298}

class PleiadesDataset(torch.utils.data.Dataset):

    def __init__(self, root, views, bands, indices=None, transform=compose_transforms, expand = False):
        assert os.path.isdir(root)
        self.indices = indices
        self.expand = expand
        self.target_bands = bands
        self.transform = transform()
        self.views = views
        self.class_names = ['0', '1', '2', '3', '4']
        self.samples_vfs = []

        # regroupement des images avec leur classe d'iqbr (représentée par le nom du répertoire dans lequel elles se trouvent)
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

        # on crée un dictionnaire en fonction des classes: classe:[liste de bandes riveraine[liste d'images]]
        z= defaultdict(list)
        for key, samp in image_dict1.items():
            z[samp[0][1]].append(samp)
        # on retient le minimum
        m = min([len(z[r]) for r in z])

        # on recré la liste en conservant le même nombre de br dans chaque classe
        # ex: s'il y a 3x plus de BR de classe 2 que le minimum, il y aura autant de BR de classe 2 que le min, mais 3 par paquet.
        # Ça permet d'aller éventuellement chercher toutes les BR du jeu de données.
        img_list = []
        for key, samp in z.items():
            nombre_par_paquet = len(samp) // m
            for i in range(m):
                b = i * nombre_par_paquet
                img_list.append(samp[b:b + nombre_par_paquet])

        # nécessaire encore pour avoir des indices en pas de 1
        image_dict = defaultdict(list)
        num = 0
        for value in img_list:
            image_dict[num] = value
            num += 1

        # samples that will be used
        self.samples = image_dict

        # version avec les vues assemblées
        samples_list = []
        if self.expand:

            for key, samp in self.samples.items():
                if key in self.indices:

                    # pour grossir le jeu de données, on choisit 3x(number) une bande riveraine dans les paquets
                    number = 1
                    # choix de la BR dans le paquet
                    for i in range(number):
                        samp1 = random.sample(samp,1)[0]
                        # choix des images de la BR
                        # if len(samp1)//self.views < 1:
                        #     samples_list.append(random.choices(samp1, k=self.views))
                        # else:
                        #     samples_list.append(random.sample(samp1, k=self.views))
                        samples_list.append(samp1)
            self.samples = samples_list

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.expand:
            sample = self.samples[idx] # returns [[path, classe]]
            random.shuffle(sample)
        else: # unused in this setup
            sample = self.samples[idx]

        iqbr = np.float32(sample[0][0].split('_')[-1][:-4]) # la valeur dans le nom en enlevant le .tiff à la fin

        image_stack = [] # toutes les images selon le nb de vues
        for samp in sample:

            raster_ds = gdal.Open(samp[0], gdal.GA_ReadOnly)
            image = [] # les bandes de l'images traitée
            for channel in self.target_bands:
                image_arr = raster_ds.GetRasterBand(channel).ReadAsArray()
                assert image_arr is not None, f"Band not found: {channel}"

                # normalisation
                image_arr = ((image_arr.astype(np.float32) - means[channel]) / stds[channel]).astype(np.float32)

                image.append(image_arr)
            image = np.dstack(image)

            # tranformations
            if self.transform:
                image = self.transform(image)

            image = torchvision.transforms.functional.to_tensor(image)
            image = CenterCrop(image, 46)

            image_stack.append(image)

        # on empile les images/vues
        image_stack = torch.stack(image_stack)
        return image_stack, np.float32(iqbr)  # return tuple with class index as 2nd member

class PleiadesTestDataset(torch.utils.data.Dataset):

    def __init__(self, root, views, bands, indices=None, transform=compose_transforms, expand = False):
        assert os.path.isdir(root)
        self.indices = indices
        self.expand = expand
        self.target_bands = bands
        self.transform = transform()
        self.views = views
        self.class_names = ['0', '1', '2', '3', '4']
        self.samples_vfs = []

        # regroupement des images avec leur classe d'iqbr (représentée par le nom du répertoire dans lequel elles se trouvent)
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

        # on crée un dictionnaire en fonction des classes: classe:[liste de bandes riveraine[liste d'images]]
        z= defaultdict(list)
        for key, samp in image_dict1.items():
            z[samp[0][1]].append(samp)
        # on retient le minimum
        m = min([len(z[r]) for r in z])

        # on recré la liste en conservant le même nombre de br dans chaque classe
        # ex: s'il y a 3x plus de BR de classe 2 que le minimum, il y aura autant de BR de classe 2 que le min, mais 3 par paquet.
        # Ça permet d'aller éventuellement chercher toutes les BR du jeu de données.
        img_list = []
        for key, samp in z.items():
            nombre_par_paquet = len(samp) // m
            for i in range(m):
                b = i * nombre_par_paquet
                img_list.append(samp[b:b + nombre_par_paquet])

        # nécessaire encore pour avoir des indices en pas de 1
        image_dict = defaultdict(list)
        num = 0
        for value in img_list:
            image_dict[num] = value
            num += 1

        # samples that will be used
        self.samples = image_dict

        # version avec les vues assemblées
        samples_list = []
        if self.expand:

            new_dict = defaultdict(list)

            for key, samp in self.samples.items():
                if key in self.indices:
                    for s in samp:
                        new_dict[s[0][0].split('/')[1][0]].append(s)


            m = min([len(z) for a, z in new_dict.items()])
            for classe, samp in new_dict.items():
                samp1 = random.sample(samp, m)
                for s in samp1:
                    samples_list.append(s)

            self.samples = samples_list

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.expand:
            sample = self.samples[idx] # returns [[path, classe]]
            random.shuffle(sample)
        else: # unused in this setup
            sample = self.samples[idx]

        iqbr = np.float32(sample[0][0].split('_')[-1][:-4]) # la valeur dans le nom en enlevant le .tiff à la fin

        image_stack = [] # toutes les images selon le nb de vues
        for samp in sample:

            raster_ds = gdal.Open(samp[0], gdal.GA_ReadOnly)
            image = [] # les bandes de l'images traitée
            for channel in self.target_bands:
                image_arr = raster_ds.GetRasterBand(channel).ReadAsArray()
                assert image_arr is not None, f"Band not found: {channel}"

                # normalisation
                image_arr = ((image_arr.astype(np.float32) - means[channel]) / stds[channel]).astype(np.float32)

                image.append(image_arr)
            image = np.dstack(image)

            # tranformations
            if self.transform:
                image = self.transform(image)

            image = torchvision.transforms.functional.to_tensor(image)
            image = CenterCrop(image, 46)

            image_stack.append(image)

        # on empile les images/vues
        image_stack = torch.stack(image_stack)
        return image_stack, np.float32(iqbr)  # return tuple with class index as 2nd member

# hyperparamètres
views = 4
batch_size = 128
epochs = 3
crop_size = 46
bands = [1,2,4]

# À cette étape, le nombre de vue n'a pas d'importance, de là expand=False. On charge toutes les images dans le dataset. C'est seulement pour générer les indices.
dataset = PleiadesDataset(root=r"K:\deep_learning\samples\jeux_separes\train\all_values_obcfiltered3_rgbnir", bands = bands, views=1, indices = None, expand=False)

# l'ensemble des bandes riveraines (en paquets) sous forme d'indice
sample_idxs = np.random.permutation(len(dataset)).tolist()

####### 5-fold creation ######
fold_count = int(0.2 * len(sample_idxs))
fold_1 = sample_idxs[0:fold_count]
fold_2 = sample_idxs[fold_count:fold_count*2]
fold_3 = sample_idxs[fold_count*2:fold_count*3]
fold_4 = sample_idxs[fold_count*3:fold_count*4]
fold_5 = sample_idxs[fold_count*4:]
folds = [fold_1,fold_2,fold_3,fold_4,fold_5]

cp = [  'MVDCNN_2021-06-13-19_44_15',
        'MVDCNN_2021-06-13-20_06_19',
        'MVDCNN_2021-06-13-20_28_18',
        'MVDCNN_2021-06-13-20_50_23',
        'MVDCNN_2021-06-13-21_13_36'
]

num = 0
for fold, check in zip(folds, cp):
    num+=1
    test_fold = fold
    # on récolte les indices qui ne font pas partie du fold de test
    train_idx = []
    for f in folds:
        if f != test_fold:
            for indice in f:
                train_idx.append(indice)

    # on garde 10% du 80% en validation
    train_sample_count = int(0.9 * len(train_idx))

    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx[0:train_sample_count])
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx[train_sample_count:])
    test_sampler = torch.utils.data.sampler.SubsetRandomSampler(test_fold)

    # creation des datasets utilisés
    # train_dataset = PleiadesDataset(root=r"K:\deep_learning\samples\jeux_separes\train\all_values_obcfiltered3_rgbnir", bands = bands,views=views,indices = train_sampler.indices, expand=True)
    # val_dataset = ValidDataset(root=r"K:\deep_learning\samples\jeux_separes\train\all_values_obcfiltered3_rgbnir", bands = bands,views=views, indices = valid_sampler.indices, expand=True)
    # le 5e fold pas utilisé dans l'article. On a tout de même les résultats de celui-ci à la fin du processus.
    test_dataset = PleiadesTestDataset(root=r"K:\deep_learning\samples\jeux_separes\train\all_values_obcfiltered3_rgbnir", bands = bands,views=views, indices = test_sampler.indices, expand=True)

    # samplers aléatoires
    # train_rsampler = torch.utils.data.sampler.RandomSampler(train_dataset)

    # dataloaders
    # train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,sampler=train_rsampler,  num_workers=0)
    # pas de sampling aléatoire car toutes les images sans transformation y passent
    # valid_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=1, num_workers=0)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, num_workers=0)

    ##### test du modèle #####

    pre_model = resnet18()
    model = MVDCNN(pre_model, 'RESNET', 1)

    checkpoint = torch.load(os.path.join(r'I:/annotation/checkpoints/', check, check + '.pth'))

    model.load_state_dict(checkpoint['model_state_dict'])
    if use_cuda:
        model = model.cuda()
    model.eval()

    y_pred = []
    test_ecart = []
    y_true = []
    for minibatch in test_loader:

        view_features = []

        # on separe la bande en groupes selon le nombre de vues
        # bcp plus rapide en entrainement que de passer toutes les images d'un seul coup avec un batch de 1
        range_v = (len(minibatch[0][0]) // 1)
        for v in range(range_v):
            # on assemble les images selon le nombre de vues
            coord = v * 1
            view_features.append(minibatch[0][0][coord:coord + 1])

        images = torch.stack(view_features)
        labels = minibatch[1]  # rappel: en format Bx1

        if use_cuda:
            images = images.to(device)
            labels = labels.to(device)
        with torch.no_grad():
            preds = model(images).view(-1)

        avg_prediction = torch.mean(preds)
        avg_prediction = avg_prediction.cpu()
        y_true.append(labels.cpu())  # une seule valeur (batch 1)
        y_pred.append(avg_prediction)
        test_ecart.append(abs(avg_prediction - labels).item())

    # les valeurs d'iqbr sont de 1.7-10 dans le jeu de données, on les remet à 17-100
    test_ecart = np.array(test_ecart)
    test_loss = np.mean(test_ecart ** 2) # MSE qu'on utilise pour le graphique
    test_MSE = test_ecart*10 # MSE IQBR de 17-100
    test_RMSE = np.sqrt(np.mean(test_MSE ** 2))
    # print(f"\nFold test RMSE = {test_RMSE:0.4f}\n")


    ##################### figure  #####################


    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111)
    ax.set_axisbelow(True)
    plt.grid()

    # pour IQBR de 17-100
    y_true = [y.item() *10 for y in y_true]
    y_pred = [y.item()*10 for y in y_pred]

    plt.scatter(y_true, y_pred, color = 'black', marker='.', s=150)
    # plt.errorbar(list(set(y_true)), means, stds, linestyle = 'None',color='black', marker = '.', capsize = 3)
    coef = np.polyfit(y_true, y_pred, 1)
    slope, intercept, r_value, p_value, std_err = stats.linregress(y_true, y_pred)
    plt.xlabel('RSQI - Ground truth', fontsize=18, labelpad=20)
    plt.ylabel('RSQI - Predicted',fontsize=18)
    plt.text(30, 95, f"R$^2$: {round(r_value**2,2)}",fontsize=18,bbox=dict(facecolor='white', edgecolor='white'))
    # plt.text(30,90, f"Écart type des erreurs: {np.round(std,2)}")
    plt.text(30,90, f"RMSE: {np.round(test_RMSE,2)}", fontsize=18,bbox=dict(facecolor='white', edgecolor='white'))
    plt.plot(y_true, intercept + (np.array(y_true) * slope), color='black')
    plt.xticks([17,30,40,50,60,70,80,90,100], fontsize=18)
    plt.yticks([17,30,40,50,60,70,80,90,100],fontsize=18)
    ax.set_aspect('equal', adjustable='box')
    # plt.savefig('checkpoints/' + name + '/scatter_' + name + '.tif')
    # show stuff
    # plt.show()
    print(f"Fold {num} : {np.round(test_RMSE,3)}, {round(r_value**2,3)}")


import numpy as np
import os

from collections import defaultdict
from PIL import Image
from utils.augmentation import compose_transforms
import random

import torch


class ValidDataset(torch.utils.data.Dataset):

    def __init__(self, root, views,indices=None, transform=compose_transforms, expand = False, test_ds = False):
        assert os.path.isdir(root)
        self.indices = indices
        self.expand = expand
        self.transform = transform
        self.views = views
        self.class_names = ['0', '1', '2', '3', '4']
        self.samples_vfs = []
        self.test_ds = test_ds

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

        if test_ds:
            self.length_ds = len(image_dict1)

        # on regarde le nombre de br dans chaque classe
        z= defaultdict(list)
        for key, samp in image_dict1.items():
            z[samp[0][1]].append(samp)
        # on retient le minimum
        m = min([len(z[r]) for r in z])
        # on recré la liste en conservant le même nombre dans chaque classe
        img_list = []
        for key, samp in z.items():

            # un paquet est attribué à un indice pour éviter les mélanges
            nombre_par_paquet = len(samp) // m
            for i in range(m):
                b = i*nombre_par_paquet
                img_list.append(samp[b:b+nombre_par_paquet])
        # del z
        del self.samples_vfs

        # nécessaire encore pour avoir des indices en pas de 1
        image_dict = defaultdict(list)
        num = 0
        for value in img_list:
            image_dict[num] = value
            num += 1

        # samples that will be used

        self.samples = image_dict

        # version avec les vues assemblées d'avance
        samples_list = []
        if self.expand:

            for key, samp in self.samples.items():
                if key in self.indices:

                    #on prend toutes les images de la bande riveraine
                    for s in samp:
                        samples_list.append(s)

            self.samples = samples_list

    def __len__(self):
        if self.test_ds:
            return self.length_ds
        else:
            return len(self.samples)

    def __getitem__(self, idx):
        if self.expand:
            sample = self.samples[idx] # returns [[path, classe]]
            random.shuffle(sample)
        else:
            sample = self.samples[idx]
        iqbr = np.float32(sample[0][0].split('_')[-1][:-4]) # la valeur dans le nom en enlevant le .tiff à la fin
        # if iqbr < 8.0:
        #     iqbr = iqbr**(1+((iqbr-8)/60))
        # iqbr = np.float32(sample[0][1])  # any second element
        # if iqbr == 0:
        #     iqbr = 0.8

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
        return image_stack, np.float32(iqbr)  # return tuple with class index as 2nd member


class TestDataset(torch.utils.data.Dataset):

    def __init__(self, root, views,indices=None, transform=compose_transforms, expand = False, test_ds = False):
        assert os.path.isdir(root)
        self.indices = indices
        self.expand = expand
        self.transform = transform
        self.views = views
        self.class_names = ['0', '1', '2', '3', '4']
        self.samples_vfs = []
        self.test_ds = test_ds

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

                #on prend toutes les images de la bande riveraine

                samples_list.append(samp)

            self.samples = samples_list

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.expand:
            sample = self.samples[idx] # returns [[path, classe]]
            random.shuffle(sample)
        else:
            sample = self.samples[idx]
        iqbr = np.float32(sample[0][0].split('_')[-1][:-4]) # la valeur dans le nom en enlevant le .tiff à la fin
        # if iqbr < 8.0:
        #     iqbr = iqbr**(1+((iqbr-8)/60))
        # iqbr = np.float32(sample[0][1])  # any second element
        # if iqbr == 0:
        #     iqbr = 0.8

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
        return image_stack, np.float32(iqbr)  # return tuple with class index as 2nd member
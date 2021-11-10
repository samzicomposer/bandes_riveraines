"""
Ce script génère les prédictions du modèles pour le test sur la variabilité inter-images.
Pour ce faire, il agence aléatoirement 4 images de deux bandes riveraines (total de 8).
400 prédictions sont faites mais c'est au choix.

"""

import numpy as np
import os

from collections import defaultdict
from PIL import Image
from scipy import stats

import torch
import torchvision
import shutil
import gdal
import torch.nn as nn
from torch.utils.data import Dataset
from models.mvdcnn import MVDCNN
from utils.augmentation import compose_transforms
import matplotlib.pyplot as plt
import random
from models.pytorch_resnet18 import resnet18

def CenterCrop(img, dim):
    width, height = img.shape[2], img.shape[1]
    crop_size= dim
    mid_x, mid_y = int(width / 2), int(height / 2)
    cw2, ch2 = int(crop_size / 2), int(crop_size / 2)
    crop_img = img[:, mid_y - ch2:mid_y + ch2, mid_x - cw2:mid_x + cw2]
    return crop_img

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
# rgb normalisation
stds = {1:5.8037, 2: 3.8330, 3:5.0484, 4:9.3750}
means = {1:49.0501, 2:74.7162, 3:86.9113, 4:109.2907}
# rg-nir
# stds = [0.0598, 0.0386, 0.1064] # nouveau jeu de données juin et full
# means =[0.2062, 0.2971,0.4101]  # nouveau jeu de données juin et full

global ident
ident = 0

class TestDataset(torch.utils.data.Dataset):

    def __init__(self, root, views, bands, indices=None, transform=compose_transforms, expand = False, test_ds = False):
        assert os.path.isdir(root)
        self.indices = indices
        self.expand = expand
        self.target_bands = bands
        self.transform = None
        self.views = views
        self.ident = ident
        self.class_names = ['0', '1', '2', '3', '4']
        self.samples_vfs = defaultdict(list)
        self.test_ds = test_ds
        all_samples = []

        for class_idx, class_set in enumerate(self.class_names):
            for class_name in class_set:
                class_dir = root + "/" + class_name
                if os.path.isdir(class_dir):
                    image_paths = [os.path.join(class_dir, p) for p in os.listdir(class_dir)]
                    for p in image_paths:
                        if p.endswith(".tif"):
                            self.samples_vfs[class_name].append(p)
                            all_samples.append(p)

        # dictionnaire classant les images par segment d'iqbr
        image_dict1 = {key: defaultdict(list) for key, item in self.samples_vfs.items()}
        for k in self.samples_vfs.keys():
            for img_paths in self.samples_vfs[k]:
                img = os.path.basename(img_paths)
                key = str(img.split('_')[0])
                image_dict1[k][key].append(img_paths)


        # nécessaire encore pour avoir des indices en pas de 1
        image_dict = defaultdict(list)
        num = 0
        for key, value in image_dict1.items():
            image_dict[num] = value
            num += 1

        # samples that will be used

        self.samples = image_dict

        # samples that will be used
        self.samples = image_dict1

        # version avec les vues assemblées d'avance
        samples_list1 = []
        samples_list2 = []
        if self.expand:

            class_combinations = []
            for i in classes:
                for j in classes:
                    if i != j:
                        class_combinations.append([i, j])

            for i in range(20):
                for c in class_combinations:
                    # premiere bande riveraine
                    keys1 = image_dict1[c[0]].keys() # les br de cette classe
                    riparian_strip1 = image_dict1[c[0]][random.sample(keys1,1)[0]]  # un choix aléatoire d'une br
                    part1 = random.choices(riparian_strip1, k=4)    # choix aléatoire d'images de cette br
                    samples_list1.append(part1)

                    keys2 = image_dict1[c[1]].keys()
                    riparian_strip2 = image_dict1[c[1]][random.sample(keys2,1)[0]]
                    part2 = random.choices(riparian_strip2, k=4)
                    samples_list2.append(part2)

            samples_list = [p1 + p2 for p1,p2 in zip(samples_list1,samples_list2)]

            self.samples = samples_list

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.expand:
            sample = self.samples[idx] # returns [[path, classe]]
            random.shuffle(sample)
        else:
            sample = self.samples[idx]
        try:
            iqbr = np.mean([np.float32(s.split('_')[-1][:-4]) for s in sample]) # la valeur dans le nom en enlevant le .tiff à la fin
        except:
            pass
        self.ident += 1

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
        return image_stack, np.float32(iqbr), self.ident, sample  # return tuple with class index as 2nd member

views = 1
batch_size = 128
out_folder =  r"D:\deep_learning\samples\jeux_separes\test\ecarts_compare\\"


pre_model = resnet18()
model = MVDCNN(pre_model,'RESNET', 1)

###### test du modèle ######

checkpoints = [
    # 'MVDCNN_2021-04-27-16_26_10' # 1 view
    'MVDCNN_2021-06-11-09_29_07' # 4 views
]

test_dataset = TestDataset(
        root=r"K:\deep_learning\samples\jeux_separes\test\all_values_obcfiltered3_rgbnir",
        views=1, bands = [1,2,3], expand=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,num_workers=0)

# si on calcule le mode
y_pred = defaultdict(list)

prob = np.array([94, 122, 43, 39, 79]) / 377.0
classes_int = [0, 1, 2, 3, 4]

###############################################################################


for c in checkpoints:
    print(c)

    checkpoint = torch.load(os.path.join(r'I:/annotation/checkpoints/', c, c + '.pth'))

    model.load_state_dict(checkpoint['model_state_dict'])
    if use_cuda:
        model = model.cuda()
    model.eval()

    test_ecart, test_total = [], 0
    y_true = []
    for minibatch in test_loader:

        images = minibatch[0]
        labels = minibatch[1]
        identifier = minibatch[2]
        samp_paths = minibatch[3]

        if use_cuda:
            images = images.to(device)
            labels = labels.to(device)
        with torch.no_grad():
            preds = model(images).view(-1)


        for p, l in zip(preds, labels):
            y_true.append(l)
            y_pred[str(views) + c].append(p)
            ecart = abs(p - l).item()

            # on copie les images dont les erreurs sont élevées
            # if ecart > 1.2:
            #
            #     for s in samp_paths:
            #         samplename = str(identifier.item()) +'_L' + str(np.round(labels.item(),2)) + '_P' + str(np.round(p.item(),2)) + '_' + os.path.basename(s[0])
            #         out_name = os.path.join(out_folder, str(np.round(ecart,2)),samplename)
            #
            #         if not os.path.isdir(os.path.dirname(out_name)):
            #             os.mkdir(os.path.dirname(out_name))
            #
            #         shutil.copyfile(s[0],out_name)

            test_ecart.append(ecart)

        test_total += labels.numel()
    test_ecart = np.array(test_ecart)*10
    test_accuracy = np.sqrt(np.mean(test_ecart**2))
    print(f"\nfinal test: RMSE = {test_accuracy:0.4f}\n")

##################### confusion matrix #####################

# conversion des prediction en matrice numpy
y_pred_np = []
for k in y_pred.keys():
    y_pred_np.append(np.array([i.cpu().numpy() for i in y_pred[k]]))
y_pred_np = np.array(y_pred_np)

# calcul de la moyenne des predictions
y_true = np.array([x.item() for x in y_true])*10
avg = np.average(y_pred_np, 0)*10
# avg[avg<17] = 17
# avg[avg>100] = 100
# y_true[y_true==40] = 80
avg_error = np.round(np.mean(abs(avg-y_true)),3)
# classe ayant la plus haute moyenne de prediction
rounded_pred = np.round(avg, 0)

mean = np.mean(abs(avg-y_true))
std = np.std(abs(avg-y_true))
RMSE = np.sqrt(np.mean((avg-y_true)**2))

means = []
stds = []
for cl in set(y_true):
    masque = np.array(y_true)!=cl
    z = np.ma.array(avg,mask=masque)
    means.append(np.mean(z))
    stds.append(np.std(z - y_true))

fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(111)
ax.set_axisbelow(True)
plt.grid()

plt.scatter(y_true, avg, color = 'black', marker='.', s=150)
# plt.errorbar(list(set(y_true)), means, stds, linestyle = 'None',color='black', marker = '.', capsize = 3)
coef = np.polyfit(y_true, avg, 1)
slope, intercept, r_value, p_value, std_err = stats.linregress(y_true, avg)
plt.xlabel('Actual RSQI', fontsize=18, labelpad=20)
plt.ylabel('Predicted RSQI',fontsize=18)
plt.text(30, 95, f"R$^2$: {round(r_value**2,2)}",fontsize=18,bbox=dict(facecolor='white', edgecolor='white'))
# plt.text(30,90, f"Écart type des erreurs: {np.round(std,2)}")
plt.text(30,90, f"RMSE: {np.round(RMSE,2)}", fontsize=18,bbox=dict(facecolor='white', edgecolor='white'))
plt.plot(y_true, intercept + (np.array(y_true) * slope), color='black')
plt.xticks([20,30,40,50,60,70,80,90,100], fontsize=18)
plt.yticks([20,30,40,50,60,70,80,90,100],fontsize=18)
ax.set_aspect('equal', adjustable='box')
# plt.savefig('checkpoints/' + name + '/scatter_' + name + '.tif')
# show stuff
plt.show()

import numpy as np
import os
from collections import defaultdict
from PIL import Image

import torch
import torchvision
from models.pytorch_resnet18 import resnet18

from torch.utils.data import Dataset
from utils.valid_dataset_rgbnir import TestDataset
from models.mvdcnn import MVDCNN
from scipy import stats

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


views = 1
batch_size = 1
crop_size = 45

pre_model = torchvision.models.vgg16_bn()

model = MVDCNN(pre_model,'VGG16', 1)

###### test du modèle ######

checkpoints = [

    'MVDCNN_2021-04-28-16_11_18'
]

test_dataset = TestDataset(
        r"D:/deep_learning/samples/jeux_separes/test/all_values_v2_rgbnir/", bands = [1,2,3],
        views=16,  expand=True)

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

    for i in range(1):

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

            # la classe moyenne predite par les assemblages de vues (toute la bande)
            avg_prediction = [torch.mean(preds)]

            for p, l in zip(avg_prediction, labels):
                y_true.append(l)
                y_pred[str(i) + str(views) + c].append(p)



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

fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(111)
ax.set_axisbelow(True)
plt.grid()

plt.scatter(y_true, avg, color = 'black', marker='.', s=150)
# plt.errorbar(list(set(y_true)), means, stds, linestyle = 'None',color='black', marker = '.', capsize = 3)
coef = np.polyfit(y_true, avg, 1)
slope, intercept, r_value, p_value, std_err = stats.linregress(y_true, avg)
print(f"slope: {slope}, intercept: {intercept}")
plt.xlabel('RSQI - Ground truth', fontsize=18, labelpad=20)
plt.ylabel('RSQI - Predicted',fontsize=18)
plt.text(30, 95, f"R$^2$: {round(r_value**2,2)}",fontsize=18,bbox=dict(facecolor='white', edgecolor='white'))
# plt.text(30,90, f"Écart type des erreurs: {np.round(std,2)}")
plt.text(30,90, f"RMSE: {np.round(rmse,2)}", fontsize=18,bbox=dict(facecolor='white', edgecolor='white'))
plt.plot(y_true, intercept + (np.array(y_true) * slope), color='black')
plt.xticks([17,30,40,50,60,70,80,90,100], fontsize=18)
plt.yticks([17,30,40,50,60,70,80,90,100],fontsize=18)
ax.set_aspect('equal', adjustable='box')

# plt.xticks(np.arange(min(y_true), max(y_true)+1, 0.5))
# plt.xlim(1.7,10.0)
plt.show()

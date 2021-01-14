import numpy as np
import os
import time
from datetime import datetime

from collections import defaultdict
from PIL import Image

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

class MyDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        # if self.transform:
        #     x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

def get_class_distribution(dataset_obj, idx_to_class):
    count_dict = {k: 0 for k, v in idx_to_class.items()}

    for element in dataset_obj:
        try:
            y_lbl = element[1]
            # y_lbl = idx_to_class[int(y_lbl)]
            count_dict[y_lbl] += 1
        except:
            pass

    return count_dict

use_cuda = torch.cuda.is_available()

# Seeds déterministes
seed = 30
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


# for confusion matrix
# classes = ['0','1','2','3','4','5']
classes = ['0', '1', '2', '3', '4']
# with r-g-slope
# means = [0.1906, 0.2918, 0.1247]
# stds = [0.0442, 0.0330, 0.1621]
# with r-n-slope
# means = [0.1906, 0.4234,0.1247]
# stds = [0.0442, 0.1070, 0.1621]
# # with r-g-nir
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
                    # samples_list.append(samp)
                    # for i in range(10):
                    #     samples_list.append(random.choices(samp, k=self.views))
                    if len(samp)//self.views < 1:
                        samples_list.append(random.choices(samp, k=self.views))
                    else:
                        for i in range(len(samp)//self.views):
                            b = i*self.views
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


views = 16
batch_size = 512
epochs = 35
crop_size = 45

base_transforms = torchvision.transforms.Compose([torchvision.transforms.RandomApply([
    torchvision.transforms.RandomRotation(45),
], p=0.9),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.RandomVerticalFlip(p=0.5),
    torchvision.transforms.ColorJitter(0.2,0.2,0.2),
    # torchvision.transforms.CenterCrop(45),
    torchvision.transforms.RandomCrop(crop_size),

    torchvision.transforms.ToTensor(),  # rescale de 0 à 1, de là les valeurs ci-dessous
    torchvision.transforms.Normalize(mean=(means[0], means[1], means[2]), std=(stds[0], stds[1], stds[2]))
])
# test_transforms = torchvision.transforms.Compose([
#     # torchvision.transforms.CenterCrop(45),
#     torchvision.transforms.RandomCrop(crop_size),
#     torchvision.transforms.ToTensor(),  # rescale de 0 à 1, de là les valeurs ci-dessous
#     torchvision.transforms.Normalize(mean=(means[0], means[1], means[2]), std=((stds[0], stds[1], stds[2])))
# ])


dataset = PleiadesDataset(root=r"D:/deep_learning/samples/jeux_separes/train/iqbr_cl_covabar_10mdist_july/", views=1, transform=base_transforms, indices = None)
temp_test_dataset = PleiadesDataset(root=r"D:/deep_learning/samples/jeux_separes/test/iqbr_cl_covabar_10mdist_july/", views=1, transform=base_transforms, indices = None)
# train_dataset = PleiadesDataset(root=r"D:/deep_learning/samples/iqbr/train", views=3, transform=base_transforms)
# val_dataset = PleiadesDataset(root=r"D:/deep_learning/samples/iqbr/val", views=3, transform=base_transforms)
# test_dataset = PleiadesDataset(root=r"D:/deep_learning/samples/iqbr/test", views=3, transform=base_transforms)



# separation du jeu de données en train, val, test
# dataset_size = len(dataset)
# train_subset, val_subset, test_subset = torch.utils.data.random_split(dataset,
#                                             (int(0.8 * dataset_size), int(0.1 * dataset_size), int(dataset_size - (int(0.8 * dataset_size) + int(0.1 * dataset_size)))))

# creation de datasets séparés, sinon les transformations ne suivent pas
# train_dataset = MyDataset(train_subset)
# val_dataset = MyDataset(val_subset)
# test_dataset = MyDataset(test_subset)

# indice et nom des classes
idx2class = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4'}
# nombre d'image par classe
# class_distribution = get_class_distribution(dataset, idx2class)
# # probabilité pour chacune des classes
# class_sample_count = [num for classe, num in class_distribution.items()]
# class_weights = 1. / torch.Tensor(class_sample_count)
# poids par image classe covabar
# class_weights = torch.Tensor([4.5844e-05, 4.3254e-05, 2.4432e-04, 4.1477e-04, 1.0662e-04])
# class_weights=torch.Tensor([0.0009, 0.0007, 0.0031, 0.0056, 0.0016])
# poids selon images single, covabar full
class_weights = 10./torch.Tensor([20739,21797,3684*0.75,1978*1.2,8483])
print(class_weights)

# liste des labels des images retenues pour ce dataset
# target_list = torch.tensor([i[1] for i in train_dataset])
# # probabilité de chaque image d'être pigée selon son label
# class_weights_all = class_weights[target_list]
# # le sampler va piger une image selon les probabilités de la classe, une image peut être pigée plusieurs fois
# train_weighted_sampler = torch.utils.data.sampler.WeightedRandomSampler(class_weights_all, len(class_weights_all), replacement=True)
#
# class_distribution = get_class_distribution(val_dataset, idx2class)
# class_sample_count = [num for classe, num in class_distribution.items()]
# class_weights = 1. / torch.Tensor(class_sample_count)
# # target_list = torch.tensor([val_dataset.subset.dataset[i][1] for i in val_dataset.subset.indices])
# target_list = torch.tensor([i[1] for i in val_dataset])
# class_weights_all = class_weights[target_list]
# val_weighted_sampler = torch.utils.data.sampler.WeightedRandomSampler(class_weights_all, len(class_weights_all), replacement=True)
#
# class_distribution = get_class_distribution(test_dataset, idx2class)
# class_sample_count = [num for classe, num in class_distribution.items()]
# class_weights = 1. / torch.Tensor(class_sample_count)
# target_list = torch.tensor([i[1] for i in  test_dataset])
# class_weights_all = class_weights[target_list]
# test_weighted_sampler = torch.utils.data.sampler.WeightedRandomSampler(class_weights_all, len(class_weights_all), replacement=True)


# Separation du jeu de données en train, val, tst
sample_idxs = np.random.permutation(len(dataset)).tolist()
# train_sample_count, valid_sample_count = int(0.8 * len(sample_idxs)), int(0.1 * len(sample_idxs))
train_sample_count = int(0.90 * len(sample_idxs))
train_sampler = torch.utils.data.sampler.SubsetRandomSampler(sample_idxs[0:train_sample_count])
# valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(
#     sample_idxs[train_sample_count:(train_sample_count + valid_sample_count)])
valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(sample_idxs[train_sample_count:])


train_dataset = PleiadesDataset(root=r"D:/deep_learning/samples/jeux_separes/train/iqbr_cl_covabar_10mdist_july/", views=views, transform=base_transforms, indices = train_sampler.indices, expand=True)
val_dataset = PleiadesDataset(root=r"D:/deep_learning/samples/jeux_separes/train/iqbr_cl_covabar_10mdist_july/", views=views, transform=base_transforms, indices = valid_sampler.indices, expand=True)

test_indices = np.random.permutation(len(temp_test_dataset)).tolist()
test_sampler = torch.utils.data.sampler.SubsetRandomSampler(test_indices)
test_dataset = PleiadesDataset(root=r"D:/deep_learning/samples/jeux_separes/test/iqbr_cl_covabar_10mdist_july/", views=views, transform=base_transforms, indices = test_sampler.indices, expand=True)

# train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, sampler=train_sampler, num_workers=0, pin_memory=True)
# valid_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=0,pin_memory=True)
# test_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, sampler=test_sampler, num_workers=0,pin_memory=True)

train_rsampler = torch.utils.data.sampler.RandomSampler(train_dataset)
val_rsampler = torch.utils.data.sampler.RandomSampler(val_dataset)
test_rsampler = torch.utils.data.sampler.RandomSampler(test_dataset)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,sampler=train_rsampler,  num_workers=0)
valid_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, sampler = val_rsampler,num_workers=0)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,sampler = test_rsampler, num_workers=0)

# counts = defaultdict(list)
# for i in test_dataset.samples:
#     counts[i[0][1]].append(len(i))


pre_model = torchvision.models.resnet18(pretrained=True)

for param in pre_model.parameters():
    param.requires_grad = False
# for param in pre_model.layer2.parameters():
#     param.requires_grad = True
for param in pre_model.layer3.parameters():
    param.requires_grad = True
for param in pre_model.layer4.parameters():
    param.requires_grad = True

model = MVDCNN(pre_model, len(classes))

# c = 'MVDCNN_2020-12-01-22_10_42'
# checkpoint = torch.load(os.path.join(r'I:/annotation/checkpoints/', c, c + '.pth'))
# model.load_state_dict(checkpoint['model_state_dict'])
#
# for p in model.classifier.parameters():
#     p.requires_grad = True

# le taux d'apprentissage qui permet de contrôler la taille du pas de mise à jour des paramètres
learning_rate = 1e-4  # à ajuster au besoin!

# le momentum (ou inertie) de l'optimiseur (SGD)
momentum = 0.9  # à ajuster au besoin!

# pénalité de régularisation (L2) sur les paramètres
weight_decay = 1e-4  # à ajuster au besoin!

# nombre d'epochs consécutives avec taux d'apprentissage fixe
lr_step_size = 20  # à ajuster au besoin!

# après le nombre d'epoch ci-haut, réduire le taux d'apprentissage par ce facteur
lr_step_gamma = 0.5  # à ajuster au besoin!

# instanciation de la fonction de perte sous forme d'un objet
# class_weights = [1.,1.,1.,1.,1.]
class_weight = torch.FloatTensor(class_weights).to(device)

loss_function = nn.CrossEntropyLoss(weight=class_weight)

# instanciation de l'optimiseur (SGD); on lui fournit les paramètres du modèle qui nécessitent une MàJ
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=learning_rate, weight_decay=weight_decay)

# instanciation du 'Scheduler' permettant de modifier le taux d'apprentissage en fonction de l'epoch
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_step_gamma)

if use_cuda:
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)

train_losses, valid_losses = [], []  # pour l'affichage d'un graphe plus tard
train_accuracies, valid_accuracies = [], []  # pour l'affichage d'un graphe plus tard
best_model_state, best_model_accuracy = None, None  # pour le test final du meilleur modèle
last_print_time = time.time()

### Entrainement du réseau ###

for epoch in range(epochs):

    # dataset.transforms = base_transforms
    # dataset.transform = base_transforms

    train_loss = 0  # on va accumuler la perte pour afficher une courbe
    train_correct, train_total = 0, 0  # on va aussi accumuler les bonnes/mauvaises classifications
    model.train()

    for batch_idx, batch in enumerate(train_loader):

        if time.time() - last_print_time > 10:
            last_print_time = time.time()
            print(f"\ttrain epoch {epoch + 1}/{epochs} @ iteration {batch_idx + 1}/{len(train_loader)}...")

        images, labels = batch
        if use_cuda:
            images = images.to(device)
            labels = labels.to(device)

        # display_batch_size = min(4, batch_size)
        # fig = plt.figure(figsize=(7, 3))
        # for ax_idx in range(display_batch_size):
        #   ax = fig.add_subplot(1, 4, ax_idx + 1)
        #   ax.grid(False)
        #   ax.set_xticks([])
        #   ax.set_yticks([])
        #   class_name = str(labels[ax_idx].item())
        #   ax.set_title(class_name)
        #   display = images[ax_idx,0, ...].numpy()
        #   display = display.transpose((1, 2, 0))  # CxHxW => HxWxC (tel que demandé par matplotlib)
        #   mean = np.array([0.485, 0.456, 0.406])  # nécessaire pour inverser la normalisation
        #   std = np.array([0.229, 0.224, 0.225])  # nécessaire pour inverser la normalisation
        #   display = (std * display + mean)  # on inverse la normalisation
        #   display = np.clip(display, 0, 1)  # on élimine les valeurs qui sortent de l'intervalle d'affichage
        #   plt.imshow(display)
        #
        # plt.show()

        preds = model(images)

        optimizer.zero_grad()
        loss = loss_function(preds, labels)
        loss.backward()
        optimizer.step()

        # ici, on rafraîchit nos métriques d'entraînement
        train_loss += loss.item()  # la fonction '.item()' retourne un scalaire à partir du tenseur
        train_correct += (preds.topk(k=1, dim=1)[1].view(-1) == labels).nonzero().numel()
        train_total += labels.numel()

    # on calcule les métriques globales pour l'epoch
    train_loss = train_loss / len(train_loader)
    train_losses.append(train_loss)
    train_accuracy = train_correct / train_total
    train_accuracies.append(train_accuracy)

    last_print_time = time.time()
    print(f"train epoch {epoch + 1}/{epochs}: loss={train_loss:0.4f}, accuracy={train_accuracy:0.4f}")

    # deuxième étape: on utilise le 'valid_loader' pour évaluer le modèle
    valid_loss = 0  # on va accumuler la perte pour afficher une courbe
    valid_correct, valid_total = 0, 0  # on va aussi accumuler les bonnes/mauvaises classifications

    model.eval()  # mise du modèle en mode "évaluation" (utile pour certaines couches...)
    # dataset.transforms = test_transforms
    # dataset.transform = test_transforms

    # boucle semblable à celle d'entraînement, mais on utilise l'ensemble de validation
    # si on calcule le mode
    y_pred = defaultdict(list)
    for batch_idx, minibatch in enumerate(valid_loader):
        for i in range(5):
            if time.time() - last_print_time > 10:
                last_print_time = time.time()
                print(f"\tvalid epoch {epoch + 1}/{epochs} @ iteration {batch_idx + 1}/{len(valid_loader)}...")

            images = minibatch[0]  # rappel: en format BxCxHxW
            labels = minibatch[1]  # rappel: en format Bx1

            if use_cuda:
                images = images.to(device)
                labels = labels.to(device)

            # ici, on n'a plus besoin de l'optimiseur, on cherche seulement à évaluer
            with torch.no_grad():  # utile pour montrer explicitement qu'on n'a pas besoin des gradients
                preds = model(images)
                loss = loss_function(preds, labels)

            valid_loss += loss.item()
            valid_correct += (preds.topk(k=1, dim=1)[1].view(-1) == labels).nonzero().numel()
            valid_total += labels.numel()

    # on calcule les métriques globales pour l'epoch
    valid_loss = valid_loss / len(valid_loader)
    valid_losses.append(valid_loss)
    valid_accuracy = valid_correct / valid_total
    valid_accuracies.append(valid_accuracy)

    # on mémorise les poids si le modèle surpasse le meilleur à date
    if best_model_accuracy is None or valid_accuracy > best_model_accuracy:
        best_model_state = model.state_dict()
        best_model_accuracy = valid_accuracy
        best_train_accuracy = train_accuracy

    last_print_time = time.time()
    print(f"valid epoch {epoch + 1}/{epochs}: loss={valid_loss:0.4f}, accuracy={valid_accuracy:0.4f}")
    print("----------------------------------------------------\n")
    # scheduler.step()
    train_dataset = PleiadesDataset(root=r"D:/deep_learning/samples/jeux_separes/train/iqbr_cl_covabar_10mdist_july/", views=views,
                                    transform=base_transforms, indices=train_sampler.indices, expand=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=train_rsampler,
                                               num_workers=0)
    val_dataset = PleiadesDataset(root=r"D:/deep_learning/samples/jeux_separes/train/iqbr_cl_covabar_10mdist_july/", views=views,
                                  transform=base_transforms, indices=valid_sampler.indices, expand=True)
    valid_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, sampler=val_rsampler,
                                               num_workers=0)
    # test_dataset = PleiadesDataset(root=r"D:/deep_learning/samples/jeux_separes/test/iqbr_cl_covabar_10mdist_july/",
    #                                views=views, transform=base_transforms, indices=test_sampler.indices,
    #                                expand=True)
    # test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, sampler=test_rsampler,
    #                                           num_workers=0)
###### test du modèle ######
# for i in range(3):
# test_dataset = PleiadesDataset(root=r"D:/deep_learning/samples/jeux_separes/test/iqbr_cl_covabar_10mdist_july/",
#                                views=views, transform=base_transforms, indices=test_sampler.indices,
#                                expand=True)
# test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, sampler=test_rsampler,
#                                           num_workers=0)
test_correct, test_total = 0, 0
y_true, y_pred = [], []
model.load_state_dict(best_model_state)
model.eval()  # mise du modèle en mode "évaluation" (utile pour certaines couches...)

# dataset.transforms = test_transforms
# dataset.transform = test_transforms

for minibatch in test_loader:
    images = minibatch[0]  # rappel: en format BxCxHxW
    labels = minibatch[1]  # rappel: en format Bx1
    # images = torch.stack([item for sublist in images for item in sublist])
    if use_cuda:
        images = images.to(device)
        labels = labels.to(device)
    with torch.no_grad():
        preds = model(images)

    top_preds = preds.topk(k=1, dim=1)[1].view(-1)
    for p, l in zip(top_preds, labels):
        y_true.append(l)
        y_pred.append(p)
    test_correct += (top_preds == labels).nonzero().numel()
    test_total += labels.numel()

test_accuracy = test_correct / test_total
print(f"\nfinal test: accuracy={test_accuracy:0.4f}\n")
#
# conf_class_map = {idx: name for idx, name in enumerate(dataset.class_names)}
# y_true = [conf_class_map[idx.item()] for idx in y_true]
# y_pred = [conf_class_map[idx.item()] for idx in y_pred]
# name = 'test'
# cm = confusion_matrix(y_true, y_pred)
# plot_confusion_matrix(cm, name, normalize=True,
#                       target_names=classes,
#                       title="Confusion Matrix", show=True, save=False)

######### name of file to be used ########
date_time = str(datetime.now().date()) + '-' + str(datetime.now().strftime('%X')).replace(':', '_')
if use_cuda:
    if torch.cuda.device_count() > 1:
        model_name = str(model.module._get_name())
    else:
        model_name = str(model._get_name())
else:
    model_name = str(model._get_name())
name = model_name + '_' + date_time

##################### confusion matrix #####################

conf_class_map = {idx: name for idx, name in enumerate(dataset.class_names)}
y_true = [conf_class_map[idx.item()] for idx in y_true]
y_pred = [conf_class_map[idx.item()] for idx in y_pred]

class_report = classification_report(y_true, y_pred, target_names=classes)
cm = confusion_matrix(y_true, y_pred)

##################### graphs #####################

x = range(1, epochs + 1)

fig = plt.figure(figsize=(8, 8))

ax = fig.add_subplot(2, 1, 1)
ax.plot(x, train_losses, label='train')
ax.plot(x, valid_losses, label='valid')
ax.set_xlabel('# epochs')
ax.set_ylabel('Loss')
ax.legend()

ax = fig.add_subplot(2, 1, 2)
ax.plot(x, train_accuracies, label='train')
ax.plot(x, valid_accuracies, label='valid')
x_test = valid_accuracies.index(best_model_accuracy) + 1
ax.scatter(x_test, test_accuracy, color='red', label='test')
ax.set_xlabel('# epochs')
ax.set_ylabel('Accuracy')
ax.legend()

##################### logs  #####################
print(class_report)
#
# graphs
os.mkdir('checkpoints/' + name)
plt.savefig('checkpoints/' + name + '/graph_' + name + '.jpg')
#
# show stuff
plt.show()
#
#
# classification report
text_file = open('checkpoints/' + name + '/class_report_' + name + '.txt', "w")
n = text_file.write(class_report)
text_file.close()
#
# # confusion matrix
plot_confusion_matrix(cm, name, normalize=True,
                      target_names=classes,
                      title="Confusion Matrix", show=True, save=True)
# logs
window_size = crop_size
file = 'iqbr_mvdcnn.csv'
comment =  f'NEW 10M DS, TEST AS VALIDLOADER, L4 et L3 * 1, (train val = 99-1), poids single img (cl2/0.75, cl3/1.2), classificateur 512, MEAN POOL, colorjitter 0.2, classes COV, AUGMENT ALL + RESHUFFLE T_DS et VAL_DS, views et segments rives isoles par dataset (une fois chaque image),random sampler, {views} views, seed = {seed}'
logger(date_time, model_name, model, dataset, window_size, epochs, learning_rate, batch_size, weight_decay, train_loss,
       valid_loss, 0,
       best_train_accuracy, best_model_accuracy, test_accuracy, name, file, save_checkpoint=True,
       pretrained='ImageNet', comment=comment)

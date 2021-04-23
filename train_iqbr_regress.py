import numpy as np
import os
import time
from datetime import datetime
from utils.get_model_name import get_model_name

from collections import defaultdict
from PIL import Image
from utils.valid_dataset import ValidDataset, TestDataset
from scipy import stats

import torch
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset
from models.mvdcnn import MVDCNN
from models.pytorch_resnet18 import resnet18

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
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

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

if not use_cuda:
    print("WARNING: PYTORCH COULD NOT LOCATE ANY AVAILABLE CUDA DEVICE.\n")
else:
    device = torch.device("cuda:0")
    print("All good, a GPU is available.")

# Seeds déterministes
seeds = np.random.randint(1,1000, size=10)
# seeds = np.append(seeds, [1,51,24])
# seeds = [1,51,24]
# for seed in seeds:


seed = 48
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True


# for confusion matrix
# classes = ['0','1','2','3','4','5']
classes = ['1.7-2.1', '2.1-4.0', '4.0-6.0', '6.0-8.0', '8.0-10']
# with r-g-slope
# means = [0.1906, 0.2918, 0.1247]
# stds = [0.0442, 0.0330, 0.1621]
# with r-n-slope
# means = [0.1906, 0.4234,0.1247]
# stds = [0.0442, 0.1070, 0.1621]
# with rgb
stds = [0.0598, 0.0386, 0.0252] # nouveau jeu de données
means =[0.2062, 0.2971,0.3408]  # nouveau jeu de données
# # with r-g-nir
# stds = [0.0598, 0.0386, 0.1064] # nouveau jeu de données
# means =[0.2062, 0.2971,0.4101]  # nouveau jeu de données

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
                    # random.shuffle(samp)
                    # samples_list.append(samp)
                    # for i in range(10):
                    #     samples_list.append(random.choices(samp, k=self.views))

                    number = 1
                    # choix de la BR dans le paquet
                    for i in range(number):
                        samp1 = random.sample(samp,1)[0]
                        if len(samp1)//self.views < 1:
                            samples_list.append(random.choices(samp1, k=self.views))
                        else:
                            samples_list.append(random.sample(samp1, k=self.views))
                            # for i in range(len(samp)//self.views):
                            #     b = i*self.views
                            #     samples_list.append(samp[b:b+self.views])

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

views = 4
batch_size = 128
epochs = 100
crop_size = 45
permutations = 3

base_transforms = torchvision.transforms.Compose([torchvision.transforms.RandomApply([
    torchvision.transforms.RandomRotation(45),
], p=0.9),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.RandomVerticalFlip(p=0.5),
    torchvision.transforms.ColorJitter(0.2,0.2,0.2),
    torchvision.transforms.CenterCrop(crop_size),
    # torchvision.transforms.Resize(224),

    torchvision.transforms.ToTensor(),  # rescale de 0 à 1, de là les valeurs ci-dessous
    torchvision.transforms.Normalize(mean=(means[0], means[1], means[2]), std=(stds[0], stds[1], stds[2]))
])
test_transforms = torchvision.transforms.Compose([torchvision.transforms.RandomApply([
    torchvision.transforms.RandomRotation(45),
], p=0.9),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.RandomVerticalFlip(p=0.5),
    torchvision.transforms.ColorJitter(0.2,0.2,0.2),
    torchvision.transforms.CenterCrop(crop_size),
    # torchvision.transforms.Resize(224),

    torchvision.transforms.ToTensor(),  # rescale de 0 à 1, de là les valeurs ci-dessous
    torchvision.transforms.Normalize(mean=(means[0], means[1], means[2]), std=(stds[0], stds[1], stds[2]))
])

dataset = PleiadesDataset(root=r"D:/deep_learning/samples/jeux_separes/train/all_values_v2_rgb/", views=1, transform=base_transforms, indices = None, expand=False)
temp_test_dataset = TestDataset(root=r"D:/deep_learning/samples/jeux_separes/test/all_values_v2_rgb/", views=1, transform=base_transforms, indices = None, test_ds=True)
# train_dataset = PleiadesDataset(root=r"D:/deep_learning/samples/iqbr/train", views=3, transform=base_transforms)
# val_dataset = PleiadesDataset(root=r"D:/deep_learning/samples/iqbr/val", views=3, transform=base_transforms)
# test_dataset = PleiadesDataset(root=r"D:/deep_learning/samples/iqbr/test", views=3, transform=base_transforms)

# indice et nom des classes
idx2class = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4'}


# Separation du jeu de données en train, val, tst
sample_idxs = np.random.permutation(len(dataset)).tolist()
# train_sample_count, valid_sample_count = int(0.8 * len(sample_idxs)), int(0.1 * len(sample_idxs))
train_sample_count = int(0.80 * len(sample_idxs))
train_sampler = torch.utils.data.sampler.SubsetRandomSampler(sample_idxs[0:train_sample_count])
# valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(
#     sample_idxs[train_sample_count:(train_sample_count + valid_sample_count)])
valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(sample_idxs[train_sample_count:])

train_dataset = PleiadesDataset(root=r"D:/deep_learning/samples/jeux_separes/train/all_values_v2_rgb/", views=views, transform=base_transforms, indices = train_sampler.indices, expand=True)
val_dataset = ValidDataset(root=r"D:/deep_learning/samples/jeux_separes/train/all_values_v2_rgb/", views=views, transform=test_transforms, indices = valid_sampler.indices, expand=True)

test_indices = np.random.permutation(len(temp_test_dataset)).tolist()
test_sampler = torch.utils.data.sampler.SubsetRandomSampler(test_indices)
test_dataset = TestDataset(root=r"D:/deep_learning/samples/jeux_separes/test/all_values_v2_rgb/", views=views, transform=test_transforms, indices = None, expand=True, test_ds=True)

train_rsampler = torch.utils.data.sampler.RandomSampler(train_dataset)
val_rsampler = torch.utils.data.sampler.RandomSampler(val_dataset)
test_rsampler = torch.utils.data.sampler.RandomSampler(test_dataset)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,sampler=train_rsampler,  num_workers=0)
# pas de sampling aléatoire car nous avons de multiples passes en val et test, on conserve le même ordre
valid_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=1, num_workers=0)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, num_workers=0)

counts = {0:0, 1:0, 2:0, 3:0, 4:0}
for idx in train_dataset.indices:
    try:
          counts[train_dataset.samples[idx][0][1]]+=len(train_dataset.samples[idx])
    except:
        pass

pre_model = torchvision.models.resnet18(pretrained=True)

for param in pre_model.parameters():
    param.requires_grad = False
# for param in pre_model.layer2.parameters():
#     param.requires_grad = True
# for param in pre_model.layer3.parameters():
#     param.requires_grad = True
for param in pre_model.layer4.parameters():
    param.requires_grad = True

model = MVDCNN(pre_model, 1)

# c = 'MVDCNN_2020-12-01-22_10_42'
# checkpoint = torch.load(os.path.join(r'I:/annotation/checkpoints/', c, c + '.pth'))
# model.load_state_dict(checkpoint['model_state_dict'])
#
# for p in model.classifier.parameters():
#     p.requires_grad = True

# le taux d'apprentissage qui permet de contrôler la taille du pas de mise à jour des paramètres
learning_rate = 5e-4 # à ajuster au besoin!

# le momentum (ou inertie) de l'optimiseur (SGD)
momentum = 0.9  # à ajuster au besoin!

# pénalité de régularisation (L2) sur les paramètres
weight_decay = 1e-5  # à ajuster au besoin!

# nombre d'epochs consécutives avec taux d'apprentissage fixe
lr_step_size = 15  # à ajuster au besoin!

# après le nombre d'epoch ci-haut, réduire le taux d'apprentissage par ce facteur
lr_step_gamma = 0.2  # à ajuster au besoin!

loss_function = nn.MSELoss()

# instanciation de l'optimiseur (SGD); on lui fournit les paramètres du modèle qui nécessitent une MàJ
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=learning_rate, weight_decay=weight_decay)

# instanciation du 'Scheduler' permettant de modifier le taux d'apprentissage en fonction de l'epoch
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_step_gamma)

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

        preds = model(images).view(-1)

        optimizer.zero_grad()
        loss = loss_function(preds, labels)
        loss.backward()
        optimizer.step()

        # ici, on rafraîchit nos métriques d'entraînement
        train_loss += loss.item()  # la fonction '.item()' retourne un scalaire à partir du tenseur
        # train_correct += (preds.topk(k=1, dim=1)[1].view(-1) == labels).nonzero().numel()
        # train_total += labels.numel()

    # on calcule les métriques globales pour l'epoch
    train_loss = train_loss / len(train_loader)
    train_losses.append(train_loss)
    # train_accuracy = train_correct / train_total
    # train_accuracies.append(train_accuracy)

    last_print_time = time.time()
    print(f"train epoch {epoch + 1}/{epochs}: loss={train_loss:0.4f}")

    ######### validation ##########

    model.eval()

    # si on calcule le mode
    y_pred = defaultdict(list)
    for i in range(permutations):

        valid_ecart, valid_total = [], 0
        y_true = []
        for minibatch in valid_loader:

            view_features = []

            # on separe la bande en groupes selon le nombre de vues
            range_v = (len(minibatch[0][0]) // 1)
            for v in range(range_v):
                # on assemble les images selon le nombre de vues
                coord = v * 1
                view_features.append(minibatch[0][0][coord:coord + 1])
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
            avg_prediction = [avg_prediction[0].cpu()]

            for p, l in zip(avg_prediction, labels):
                y_true.append(l)
                y_pred[str(i) + str(views)].append(p)
                valid_ecart.append(abs(p - l).item())
            # on arrondi la moyenne à la classe la plus près pour le test de classification

            valid_total += labels.numel()
        valid_ecart = np.array(valid_ecart)
        test_accuracy = np.mean(valid_ecart ** 2)
        print(f"\nValidation test {i}: MSE = {test_accuracy:0.4f}\n")

    ##################### calcul des moyennes #####################

    # conversion des prediction en matrice numpy
    y_pred_np = []
    for k in y_pred.keys():
        y_pred_np.append(np.array([i.numpy() for i in y_pred[k]]))
    y_pred_np = np.array(y_pred_np)

    # calcul de la moyenne des predictions
    y_true = np.array([x.item() for x in y_true])
    avg = np.average(y_pred_np, 0)
    # avg[avg<17] = 17
    # avg[avg > 100] = 100
    # y_true[y_true==40] = 80
    avg_error = np.round(np.mean(abs(avg - y_true)), 3)
    # classe ayant la plus haute moyenne de prediction
    rounded_pred = np.round(avg, 0)

    mean = np.mean(abs(avg - y_true))
    std = np.std(abs(avg - y_true))
    MSE = np.mean((avg - y_true) ** 2)
    valid_loss = MSE
    valid_losses.append(valid_loss)

    print(f"Validation average MSE: {np.round(MSE,2)}")

    #######################

    # on mémorise les poids si le modèle surpasse le meilleur à date
    if best_model_accuracy is None or valid_loss < best_model_accuracy:
        best_model_state = model.state_dict()
        best_model_accuracy = valid_loss
        best_epoch = epoch+1
        # best_train_accuracy = train_accuracy

    scheduler.step()

    # if (epoch+1) % 10 == 0:
    train_dataset = PleiadesDataset(root=r"D:/deep_learning/samples/jeux_separes/train/all_values_v2_rgb/", views=views,
                                    transform=base_transforms, indices=train_sampler.indices, expand=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=train_rsampler,
                                               num_workers=0)

###### test du modèle ######

model.load_state_dict(best_model_state)
model.eval()  # mise du modèle en mode "évaluation" (utile pour certaines couches...)

# pour de multiple passes
y_pred = defaultdict(list)
for i in range(permutations):

    test_ecart, test_total = [], 0
    y_true = []
    for minibatch in test_loader:

        view_features = []

        # on separe la bande en groupes selon le nombre de vues
        range_v = (len(minibatch[0][0]) // 1)
        for v in range(range_v):
            # on assemble les images selon le nombre de vues
            coord = v * 1
            view_features.append(minibatch[0][0][coord:coord + 1])
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
        avg_prediction = [avg_prediction[0].cpu()]

        for p, l in zip(avg_prediction, labels):
            y_true.append(l)
            y_pred[str(i) + str(views)].append(p)
            test_ecart.append(abs(p - l).item())
        # on arrondi la moyenne à la classe la plus près pour le test de classification

        test_total += labels.numel()
    test_ecart = np.array(test_ecart) *10
    test_accuracy = np.sqrt(np.mean(test_ecart ** 2))
    print(f"\nJeu test {i}: RMSE = {test_accuracy:0.4f}\n")

# conversion des prediction en matrice numpy
y_pred_np = []
for k in y_pred.keys():
    y_pred_np.append(np.array([i.numpy() for i in y_pred[k]]))
y_pred_np = np.array(y_pred_np)

# calcul de la moyenne des predictions
y_true = np.array([x.item() for x in y_true]) *10
avg = np.average(y_pred_np, 0) *10
# avg[avg<17] = 17
avg[avg > 100] = 100
# y_true[y_true==40] = 80
avg_error = np.round(np.mean(abs(avg - y_true)), 3)
# classe ayant la plus haute moyenne de prediction
rounded_pred = np.round(avg, 0)

mean = np.mean(abs(avg - y_true))
std = np.std(abs(avg - y_true))
MSE = np.mean(((avg/10) - (y_true/10)) ** 2)
test_loss = MSE
RMSE = np.sqrt(np.mean((avg-y_true)**2))

print (f"Test final: RMSE moyen = {np.round(RMSE,2)}")


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


##################### graphs #####################
#
x = range(1, epochs + 1)

fig = plt.figure(figsize=(8, 4))
#
ax = fig.add_subplot(1, 1, 1)
ax.plot(x, train_losses, label='train')
ax.plot(x, valid_losses, label='valid')
ax.scatter(best_epoch, test_loss/10, color='red', label='test')
ax.set_xlabel('# epochs')
ax.set_ylabel('Loss (MSE)')
ax.legend()

##################### logs  #####################
# print(class_report)
#
# graphs
os.mkdir('checkpoints/' + name)
plt.savefig('checkpoints/' + name + '/graph_' + name + '.tif')
#

plt.figure(figsize=(9,6))

plt.scatter(y_true, avg, color = 'black', marker='.')
# plt.errorbar(list(set(y_true)), means, stds, linestyle = 'None',color='black', marker = '.', capsize = 3)
coef = np.polyfit(y_true, avg, 1)
slope, intercept, r_value, p_value, std_err = stats.linregress(y_true, avg)
plt.xlabel('IQBR terrain')
plt.ylabel('IQBR prédit')
plt.text(30, 90, f"r2: {round(r_value**2,2)}")
plt.text(30,85, f"Écart type des erreurs: {np.round(std,2)}")
plt.text(30,80, f"RMSE: {np.round(RMSE,2)}")
plt.plot(y_true, intercept + (np.array(y_true) * slope))
plt.grid()
plt.savefig('checkpoints/' + name + '/scatter_' + name + '.tif')
# show stuff
# plt.show()


# logs
window_size = crop_size
file = 'iqbr_mvdcnn_obcfiltered3_rgb_regress.csv'
comment =  f'L4, {views} vues, single FC, iqbr non modifié, valid et test voient toutes les images, 5 classes pour balance, 1* jeux de données, vraies valeurs, sched 20-0.2, centercrop, {permutations} permutations, (train val = 80-20), reload train_dataset each 1, MEAN POOL, classes COV, CJ 0.2, AUGMENT, 1 view par BR - nombre équilibré, paquet attrib à indice, seed = {seed}'
logger(date_time, model_name, model, dataset, window_size, epochs, learning_rate, batch_size, weight_decay, train_loss,
       best_model_accuracy, test_loss,
       0, 0, 0, name, file, save_checkpoint=True,
       pretrained='ImageNet', comment=comment)

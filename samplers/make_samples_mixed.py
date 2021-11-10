
#######################################################################################################
"""
Échantillonnage d'images pour des fausses bandes riveraines à partir d'un fichier shp de points.. Par exemple, un vecteur qui passe d'une forêt à une culture.
Ce script utilise un shp réalisé manuellement.
"""

import numpy as np
import os

import torch
import torchvision

from scipy import ndimage

from tqdm import tqdm
import rasterio
from rasterio.windows import Window
from shapely import wkt
from osgeo import ogr
import random


def closest(list, Number):
    temp = []
    for i in list:
        temp.append(abs(Number - i))

    return temp.index(min(temp))


def crop_center(img, cropx, cropy):
    y, x = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty:starty + cropy, startx:startx + cropx]


def round_of_rating(number):
    return round(number * 2) / 2

class SentinelImage(torch.utils.data.Dataset):
    def __init__(self, root, points_path, target_bands, window_size, type):
        assert os.path.isdir(root)
        self.bands = target_bands
        self.class_names = [0, 1, 2]
        self.window_size = window_size
        self.type = type

        images = []
        for r, d, f in os.walk(root):
            for file in f:
                if 'acadie_full.tif' == file:
                    images.append(os.path.join(r, file))

        # used for normalization
        self.stats = {1: [16.3727, 51.6343], 2: [12.4172, 76.0894], 3: [11.2851, 88.804], 4: [28.6557, 102.264],
                      5: [16.0906, 6.85636]}
        self.opened_images = []  # objets de fichiers ouverts
        for i in images:
            src = rasterio.open(i)
            global nodata
            nodata = src.nodata
            self.opened_images.append(src)

            # pour la normalisation et la gestion des NoData
            # for b in (target_bands):
            #     band = src.read(b)
            # no_data_mask = band == nodata
            # masked_array = np.ma.array(band, mask = no_data_mask)
            # mean = np.average(masked_array)
            # std = np.std(masked_array)
            # break
            # self.stats[b] = [std, mean]

        self.points_path = points_path

        self.points_list = []
        driver = ogr.GetDriverByName('ESRI Shapefile')
        ds = driver.Open(points_path, 0)
        points_lyr = ds.GetLayer()

        # checking if point inside image
        count = 0
        for l in tqdm(points_lyr):
            try:
                for img in self.opened_images:  # first UTM is easting, sol columns
                    row, col = img.index(wkt.loads(l.geometry().ExportToWkt()).x,
                                         wkt.loads(l.geometry().ExportToWkt()).y)
                # if col > img.shape[1]:
                #     print(col)
                # checking if all pixels of window are valid (without nodata) and inside image
                assert 0 < row < img.shape[0]
                assert 0 < col < img.shape[1]

                # nombre d'image de la classe dans la br

                if l.GetField('type') == self.type:
                    br_id = count // nb
                    self.points_list.append(
                        [l.geometry().ExportToWkt(), br_id, l.GetField('Angle'), l.GetField('type')])
                    count += 1
            except:
                pass

        # self.points_list =  [self.points_list[i] for i in self.idxs]
        # random.shuffle(self.points_list)
        # self.points_list = self.points_list[200:204]

        global opened
        opened = self.opened_images[0]

    def __len__(self):
        return len(self.points_list)

    def __getitem__(self, idx):
        point = self.points_list[idx]

        br_id = point[1]
        type = point[3]
        shapely_point = wkt.loads(point[0])
        x, y = shapely_point.x, shapely_point.y


        bands = []

        try:
            for img in self.opened_images:
                row, col = img.index(x, y)
                for b in self.bands:
                    enlarged_window = self.window_size // 0.66  # needed before rotation and crop
                    # (col_off, row_off, width, height)
                    window = img.read(b, window=Window(col - enlarged_window // 2, row - enlarged_window // 2,
                                                       enlarged_window, enlarged_window))
                    # window = ndimage.rotate(window, round(point[2], 0))
                    window = crop_center(window, self.window_size, self.window_size)

                    # normalisation par bande, moyenne, ecart-type
                    # window = ((window.astype(np.float32) - self.stats[b][1]) / self.stats[b][0]).astype(np.float32)
                    bands.append(window)

            bands = np.dstack(bands)
        except:
            print('shit')

        return bands, br_id, type  # return tuple with class index as 2nd member

type = 'culture'
nb = 4
target_bands = [1, 2, 3]
root = r'D:\deep_learning\images\use'
points_shp_path = r"D:\deep_learning\samples\sampling\manual_br\mixed_br_21m_points_all.shp"
outdir = f"D:\deep_learning\samples\manual_br\mixed_br\{type}{nb}\\"
batch_size = 1
window_size = 70

dataset = SentinelImage(root, points_shp_path, target_bands, window_size, type)

train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, num_workers=0)

if not os.path.isdir(outdir):
    os.mkdir(outdir)

month = '_july'
count = 0
br_id = np.arange(0,340)
for batch_idx, batch in enumerate(train_loader):
    bands, br_id, type = batch
    try:
        assert not 0 in bands

        fid = int(br_id)
        bands = bands[0].numpy().transpose(2, 0, 1)

        # sépare le jeu de données en val et train de manière aléatoire
        choice_list = ['train', 'val', 'test']
        distribution = [0.8, 0.1, 0.1]
        # choice = random.choices(choice_list, distribution)

        samplename = os.path.join(outdir, str(br_id.item()) + '_' + type[0] + str(count) + '.tif')
        count+=1

        if not os.path.isdir(os.path.dirname(samplename)):
            os.mkdir(os.path.dirname(samplename))
        with rasterio.open(samplename, 'w', driver='GTiff',
                           height=bands.shape[1],
                           width=bands.shape[2],
                           count=bands.shape[0],
                           dtype=bands.dtype,
                           # crs=opened.crs,
                           ) as dst:
            for k in range(3):
                dst.write(bands[k], indexes=k + 1)
    except:
        pass
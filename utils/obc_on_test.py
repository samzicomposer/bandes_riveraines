"""
Ce script test les résultats de l'orienté-objet sur chacun des folds lors de la 5-fold cross-validation.
Les fichiers .csv contenant les informations sur les folds sont dans ./autres

"""
from osgeo import ogr
import numpy as np
from collections import defaultdict
import csv

from scipy import stats
import matplotlib.pyplot as plt

# lecture du csv qui contient les index (sur 850) de chaque fold
file1 = r"K:\deep_learning\sampling\folds_idx.csv"
# lecture du csv qui contient les serials de bande riveraine pour chacun des index
file2 = r"K:\deep_learning\sampling\br_idx1.csv"
file3 = r"K:\deep_learning\sampling\br_idxfold5.csv"

fold_idxs_list = defaultdict(list)
with open(file1) as folds_idx:
    csv_reader = csv.reader(folds_idx, delimiter=',')
    for row in csv_reader:
        fold_idxs_list[row[0]] = row[1:]

serials_list_by_index = {}
with open(file2) as folds_idx:
    csv_reader = csv.reader(folds_idx, delimiter=';')
    for row in csv_reader:
        serials_list_by_index[row[0]] = row[1:]

folds_serials = defaultdict(list)

for fold, indexes in fold_idxs_list.items():
    for i in indexes:
        for br in serials_list_by_index[i]:
            folds_serials[fold].append(br[1:]) # on enlève le 'z'

### version pour les folds balancés ###

balanced_serials = []
with open(file3) as folds_idx:
    csv_reader = csv.reader(folds_idx, delimiter=';')
    for row in csv_reader:
        balanced_serials.append(row[0][1:])

############################################################################

points_shp_path = r"K:\deep_learning\sampling\rotated_poly_2m_10m.shp"

driver = ogr.GetDriverByName('ESRI Shapefile')
ds = driver.Open(points_shp_path, 0)
points_lyr = ds.GetLayer()

results_covabar = []
results_obc = []
serial = None
used_serials = []

# attention fold 1 : ne pas prendre iqbr_fid == '437e16' !!!
for p in points_lyr:
    try:
        # assert p.GetField('iqbr_diff') < 3.0
        assert p.GetField('test_ds') == 2
        # assert p.GetField('iqbr_diff') <=3.0
        assert p.GetField('IQBR') >=1.7
        if p.GetField('iqbr_srl') != serial:
            serial = p.GetField('iqbr_srl')
            iqbr_fid = serial[-7:-1]

            # on va chercher les indices de chacun des folds pour évaluer les résultats
            # if iqbr_fid in folds_serials['4']:

            # if iqbr_fid in balanced_serials and iqbr_fid != '437e16':
            iqbr_value = p.GetField('IQBR')
            results_covabar.append(iqbr_value)

            iqbr_obc = p.GetField('iqbr_seg')
            results_obc.append(iqbr_obc)
            used_serials.append(iqbr_fid)
        # else:
        #     pass
    except:
        pass

##################### confusion matrix #####################
# conversion des prediction en matrice numpy
print(used_serials)
print(results_obc)

y_true = np.array(results_covabar)*10
y_pred = np.array(results_obc)*10

rmse = np.sqrt(np.mean((y_pred-y_true)**2))


fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(111)
ax.set_axisbelow(True)
plt.grid()

plt.scatter(y_true, y_pred, color = 'black', marker='.', s=150)
# plt.errorbar(list(set(y_true)), means, stds, linestyle = 'None',color='black', marker = '.', capsize = 3)
coef = np.polyfit(y_true, y_pred, 1)
slope, intercept, r_value, p_value, std_err = stats.linregress(y_true, y_pred)
print(f"slope: {slope}, intercept: {intercept}")
plt.xlabel('Actual RSQI', fontsize=18, labelpad=20)
plt.ylabel('Predicted RSQI',fontsize=18)
plt.text(30, 95, f"R$^2$: {round(r_value**2,2)}",fontsize=18,bbox=dict(facecolor='white', edgecolor='white'))
# plt.text(30,90, f"Écart type des erreurs: {np.round(std,2)}")
plt.text(30,90, f"RMSE: {np.round(rmse,2)}", fontsize=18,bbox=dict(facecolor='white', edgecolor='white'))
plt.plot(y_true, intercept + (np.array(y_true) * slope), color='black')
plt.xticks([17,30,40,50,60,70,80,90,100], fontsize=18)
plt.yticks([17,30,40,50,60,70,80,90,100],fontsize=18)
ax.set_aspect('equal', adjustable='box')

print(f"RMSE: {rmse}")
print(f"R2: {r_value**2}")

# plt.xticks(np.arange(min(y_true), max(y_true)+1, 0.5))
# plt.xlim(1.7,10.0)
plt.show()



from osgeo import ogr
import numpy as np

from sklearn.metrics import confusion_matrix, classification_report
from utils.plot_a_confusion_matrix import plot_confusion_matrix
import matplotlib.pyplot as plt

points_shp_path = r"D:\deep_learning\samples\sampling\acadie_full_21m_points.shp"

driver = ogr.GetDriverByName('ESRI Shapefile')
ds = driver.Open(points_shp_path, 0)
points_lyr = ds.GetLayer()

results_covabar = []
results_obc = []
serial = None
for p in points_lyr:
    try:
        # assert p.GetField('iqbr_diff') < 3.0
        assert p.GetField('test_ds') == 1
        if p.GetField('iqbr_srl') != serial:
            serial = p.GetField('iqbr_srl')
            iqbr_value = p.GetField('IQBR')

            if iqbr_value <= 2.10:
                iqbr = 0
            elif 2.10 < iqbr_value <= 4.0:
                iqbr = 1
            elif 4.0 < iqbr_value <= 6.0:
                iqbr = 2
            elif 6.0 < iqbr_value <= 8.0:
                iqbr = 3
            else:
                iqbr = 4
            results_covabar.append(iqbr)

            iqbr_value = p.GetField('OBC_iqbr')
            if iqbr_value <= 2.10:
                iqbr = 0
            elif 2.10 < iqbr_value <= 4.0:
                iqbr = 1
            elif 4.0 < iqbr_value <= 6.0:
                iqbr = 2
            elif 6.0 < iqbr_value <= 8.0:
                iqbr = 3
            else:
                iqbr = 4
            results_obc.append(iqbr)
        else:
            pass
    except:
        pass

ecart = np.abs(np.array(results_covabar) - np.array(results_obc).astype(int))
one_class_diff_miss_percent = np.round(np.count_nonzero((np.array(ecart) == 1)) / np.count_nonzero(ecart != 0), 3)
one_class_diff_percent = np.round(np.count_nonzero((np.array(ecart) <= 1)) / len(ecart), 3)

print(one_class_diff_percent)

# conf_class_map = {idx: name for idx, name in enumerate(dataset.class_names)}
# y_true = [conf_class_map[idx.item()] for idx in y_true]
# y_pred = [conf_class_map[idx.item()] for idx in y_pred]
classes = ['1.7-2.1', '2.1-4.0', '4.0-6.0', '6.0-8.0', '8.0-10']
class_report = classification_report(results_covabar, results_obc, target_names=classes)
cm = confusion_matrix(results_covabar, results_obc)

print(class_report)

# # confusion matrix
plot_confusion_matrix(cm, 'nom', normalize=True,
                      target_names=classes,
                      title="Confusion Matrix", show=True, save=False)
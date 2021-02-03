# https://glenbambrick.com/2017/09/15/osgp-create-points-along-line/

from osgeo import ogr
from shapely.geometry import LineString, Point
from shapely import wkt, ops
from tqdm import tqdm
import math
import random
import string
import os

def to_degre(angle):
    degre = 360.0 * angle / (2 * math.pi)
    return round(degre,2)

def calculateAngle(point1, point2):
    """
    :param line: Shapely Point object
    :return: angle in degre
    """
    try:
        angle = math.atan((point2.y - point1.y) / (point2.x - point1.x))
    except ZeroDivisionError:
        # when rotating, no polygon is created if angle == 0, pi rad is then chosen
        angle = math.pi
    if angle == 0:
        # when rotating, no polygon is created if angle == 0, pi rad is then chosen
        angle = math.pi
    angle = (to_degre(angle))
    if 0 > angle > -45:
        angle = -angle
    elif angle < -45:
        angle = - 90 - angle
    elif 45 > angle > 0:
        angle = -angle
    elif angle > 45:
        angle = 90 - angle
    return angle


## set the driver for the data
driver = ogr.GetDriverByName('ESRI Shapefile')
################################################################################
## path to the
shp_droite = r"I:/donnees/Covabar/nettoyage/IQBR_IQHP_utm.shp"


dirname = os.path.dirname(shp_droite)
## open the GDB in write mode (1)
ds = driver.Open(shp_droite, 0)


## single linear feature
input_lyr = ds.GetLayer()

## distance between each points
# pour les dernières tuiles 50x50
distance = 21
# distance = 15
## distance from wich the parallel lines will be created
# pour les dernières tuiles 50x50
# distance_paral = 10
distance_paral = 10

## output point fc name
output_paral = r"D:/deep_learning/samples/sampling/nord/{}_parall_{}m".format(os.path.basename(shp_droite[:-4]), distance_paral)
## output point fc name
output_pts = r"D:/deep_learning/samples/sampling/nord/{}_{}m_points".format(os.path.basename(shp_droite[:-4]), distance)

################################################################################

## create a new point layer with the same spatial ref as lyr
outShapefile = "{}_all_iqbr.shp".format(output_pts)
outShapeParal = "{}_center_iqbr.shp".format(output_paral)

if os.path.isfile(outShapefile):
    os.remove(outShapefile)
    print ("Deleting: {0}".format(output_pts))
if os.path.isfile(outShapeParal):
    os.remove(outShapeParal)
    print ("Deleting: {0}".format(output_paral))

outDataSource = driver.CreateDataSource(outShapefile)
out_lyr = outDataSource.CreateLayer(output_pts, input_lyr.GetSpatialRef(), ogr.wkbPoint)

################################################################################
# ouvrir les cours d'eau pour exclure les points qui sont trop loin
# shp_water = r"/Users/samueldelasablonniere/Documents/OneDrive - USherbrooke/Maitrise/Données/Covabar/Caracterisation_full/cours_eau_buff_13.shp"
# water = driver.Open(shp_water, 0)
# water_lyr = water.GetLayer()

# load le fichier des cours d'eau en format shapely
# water_shapely = []
# for wa in water_lyr:
#     geom = wa.geometry().ExportToWkt()
#     shapely_wa = wkt.loads(geom)
#     water_shapely.append(shapely_wa)
################################################################################

## create fields to hold values
angle_fld = ogr.FieldDefn("Angle", ogr.OFTReal)
largeur_fld = ogr.FieldDefn("Largeur", ogr.OFTReal)
iqbr_fld = ogr.FieldDefn("IQBR", ogr.OFTReal)
classe_fld = ogr.FieldDefn("Classe", ogr.OFTInteger)
serial_fld = ogr.FieldDefn("iqbr_srl", ogr.OFTString)
test_fld = ogr.FieldDefn("test_ds", ogr.OFTInteger)
obc_iqbr = ogr.FieldDefn("OBC_iqbr", ogr.OFTReal)
iqbr_diff = ogr.FieldDefn("iqbr_diff", ogr.OFTReal)
out_lyr.CreateField(angle_fld)
out_lyr.CreateField(largeur_fld)
out_lyr.CreateField(iqbr_fld)
out_lyr.CreateField(classe_fld)
out_lyr.CreateField(serial_fld)
out_lyr.CreateField(test_fld)
out_lyr.CreateField(iqbr_diff)
out_lyr.CreateField(obc_iqbr)

#################################
# trouver tous les attributs
schema = []
ldefn = input_lyr.GetLayerDefn()
for n in range(ldefn.GetFieldCount()):
    fdefn = ldefn.GetFieldDefn(n)
    schema.append(fdefn.name)
print (schema)

#############################################
# génère des paires gauche-droite de vecteurs
# la loop ne fonctionne pas si on itère directement sur input_layer, la raison de lyr1 et lyr2

# classes = [[0, 0.99], [1.0, 3.0], [3.1, 5.5], [5.6, 200]] # v2
# classes = [[0, 0.99], [1.0, 2.99], [3.0, 5.5], [5.6, 200]] # v1
classes = [[0, 1.0], [1.01, 2.99], [3.0, 5.5], [5.6, 200]] # v3
classes_1m = [[0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 300]]
classe_iqbr = [[1.7, 4.0], [4.0, 6.0], [6.0, 8.0], [8.0, 10]]

stations_numbers = []
pairlist = []
lyr1 = [z for z in input_lyr]

# l'identifiant unique pour les paires est (Station, No_GPS)
for line1 in tqdm(lyr1):
    # largeur1 = line1.GetField('Largeur_D')
    # largeur2 = line1.GetField('Largeur_G')
    # iqbr1 = line1.GetField('IQBR_Gauch')
    # iqbr2 = line1.GetField('IQBR_Droit')


    # if any( (z[0] <= iqbr1  <= z[1]) and (z[0] <= iqbr2  <= z[1]) for z in classe_iqbr):

    # if abs(largeur1 - largeur2) <= 1.0:
    pairlist.append([line1])

# génère des lignes parallèles à celle du shp
linestring_list = []
for ln in tqdm(pairlist):
    # numero station
    station = ln[0].GetField('Station')
    try:# to shapely object
        line_geom_side1 = ln[0].geometry().ExportToWkt()

        shapely_line1 = wkt.loads(line_geom_side1)
        paral_right = shapely_line1.parallel_offset(distance_paral, 'right', join_style=2)
        paral_left = shapely_line1.parallel_offset(distance_paral, 'left', join_style=2)

        if ln[0].GetField('true_dir') == 0:
            linestring_list.append(
                {'geom': paral_right, 'Largeur': ln[0].GetField('Largeur_D') , 'IQBR': ln[0].GetField('IQBR_Droit'), 'serial': ln[0].GetField('serial_0'), 'test_ds':ln[0].GetField('test_ds'),
                 'iqbr_diff':abs(ln[0].GetField('IQBR_Droit') - ln[0].GetField('OBC_iqbr_0')), 'OBC_iqbr': ln[0].GetField('OBC_iqbr_0')})
            linestring_list.append(
                {'geom': paral_left, 'Largeur': ln[0].GetField('Largeur_G'), 'IQBR': ln[0].GetField('IQBR_Gauch'), 'serial': ln[0].GetField('serial_1'), 'test_ds':ln[0].GetField('test_ds'),
                 'iqbr_diff':abs(ln[0].GetField('IQBR_Gauch') - ln[0].GetField('OBC_iqbr_1')), 'OBC_iqbr': ln[0].GetField('OBC_iqbr_1')})
        else:
            linestring_list.append(
                {'geom': paral_left, 'Largeur': ln[0].GetField('Largeur_D'), 'IQBR': ln[0].GetField('IQBR_Droit'), 'serial': ln[0].GetField('serial_0'), 'test_ds':ln[0].GetField('test_ds'),
                 'iqbr_diff':abs(ln[0].GetField('IQBR_Droit') - ln[0].GetField('OBC_iqbr_0')), 'OBC_iqbr': ln[0].GetField('OBC_iqbr_0')})
            linestring_list.append(
                {'geom': paral_right, 'Largeur': ln[0].GetField('Largeur_G'), 'IQBR': ln[0].GetField('IQBR_Gauch'), 'serial': ln[0].GetField('serial_1'), 'test_ds':ln[0].GetField('test_ds'),
                 'iqbr_diff':abs(ln[0].GetField('IQBR_Gauch') - ln[0].GetField('OBC_iqbr_1')), 'OBC_iqbr': ln[0].GetField('OBC_iqbr_1')})

    except:
        pass
    # linestring_list.append(
    #     {'geom': shapely_line1, 'Largeur': abs(ln[0].GetField('Largeur_D') + ln[0].GetField('Largeur_G') / 2), 'IQBR': ln[0].GetField('IQBR_Droit')})

## Génère des points sur les lignes parallèles créées précédemment
for ln in tqdm(linestring_list):
    ## list to hold all the point coords
    list_points = []

    if ln['geom'].geom_type == 'LineString':

        ## set the current distance to place the point
        current_dist = distance
        line_length = ln['geom'].length

        # début de ligne
        #list_points.append(Point(ln['geom'].coords[0]))

        while current_dist < line_length:
            ## use interpolate and increase the current distance
            list_points.append(ln['geom'].interpolate(current_dist))
            current_dist += distance

        # fin de la ligne
        #list_points.append(Point(ln['geom'].coords[-1]))

    elif ln['geom'].geom_type == 'MultiLineString':
        ## append the starting coordinate to the list
        for l in ln['geom']:
            line_length =l.length
            current_dist = distance

            # début de ligne
            #list_points.append(Point(l.coords[0]))

            while current_dist < line_length:
                ## use interpolate and increase the current distance
                list_points.append(l.interpolate(current_dist))
                current_dist += distance

            # fin de la ligne
            #list_points.append(Point(l.coords[-1]))


    # add points to the layer
    # for each point in the list
    for pt in list_points:

        # get angle between two subsequent points
        index = list_points.index(pt)
        try:
            angle = calculateAngle(pt, list_points[index+1])
        except IndexError:
            angle = calculateAngle(pt, list_points[index-1])

        ## create a point object
        pnt = ogr.Geometry(ogr.wkbPoint)
        pnt.AddPoint(pt.x, pt.y)
        feat_dfn = out_lyr.GetLayerDefn()
        feat = ogr.Feature(feat_dfn)
        feat.SetGeometry(pnt)
        feat.SetField("Angle", angle)
        if ln['Largeur'] > 15:
            feat.SetField("Largeur", 15)
        else:
            feat.SetField("Largeur", ln['Largeur'])
        feat.SetField("IQBR", ln['IQBR'])
        feat.SetField("iqbr_srl", ln['serial'])
        feat.SetField("test_ds", ln['test_ds'])
        feat.SetField("iqbr_diff", ln['iqbr_diff'])
        feat.SetField("OBC_iqbr", ln['OBC_iqbr'])

        ## add the point feature to the output.
        out_lyr.CreateFeature(feat)


# output du résultat des lignes parallèles dans un shape
# outDataSource = driver.CreateDataSource(outShapeParal)
# out_lyr_paral = outDataSource.CreateLayer(output_paral, input_lyr.GetSpatialRef(), ogr.wkbLineString)
# for line in linestring_list:
#     ## create a point object
#
#     geom = ogr.CreateGeometryFromWkt(line['geom'].wkt)
#
#     feat_dfn = out_lyr_paral.GetLayerDefn()
#     feat = ogr.Feature(feat_dfn)
#     feat.SetGeometry(geom)
#
#     ## add the line feature to the output.
#     out_lyr_paral.CreateFeature(feat)

del ds, out_lyr #, out_lyr_paral
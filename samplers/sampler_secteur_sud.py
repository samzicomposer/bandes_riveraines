# https://glenbambrick.com/2017/09/15/osgp-create-points-along-line/

"""
Ce script crée des points de chaque côté des cours d'eau du secteur sud. Le shp à partir duquel il travaille
contient un vecteur par rive. La gauche et la droite est donc déjà déterminée.

"""

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
shp_droite = r"I:/donnees/Covabar/nettoyage/iqbr_sec_sud_edit.shp"
shp_gauche = r"I:/donnees/Covabar/nettoyage/iqbr_sec_sud_edit.shp"
shps = [shp_droite,shp_gauche]

dirname = os.path.dirname(shp_droite)
## open the GDB in write mode (1)
ds = driver.Open(shp_droite, 0)
## nécessaire pour la formation des paires
ds2 = driver.Open(shp_gauche, 0)

## single linear feature
input_lyr = ds.GetLayer()
## nécessaire pour la formation des paires
input_lyr2 = ds2.GetLayer()

## distance between each points
distance = 21
## distance from wich the parallel lines will be created
distance_paral = 10

## output point fc name
output_paral = r"K:/deep_learning/sampling/sud/{}_parall_{}m".format(os.path.basename(shp_gauche[:-4]), distance_paral)
## output point fc name
output_pts = r"K:/deep_learning/sampling/sud/{}_2m_{}m_points".format(os.path.basename(shp_gauche[:-4]), distance)

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
# classes = [[0, 1.0], [1.01, 2.99], [3.0, 5.5], [5.6, 200]] # v3
classes_1m = [[0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 300]]
classe_iqbr = [[1.7, 4.0], [4.0, 6.0], [6.0, 8.0], [8.0, 10]]


stations_numbers = []
pairlist = []
lyr1 = [z for z in input_lyr]
lyr2 = [y for y in input_lyr2]

# l'identifiant unique pour les paires est (Station, No_GPS)
for line1 in tqdm(lyr1):
    station1 = line1.GetField('Station')
    gps1 = line1.GetField('No_GPS')
    largeur1 = line1.GetField('Largeur')
    iqbr1 = line1.GetField('IQBR')

    for line2 in lyr2:
        station2 = line2.GetField('Station')
        gps2 = line2.GetField('No_GPS')
        largeur2 = line2.GetField('Largeur')
        iqbr2 = line2.GetField('IQBR')

        # conditions pour que les bv soient pareilles de chauqe côté
        # if station1 == station2 and line1.GetFID() != line2.GetFID() and gps1 == gps2 and [station1,gps1] not in stations_numbers and \
        #         any( (z[0] <= largeur1  <= z[1]) and (z[0] <= largeur2  <= z[1]) for z in classes):

        if station1 == station2 and line1.GetFID() != line2.GetFID() and gps1 == gps2 and [station1, gps1] not in stations_numbers :

        # if station1 == station2 and line1.GetFID() != line2.GetFID() and gps1 == gps2 and [station1, gps1] not in stations_numbers:

        # if station1 == station2 and line1.GetFID() != line2.GetFID() and gps1 == gps2 and [station1,gps1] not in stations_numbers and \
        #         abs(largeur1 - largeur2) <= 1.0:

            pairlist.append([line1, line2])
            stations_numbers.append([station1,gps1])


# génère des lignes parallèles à celle du shp
linestring_list = []
for ln in tqdm(pairlist):
    # numero station
    station = ln[0].GetField('Station')
    # to shapely object
    line_geom_side1 = ln[0].geometry().ExportToWkt()
    shapely_line1 = wkt.loads(line_geom_side1)
    line_geom_side2 = ln[1].geometry().ExportToWkt()
    shapely_line2 = wkt.loads(line_geom_side2)

    paral_right_1 = shapely_line1.parallel_offset(distance_paral, 'right', join_style =2)
    paral_right_2 = shapely_line1.parallel_offset(distance_paral, 'left', join_style =2)
    paral_left_1 = shapely_line2.parallel_offset(distance_paral, 'right', join_style =2)
    paral_left_2 = shapely_line2.parallel_offset(distance_paral, 'left', join_style =2)

    try:
        # on cherche la ligne parallele la plus proche de celle de l'autre rive
        if paral_right_1.distance(paral_left_1) < paral_right_2.distance(paral_left_1): # inversé pour images sentinel, on s'éloigne de la rive
            linestring_list.append({'geom': paral_right_1, 'Largeur': ln[0].GetField('Largeur') , 'IQBR':ln[0].GetField('IQBR'), 'serial': ln[0].GetField('serial'), 'test_ds':ln[0].GetField('test_ds'),
                                    'iqbr_diff':abs(ln[0].GetField('IQBR') - ln[0].GetField('OBC_iqbr')), 'OBC_iqbr': ln[0].GetField('OBC_iqbr')})
            # linestring_list.append({'geom': paral_right_1, 'Classe': 1})
        else:
            linestring_list.append({'geom': paral_right_2, 'Largeur': ln[0].GetField('Largeur') , 'IQBR':ln[0].GetField('IQBR'), 'serial': ln[0].GetField('serial'),'test_ds':ln[0].GetField('test_ds'),
                                    'iqbr_diff':abs(ln[0].GetField('IQBR') - ln[0].GetField('OBC_iqbr')), 'OBC_iqbr': ln[0].GetField('OBC_iqbr')})
            # linestring_list.append({'geom': paral_right_2, 'Classe': 1})

        if paral_left_1.distance(paral_right_1) < paral_left_2.distance(paral_right_1):  # inversé pour images sentinel, on s'éloigne de la rive
            linestring_list.append({'geom': paral_left_1, 'Largeur': ln[1].GetField('Largeur'), 'IQBR': ln[1].GetField('IQBR'), 'serial': ln[1].GetField('serial'), 'test_ds':ln[1].GetField('test_ds'),
                                    'iqbr_diff':abs(ln[1].GetField('IQBR') - ln[1].GetField('OBC_iqbr')), 'OBC_iqbr': ln[1].GetField('OBC_iqbr')})
            # linestring_list.append({'geom': paral_right_1, 'Classe': 1})
        else:
            linestring_list.append({'geom': paral_left_2, 'Largeur': ln[1].GetField('Largeur'), 'IQBR': ln[1].GetField('IQBR'), 'serial': ln[1].GetField('serial'), 'test_ds':ln[1].GetField('test_ds'),
                                    'iqbr_diff':abs(ln[1].GetField('IQBR') - ln[1].GetField('OBC_iqbr')), 'OBC_iqbr': ln[1].GetField('OBC_iqbr')})
            # linestring_list.append({'geom': paral_right_2, 'Classe': 1})
    except:
        pass

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

        while current_dist < line_length :
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

            while current_dist < line_length :
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
        # feat.SetField("Classe", ln['Classe'])

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
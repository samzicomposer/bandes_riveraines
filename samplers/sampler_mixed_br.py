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
shp_droite = r"D:/deep_learning/samples/sampling/manual_br/mixed_br.shp"

dirname = os.path.dirname(shp_droite)
## open the GDB in write mode (1)
ds = driver.Open(shp_droite, 0)

## single linear feature
input_lyr = ds.GetLayer()

## distance between each points
# pour les dernières tuiles 50x50
distance = 21

distance_paral = 0

## output point fc name
output_paral = r"D:/deep_learning/samples/sampling/manual_br/{}_parall_{}m".format(os.path.basename(shp_droite[:-4]), distance_paral)
## output point fc name
output_pts = r"D:/deep_learning/samples/sampling/manual_br/{}_{}m_points".format(os.path.basename(shp_droite[:-4]), distance)

################################################################################

## create a new point layer with the same spatial ref as lyr
outShapefile = "{}_all.shp".format(output_pts)
outShapeParal = "{}_center_iqbr.shp".format(output_paral)

if os.path.isfile(outShapefile):
    os.remove(outShapefile)
    print ("Deleting: {0}".format(output_pts))
if os.path.isfile(outShapeParal):
    os.remove(outShapeParal)
    print ("Deleting: {0}".format(output_paral))

outDataSource = driver.CreateDataSource(outShapefile)
out_lyr = outDataSource.CreateLayer(output_pts, input_lyr.GetSpatialRef(), ogr.wkbPoint)

## create fields to hold values
angle_fld = ogr.FieldDefn("Angle", ogr.OFTReal)
type_fld = ogr.FieldDefn("type", ogr.OFTString)
fid_fld = ogr.FieldDefn("fid", ogr.OFTInteger)
out_lyr.CreateField(angle_fld)
out_lyr.CreateField(type_fld)
out_lyr.CreateField(fid_fld)

#################################
# trouver tous les attributs
schema = []
ldefn = input_lyr.GetLayerDefn()
for n in range(ldefn.GetFieldCount()):
    fdefn = ldefn.GetFieldDefn(n)
    schema.append(fdefn.name)
print (schema)

#############################################

stations_numbers = []
pairlist = []
lyr1 = [z for z in input_lyr]

# l'identifiant unique pour les paires est (Station, No_GPS)
for line1 in tqdm(lyr1):

    pairlist.append([line1])

# génère des lignes parallèles à celle du shp
linestring_list = []
for ln in tqdm(pairlist):
    # numero station
    try:# to shapely object
        line_geom_side1 = ln[0].geometry().ExportToWkt()

        shapely_line1 = wkt.loads(line_geom_side1)
        paral_right = shapely_line1.parallel_offset(distance_paral, 'right', join_style=2)
        linestring_list.append({'geom': paral_right, 'fid': ln[0].GetField('id'), 'type': ln[0].GetField('classe')})

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

        feat.SetField("fid", ln['fid'])
        feat.SetField("type", ln['type'])

        ## add the point feature to the output.
        out_lyr.CreateFeature(feat)


del ds, out_lyr #, out_lyr_paral


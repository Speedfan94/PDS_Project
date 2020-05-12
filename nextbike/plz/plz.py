import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import folium
import seaborn as sns
import shapely.geometry as shapely
import json
import vincenty
from shapely.geometry import shape, Point

plz_value = {}

with open('../nextbike/data/input/postleitzahlen-nuremberg.geojson') as f:
    geo = json.load(f)

# In which state is the **center** of Germany
# Note reverse notation compared to before
# searchpoint = Point(-0.109970, 51.529163)


# Just for a single Point
def plz1(df):

    searchpoint = Point(11.076750, 49.452030)
    # df_london_center["Longitude"], df_london_center["Latitude"])

    for feature in geo['features']:
        polygon = shape(feature['geometry'])
        if polygon.contains(searchpoint):
            print('Found containing polygon:', feature["properties"]["plz"])
    return Point


def plz(df):

    for feature in geo['features']:
        if feature['geometry']['type'] == 'MultiPolygon':
            plz_value[feature['properties']['plz']] = list(shapely.shape(feature['geometry']))
        elif feature['geometry']['type'] == 'Polygon':
            plz_value[feature['properties']['plz']] = shapely.shape(feature['geometry'])

    df['plz_start'] = df.apply(lambda x: get_plz(x['Latitude_start'], x['Longitude_start']), axis=1)
    return df
    # df_plz = df.groupby('plz_start', as_index=False)['Capacity'].count()
    # df_boroughs.rename({'Capacity': 'Stations'}, axis=1, inplace=True)


def get_plz(lat, lon):
    p = shapely.Point(lon, lat)
    for plz, shape in plz_value.items():
        if type(shape) == list:
            for poly in shape:
                if poly.contains(p):
                    return plz
        elif shape.contains(p):
            return plz


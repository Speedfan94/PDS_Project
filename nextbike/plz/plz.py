import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import folium
import seaborn as sns

import json
import vincenty
from shapely.geometry import shape, Point

with open('../nextbike/data/input/postleitzahlen-nuremberg.geojson') as f:
    geo = json.load(f)

# In which state is the **center** of Germany
# Note reverse notation compared to before
# searchpoint = Point(-0.109970, 51.529163)

def plz(df):

    searchpoint = Point(11.076750, 49.452030)
    # df_london_center["Longitude"], df_london_center["Latitude"])

    for feature in geo['features']:
        polygon = shape(feature['geometry'])
        if polygon.contains(searchpoint):
            print('Found containing polygon:', feature["properties"]["plz"])
    return Point


def plz2(df):
    for feature in geo['features']:
        # x = stations_df_copy["Longitude"]
        # y = stations_df_copy["Latitude"]
        searchpoint = Point(df["Longitude_start"], df["Latitude_start"])
        polygon = shape(feature['geometry'])
        if polygon.contains(searchpoint):
            df["plz"] = feature["properties"]["plz"]
    return df

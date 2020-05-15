import shapely.geometry as shapely
import json
from .. import io
from shapely.geometry import shape, Point

# TODO: Try iterating through PLZs, find matching data points & kick them out instead of vice versa

plz_value = {}

with open(io.get_path("postleitzahlen-nuremberg.geojson", "input")) as f:
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


def plz_value_def():
    if plz_value == {}:
        for feature in geo['features']:
            if feature['geometry']['type'] == 'MultiPolygon':
                plz_value[feature['properties']['plz']] = list(shapely.shape(feature['geometry']))
            elif feature['geometry']['type'] == 'Polygon':
                plz_value[feature['properties']['plz']] = shapely.shape(feature['geometry'])


def plz(df):

    plz_value_def()

    df['plz_start'] = df.apply(lambda x: get_plz(x['Latitude_start'], x['Longitude_start']), axis=1)
    return df


def plz_end(df):

    plz_value_def()

    df['plz_end'] = df.apply(lambda x: get_plz(x['Latitude_end'], x['Longitude_end']), axis=1)
    return df


def get_plz(lat, lon):
    p = shapely.Point(lon, lat)
    for plz, shape in plz_value.items():
        if type(shape) == list:
            for poly in shape:
                if poly.contains(p):
                    return plz
        elif shape.contains(p):
            return plz

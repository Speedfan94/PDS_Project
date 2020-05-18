import shapely.geometry as shapely
import json
from nextbike import io
from shapely.geometry import shape, Point
import pandas as pd

# TODO: Try iterating through PLZs, find matching data points & kick them out instead of vice versa

plz_value = {}

with open(io.get_path("postleitzahlen-nuremberg.geojson", "input")) as f:
    geo = json.load(f)


# In which state is the **center** of Germany
# Note reverse notation compared to before
# searchpoint = Point(-0.109970, 51.529163)


# Just for a single Point
def plz1(df):
    """TODO:What does this method do?

    Args:
        df (DataFrame):
    Returns:
        Point (Point):
    """
    searchpoint = Point(11.076750, 49.452030)
    # df_london_center["Longitude"], df_london_center["Latitude"])

    for feature in geo['features']:
        polygon = shape(feature['geometry'])
        if polygon.contains(searchpoint):
            print('Found containing polygon:', feature["properties"]["plz"])
    return Point


def plz_value_def():
    """TODO:What does this method do?

    Args:
        no args
    Returns:
        no return
    """
    if plz_value == {}:
        for feature in geo['features']:
            if feature['geometry']['type'] == 'MultiPolygon':
                plz_value[feature['properties']['plz']] = list(shapely.shape(feature['geometry']))
            elif feature['geometry']['type'] == 'Polygon':
                plz_value[feature['properties']['plz']] = shapely.shape(feature['geometry'])


def plz(df):
    """Add the corresponding plz to each start long and lat in DataFrame

    Args:
        df (DataFrame): DataFrame of trips
    Returns:
        df (DataFrame): DataFrame of trips with start plz
    """
    plz_value_def()

    df['plz_start'] = df.apply(lambda x: get_plz(x['Latitude_start'], x['Longitude_start']), axis=1)
    return df


# TODO: no concise way of workflow in program structure
#  => "plz" starts initialization of "plz_value" in method plz_value_def
#  => "plz" add start plz with method "get_plz"
#  => end plz is then added by plz_end which is called in another file...

def plz_end(df):
    """Add the corresponding plz to each end long and lat in DataFrame

    Args:
        df (DataFrame): DataFrame of trips
    Returns:
        df (DataFrame): DataFrame of trips with end plz
    """
    plz_value_def()

    # TODO: Fix that bad style right here:
    pd.options.mode.chained_assignment = None
    df['plz_end'] = df.apply(lambda x: get_plz(x['Latitude_end'], x['Longitude_end']), axis=1)
    # Todo: fix that bad style right here
    pd.options.mode.chained_assignment = "warn"
    return df


def get_plz(lat, lon):
    """return the plz for given longitude and latitude

    Args:
        lat (int): latitude of trip
        lon (int): longitude of trip
    Returns:
        i_plz (int): plz for given long and lat
        """
    p = shapely.Point(lon, lat)
    for i_plz, i_shape in plz_value.items():
        if type(i_shape) == list:
            for poly in i_shape:
                if poly.contains(p):
                    return i_plz
        elif i_shape.contains(p):
            return i_plz

import shapely.geometry as shapely
import json
from nextbike import io
from shapely.geometry import shape, Point
import pandas as pd
from datetime import datetime


# TODO: Try iterating through PLZs, find matching data points & kick them out instead of vice versa

plz_value = {}

with open(io.get_path(p_filename="postleitzahlen-nuremberg.geojson", p_io_folder="input")) as f:
    geo = json.load(f)


# In which state is the **center** of Germany
# Note reverse notation compared to before
# searchpoint = Point(-0.109970, 51.529163)


def only_nuremberg(p_df):
    """Calculates corresponding zip codes to each data point and
    filters out all data points not in nuremberg (based on zip codes)

    Args:
        p_df (DataFrame): DataFrame with trip data
    Returns:
        df_nuremberg (DataFrame): DataFrame with trip data from nuremberg
    """
    # DropTrips outside of Nuremberg with no PLZ, depending on their Start
    # Information: Nuremberg City Center: Lat: 49.452030, Long: 11.076750
    # --> https://www.laengengrad-breitengrad.de/gps-koordinaten-von-nuernberg

    # adding plz to df
    # Add PLZ to trip and drop trips without start or end PLZ

    # TODO: resolve names of methods or file
    # add start plz
    df_plz = plz(p_df)
    df_nurem = df_plz.dropna(axis=0)
    # add end plz
    # df_nurem = plz_end(df_nurem)
    # df_nurem = df_nurem.dropna(axis=0)

    return df_nurem


# TODO: Do we actually need this method?
# Just for a single Point
def plz1(p_df):
    """TODO:What does this method do?

    Args:
        p_df (DataFrame):
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


def plz(p_df):
    """Add the corresponding plz to each start long and lat in DataFrame

    Args:
        p_df (DataFrame): DataFrame of trips
    Returns:
        df (DataFrame): DataFrame of trips with start plz
    """
    plz_value_def()

    p_df_new = get_plz(p_df)
    # p_df['plz_start'] = p_df.apply(lambda x: get_plz(x['Latitude_start'], x['Longitude_start']), axis=1)
    return p_df_new


# TODO: no concise way of workflow in program structure
#  => "plz" starts initialization of "plz_value" in method plz_value_def
#  => "plz" add start plz with method "get_plz"
#  => end plz is then added by plz_end which is called in another file...

def plz_end(p_df):
    """Add the corresponding plz to each end long and lat in DataFrame

    Args:
        p_df (DataFrame): DataFrame of trips
    Returns:
        df (DataFrame): DataFrame of trips with end plz
    """
    plz_value_def()

    # TODO: Fix that bad style right here:
    pd.options.mode.chained_assignment = None
    # p_df['plz_end'] = p_df.apply(lambda x: get_plz(x['Latitude_end'], x['Longitude_end']), axis=1)
    # Todo: fix that bad style right here
    pd.options.mode.chained_assignment = "warn"
    return p_df


def get_plz(p_df):
    """return the plz for given longitude and latitude

    Args:
        p_lat (int): latitude of trip
        p_lon (int): longitude of trip
    Returns:
        i_plz (int): plz for given long and lat
    """

    print("Calculating starting points for df...")
    # p_df["Point_start"] = p_df.apply(lambda row: shapely.Point(row["Longitude_start"], row["Latitude_start"]), axis=1)
    p_df["plz_start"] = None
    print("DONE Calculating starting points for df")
    for iter_plz, iter_shape in plz_value.items():
        print("Checking PLZ", iter_plz)
        start_time = datetime.now().replace(microsecond=0)
        if type(iter_shape) == list:
            for poly in iter_shape:
                p_df["plz_start"] = p_df.apply(lambda row: set_plz(row["Longitude_start"], row["Latitude_start"], poly, iter_plz) if (row["plz_start"] is None) else None, axis=1)
                """
                for index, row in p_df.iterrows():
                    if row["plz_start"] == 0:
                        starting_point = shapely.Point(row['Longitude_start'], row['Latitude_start'])
                        if poly.contains(starting_point):
                            row["plz_start"] = iter_plz
                """
                print("PLZ Start (Poly):", p_df["plz_start"])
        else:
            p_df["plz_start"] = p_df.apply(lambda row: set_plz(row["Longitude_start"], row["Latitude_start"], iter_shape, iter_plz) if (row["plz_start"] is None) else None, axis=1)
            """
            for index, row in p_df.iterrows():
                if row["plz_start"] == 0:
                    starting_point = shapely.Point(row['Longitude_start'], row['Latitude_start'])
                    if iter_shape.contains(starting_point):
                        row["plz_start"] = iter_plz
            """
            print("PLZ Start (Shape):", p_df["plz_start"])
        print("TIME FOR PLZ:", (datetime.now().replace(microsecond=0) - start_time))

    return p_df


def set_plz(p_lng, p_lat, p_poly, p_iter_plz):
    point = shapely.Point(p_lng, p_lat)
    if p_poly.contains(point):
        return p_iter_plz

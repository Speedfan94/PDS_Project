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
    print("Calculating PLZ shapes...")
    start_time_plz_def = datetime.now().replace(microsecond=0)
    plz_value_def()
    print("TIME FOR PLZ DEF:", (datetime.now().replace(microsecond=0) - start_time_plz_def))

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

    df_start = p_df[["Longitude_start", "Latitude_start"]]
    df_geo_start = df_start.rename({
        "Longitude_start": "Longitude",
        "Latitude_start": "Latitude"
    }, axis=1)

    df_end = p_df[["Longitude_end", "Latitude_end"]]
    df_geo_end = df_end.rename({
        "Longitude_end": "Longitude",
        "Latitude_end": "Latitude"
    }, axis=1)

    df_all_points = df_geo_start.append(df_geo_end, ignore_index=True)
    # Could be rounded to boost performance
    # df_all_points["Longitude"] = df_all_points["Longitude"].round(6)
    # df_all_points["Latitude"] = df_all_points["Latitude"].round(6)
    print("Number of trips:", len(p_df))
    print("Number of points:", len(df_all_points))
    df_unique_points = df_all_points.drop_duplicates(["Longitude", "Latitude"], ignore_index=True)
    number_of_unique_points = len(df_unique_points)
    print("Number of UNIQUE points:", number_of_unique_points)

    start_time_plz_calc = datetime.now().replace(microsecond=0)
    df_unique_points["plz"] = df_unique_points.apply(lambda row: calc_postalcode_for_coords(row, number_of_unique_points), axis=1)
    print("")
    print("=== TIME FOR PLZ CALC:", (datetime.now().replace(microsecond=0) - start_time_plz_calc))

    start_time_plz_to_original_df = datetime.now().replace(microsecond=0)
    df_nurem_points = df_unique_points.dropna()
    number_of_nurem_points = len(df_nurem_points)
    df_nurem_points.apply(lambda row: map_postalcode_to_original_df(row, p_df, number_of_nurem_points), axis=1)
    print("=== TIME FOR PLZ TO ORIGINAL DF:", (datetime.now().replace(microsecond=0) - start_time_plz_to_original_df))

    print(p_df[["Longitude_start", "Latitude_start", "plz_start", "Longitude_end", "Latitude_end", "plz_end"]].head(50))
    return p_df


def calc_postalcode_for_coords(p_row, p_number_of_points):
    index = p_row.name
    print("Checking coordinates: ", (index/p_number_of_points*100).round(1), "% (", index, "/", p_number_of_points, ") \r", end="")
    point = shapely.Point(p_row["Longitude"], p_row["Latitude"])
    for iter_plz, iter_shape in plz_value.items():
        if type(iter_shape) == list:
            for poly in iter_shape:
                if poly.contains(point):
                    return iter_plz
        else:
            if iter_shape.contains(point):
                return iter_plz


def map_postalcode_to_original_df(p_row, p_df_original, p_number_of_nurem_points):
    index = p_row.name
    print("Mapping postalcodes to dataset: ", (index / p_number_of_nurem_points * 100).round(1), "% (", index, "/", p_number_of_nurem_points,
          ") \r", end="")
    p_df_original.loc[
        (p_df_original["Longitude_start"] == p_row["Longitude"]) & (
         p_df_original["Latitude_start"] == p_row["Latitude"]), "plz_start"] = p_row["plz"]
    p_df_original.loc[
        (p_df_original["Longitude_end"] == p_row["Longitude"]) & (
         p_df_original["Latitude_end"] == p_row["Latitude"]), "plz_end"] = p_row["plz"]

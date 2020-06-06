import pandas as pd
import geopy.distance as geodis
import numpy as np

# imports needed for alternative distance methods
# import geopandas as gpd
# from shapely.geometry import Point


def additional_feature_creation(p_df_trips):
    """Adds the following additional features to the df:
        - Weekend: boolean whether it was a weekend day (True if it was a saturday or sunday)
        - Duration: describes the trip duration in minutes
        - Month, Day, Hour, Minute, Day_of_year for start and end of trip
        - Season: winter: 1, spring: 2, summer: 3, fall: 4

    Args:
        p_df_trips (DataFrame): DataFrame with trip data from nuremberg
    Returns:
        df_trips (DataFrame): DataFrame with added columns [Weekend, Duration]
    """
    # Calculating if trip was on a weekend, storing a boolean
    # Check which day of the week the given date is
    # Counting from 0 to 6 (0=monday, 1=tuesday, ...) a 5 or 6 means it was a saturday or sunday
    # So storing if dayofweek is bigger than 4 builds a weekend boolean
    p_df_trips['Weekend'] = (p_df_trips['Start_Time'].dt.dayofweek > 4)

    # Calculation trip duration of each trip
    # Calculating simply (end time - start time) for trip duration would
    #   build the duration in the format 'X days HH:MM:SS.sssssssss'
    # So to better calculate with this value in the future,
    #   lets get the total seconds of the duration and
    #   divide it be 60 to manually calculate duration in minutes
    #   and round it to two decimals
    p_df_trips['Duration'] = ((p_df_trips['End_Time'] - p_df_trips['Start_Time']).dt.total_seconds() / 60.0).round(2)
    # TODO: Fix that bad style right here:
    pd.options.mode.chained_assignment = None
    p_df_trips["Month_start"] = p_df_trips["Start_Time"].dt.month
    p_df_trips["Day_start"] = p_df_trips["Start_Time"].dt.day
    p_df_trips["Hour_start"] = p_df_trips["Start_Time"].dt.hour
    p_df_trips["Minute_start"] = p_df_trips["Start_Time"].dt.minute
    p_df_trips["Day_of_year_start"] = p_df_trips["Start_Time"].dt.dayofyear
    p_df_trips["Season"] = (p_df_trips["Month_start"] % 12 + 3) // 3  # winter: 1, spring: 2, summer: 3, fall: 4
    p_df_trips["Month_end"] = p_df_trips["End_Time"].dt.month
    p_df_trips["Day_end"] = p_df_trips["End_Time"].dt.day
    p_df_trips["Hour_end"] = p_df_trips["End_Time"].dt.hour
    p_df_trips["Minute_end"] = p_df_trips["Start_Time"].dt.minute
    p_df_trips["Day_of_year_end"] = p_df_trips["End_Time"].dt.dayofyear

    return p_df_trips


def quick_create_dist(p_df):
    """Calculates distances of start and end to university and therefore the direction of a trip.

    Calculates the distance between the start point of the trip and the university (49.452210, 11.079575).
    Calculates the distance between the end point of the trip and the university.
    Calculates if the trip ends nearer to the university than it started.
    Direction is true if trip goes in direction to university and false otherwise
    Args:
        p_df (DataFrame): DataFrame with trip data from nuremberg
    Returns:
        p_df (DataFrame): DataFrame with added columns [Dist_start, Dist_end, Direction]
    """

    """
    # ===== METHOD 1: Geopandas =====
    
    uni = Point(11.079575, 49.452210)
    # uni_geometry = [Point(11.079575, 49.452210)]
    # gdf_uni = gpd.GeoDataFrame(geometry=uni, crs={'init':'epsg:4326', 'unit':'m'})
    # gdf_uni = gdf_uni.to_crs(epsg=3395)
    # gdf_uni = gdf_uni.to_crs(epsg=32633)

    start_pos = gpd.points_from_xy(p_df["Longitude_start"], p_df["Latitude_start"])
    end_pos = gpd.points_from_xy(p_df["Longitude_end"], p_df["Latitude_end"])
    gdf_start_pos = gpd.GeoDataFrame(p_df, geometry=start_pos, crs={'init':'epsg:4326', 'units':'m'})
    # gdf_start_pos = gdf_start_pos.to_crs(epsg=25832)
    # gdf_start_pos = gdf_start_pos.to_crs(epsg=3310)
    # gdf_start_pos = gdf_start_pos.to_crs(epsg=3395)
    # gdf_start_pos = gdf_start_pos.to_crs(epsg=32633)
    print(gdf_start_pos)
    gdf_end_pos = gpd.GeoDataFrame(p_df, geometry=end_pos, crs={'init':'epsg:4326', 'units':'m'})
    # gdf_end_pos = gdf_end_pos.to_crs(epsg=25832)
    # gdf_end_pos = gdf_end_pos.to_crs(epsg=3310)
    # gdf_end_pos = gdf_end_pos.to_crs(epsg=3395)
    # gdf_end_pos = gdf_end_pos.to_crs(epsg=32633)
    print(gdf_end_pos)
    p_df["Dist_start"] = gdf_start_pos.distance(uni)*6373
    p_df["Dist_end"] = gdf_end_pos.distance(uni)*6373
    p_df["Direction"] = p_df["Dist_start"] > p_df["Dist_end"]
    print(p_df[["Dist_start", "Dist_end", "Direction"]])
    
    # ===== Troubles with converting coordinates to km, even with different epsg codes =====
    # ===== So direction is correct, but distances are crappy values =====
    """

    """
    # ===== METHOD 2: Manually calculate distances =====

    uni_lng = 11.079575
    uni_lat = 49.452210
    p_df["Dist_start"] = haversine_vectorize(p_df["Longitude_start"], p_df["Latitude_start"], uni_lng, uni_lat)
    p_df["Dist_end"] = haversine_vectorize(p_df["Longitude_end"], p_df["Latitude_end"], uni_lng, uni_lat)
    p_df["Direction"] = p_df["Dist_start"] > p_df["Dist_end"]
    print(p_df[["Dist_start", "Dist_end", "Direction"]])

    # ===== Much more performant, but slightly different distances, seems to calculate wrong values =====
    """

    # ===== METHOD 3: Geopy =====

    uni = (11.079575, 49.452210)
    p_df['Dist_start'] = p_df.apply(lambda row: geodis.distance((row['Longitude_start'],
                                                                 row['Latitude_start']), uni).km, axis=1)
    p_df['Dist_end'] = p_df.apply(lambda row: geodis.distance((row['Longitude_end'],
                                                               row['Latitude_end']), uni).km, axis=1)
    p_df['Direction'] = p_df['Dist_start'] > p_df['Dist_end']  # to uni: True, away: False
    print(p_df[["Dist_start", "Dist_end", "Direction"]])

    # ===== Most inperformant, but most precise and simple solution =====

    return p_df


def haversine_vectorize(lon1, lat1, lon2, lat2):
    """ Calculates manually distances between two points by longitude and latitude, returns distance in km

    Args:
        lon1: longitude of first point
        lat1: latitude of first point
        lon2: longitude of second point
        lat2: latitude of second point
    Returns:
        distance between points in km
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    newlon = lon2 - lon1
    newlat = lat2 - lat1

    haver_formula = np.sin(newlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(newlon / 2.0) ** 2

    dist = 2 * np.arcsin(np.sqrt(haver_formula))
    km = 6367 * dist  # 6367 for distance in KM for miles use 3958
    return km

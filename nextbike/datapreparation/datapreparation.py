import pandas as pd
import numpy as np
from .. import io
from . import plz
# remove in the end, just for testing the time

NUREMBERG_CITY_LONG = 49.452030
NUREMBERG_CITY_LAT = 11.076750
DISTANCE = 0.07 * 1
# TODO: Fix that bad style right here:
pd.options.mode.chained_assignment = None

# --> only PLZ = 189112

# ToDo : comment all functions in detail
def data_preparation(df_original):
    """clean data and create trips from it

    This method drops all duplicates from the raw data.
    It creates trip data for each bike by joining the rows with corresponding start and end .
    Args:
        df_original (DataFrame): DataFrame from raw csv
    Returns:
        df_merged (DataFrame): DataFrame of cleaned trip data
    """

    # print("Prepare columns...")
    # create new DataFrame from received DF and remove unnecessary columns.
    # df_clean = df_original.drop(["Unnamed: 0",
    #                              "p_spot",
    #                              "p_place_type",
    #                              "p_bike",
    #                              "b_bike_type",
    #                              "p_bikes",
    #                              "p_uid",
    #                              # "p_number"
    #                              ], axis=1)

    # Renaming the columns
    # df_clean.rename({"p_lat": "Latitude",
    #                  "p_lng": "Longitude",
    #                  "p_name": "Place",
    #                  "b_number": "Bike Number"}, axis=1, inplace=True)

    # Drop Duplicates and creating new df with only unique files
    print("Drop duplicates...")
    df_original.drop_duplicates(subset=df_original.columns.difference(["p_lng", "p_lat"]), inplace=True)

    # Drop trip first/last
    print("Filter on start/end...")

    df_clean_unique_trip = df_original[(df_original["trip"] == "start") | (df_original["trip"] == "end")]

    df_clean_unique_trip.sort_values(["b_number", "datetime"], inplace=True)

    # We do not check, whether every bike has same amount of starts and ends because merge only returns valid entries

    # TODO: check bike number of start and end trips
    # TODO: check first position of df must be a "start"
    # TODO: check last position of df must be a "end"

    # Theoretischer FAll: Special FAll: Die allererste BikeID faengt mit Ende an und das allerletzte Bike enden mit Start
    # 2: Fahrrad x: endet mit Start und Fahrrad x+1 startet mit End
    # 3: Loesungsansaetze: dadurch, dass die Fahrraeder nach Zeit sortiert, ist die wahrscheinlich gering

    # Eliminate Noise
    print("Eliminate booking errors...")
    # compare the trip value of each row with the row above (to check for multiple start entries)
    sr_previous_entry_differs = (df_clean_unique_trip['trip'] != df_clean_unique_trip['trip'].shift())

    # compare the trip value of each row with the row below (to check for multiple end entries)
    sr_next_entry_differs = (df_clean_unique_trip['trip'] != df_clean_unique_trip['trip'].shift(-1))

    # check if entries are valid
    #   just checking if previous or next entry differs does not work!
    #   because otherwise we would pick
    #       either the first start and first end
    #       or the last start and the last end of each trip
    #   but we want to have the first start and the last end of each trip
    #   so for starts only the previous entry is relevant, for ends only next entry is relevant
    sr_valid_start = ((df_clean_unique_trip['trip'] == 'start') & (sr_previous_entry_differs == True))
    sr_valid_end = ((df_clean_unique_trip['trip'] == 'end') & (sr_next_entry_differs == True))

    # only take valid trip entries (valid starts and valid ends)
    df_final = df_clean_unique_trip[(sr_valid_start == True) | (sr_valid_end == True)]

    print("Merge corresponding start and end...")

    df_start = df_final[df_final["trip"] == "start"]
    df_end = df_final[df_final["trip"] == "end"]

    df_end.reset_index(drop=True, inplace=True)
    df_start.reset_index(drop=True, inplace=True)

    df_merged = df_start.merge(df_end, left_on=df_start.index, right_on=df_end.index, suffixes=('_start', '_end'))
    df_merged.drop(["key_0",
                    "trip_start",
                    "b_number_end",
                    "trip_end"], axis=1, inplace=True)
    df_merged.rename({"datetime_start": "Start Time",
                      "b_number_start": "Bike Number",
                      "datetime_end": "End Time",
                      "p_number_start": "Start Place_id",
                      "p_number_end": "End Place_id",
                      "p_lat_start": "Latitude_start",
                      "p_lng_start": "Longitude_start",
                      "p_lat_end": "Latitude_end",
                      "p_lng_end": "Longitude_end",
                      "p_name_start": "Place_start",
                      "p_name_end": "Place_end"
                      },
                     axis=1, inplace=True)

    return df_merged


def additional_feature_creation(df_trips):
    """Adds the following additional features to the df:
        - Weekend: boolean whether it was a weekend day (True if it was a saturday or sunday)
        - Duration: describes the trip duration in minutes

    Args:
        df_trips (DataFrame): DataFrame with trip data from nuremberg
    Returns:
        df_trips (DataFrame): DataFrame with added columns [Weekend, Duration]
    """
    # Calculating if trip was on a weekend, storing a boolean
    print("Adding column 'Weekend'...")
    # First convert start time string into datetime object
    df_trips['Start Time'] = pd.to_datetime(df_trips['Start Time'])
    # Then check which day of the week the given date is
    # Counting from 0 to 6 (0=monday, 1=tuesday, ...) a 5 or 6 means it was a saturday or sunday
    # So storing if dayofweek is bigger than 4 builds a weekend boolean
    df_trips['Weekend'] = (df_trips['Start Time'].dt.dayofweek > 4)
    print("DONE adding 'Weekend'")

    # Calculation trip duration of each trip
    print("Adding column 'Duration'...")
    # First also convert end time string into datetime object
    df_trips['End Time'] = pd.to_datetime(df_trips['End Time'])
    # Calculating simply (end time - start time) for trip duration would
    #   build the duration in the format 'X days HH:MM:SS.sssssssss'
    # So to better calculate with this value in the future,
    #   lets get the total seconds of the duration and
    #   divide it be 60 to manually calculate duration in minutes
    #   and round it to two decimals
    df_trips['Duration'] = ((df_trips['End Time'] - df_trips['Start Time']).dt.total_seconds() / 60.0).round(2)
    print("DONE adding 'Duration'")

    return df_trips


def calculate_aggregate_statistics(df_trips):
    """Calculates the following aggregate statistics and saves them as png file:
        - aggr_stats_whole_df: mean and standard deviation of the whole df, of all weekdays and of all weekends
        - calls plot_and_save_aggregate_stats method to do the same on months, days and hours

    Args:
        df_trips (DataFrame): DataFrame with trip data from nuremberg
    Returns:
        no return
    """

    print("Drop trips < 1min & upper 2%...")
    # Calculate lower and upper bounds for duration (in minutes)
    # Hardcode lower bound because trips of about 2 minutes may be relevant
    lower_duration_bound = 1.0
    upper_duration_bound = df_trips["Duration"].quantile(0.98)
    # Drop values out of duration bounds
    df_trips = df_trips[(df_trips["Duration"] > lower_duration_bound) & (df_trips["Duration"] < upper_duration_bound)]

    # Split into weekends df and weekdays df
    df_weekends = df_trips[df_trips['Weekend'] == True]
    df_weekdays = df_trips[df_trips['Weekend'] == False]

    aggr_stats = {
        "mean_all": df_trips['Duration'].mean(),
        "std_all": df_trips['Duration'].std(),
        "mean_weekends": df_weekends['Duration'].mean(),
        "std_weekends": df_weekends['Duration'].std(),
        "mean_weekdays": df_weekdays['Duration'].mean(),
        "std_weekdays": df_weekdays['Duration'].std()
    }

    df_aggr_stats = pd.DataFrame.from_dict(aggr_stats, orient="index")
    fig = df_aggr_stats.plot(kind='barh', figsize=(16, 16), fontsize=20).get_figure()
    io.save_fig(fig, 'aggr_stats_whole_df.png')

    # TODO: Fix that bad style right here:
    pd.options.mode.chained_assignment = None
    df_trips["month"] = df_trips["Start Time"].dt.month
    df_trips["day"] = df_trips["Start Time"].dt.day
    df_trips["hour"] = df_trips["Start Time"].dt.hour
    plot_and_save_aggregate_stats(df_trips)

    print("DONE calculating aggregate statistics!")


def plot_and_save_aggregate_stats(df_trips):
    """Aggregates on different time slots.
        - Calculates count, mean and standard deviation
        - Plots them as horizontal bar chart
        - Saves plot as png file

    Args:
        df_trips (DataFrame): Modified DataFrame with trip data from nuremberg (with additional columns month, day and hour)
    Returns:
        no return
    """

    for time_to_aggregate_on in ["month", "day", "hour"]:
        sr_counts = df_trips[["Duration", time_to_aggregate_on]].groupby(by=time_to_aggregate_on).count()
        fig = sr_counts.plot(kind='barh', figsize=(16, 16), fontsize=22).get_figure()
        io.save_fig(fig, 'counts_' + time_to_aggregate_on + '.png')
        sr_means = df_trips[["Duration", time_to_aggregate_on]].groupby(by=time_to_aggregate_on).mean()
        fig = sr_means.plot(kind='barh', figsize=(16, 16), fontsize=22).get_figure()
        io.save_fig(fig, 'means_' + time_to_aggregate_on + '.png')
        sr_stds = df_trips[["Duration", time_to_aggregate_on]].groupby(by=time_to_aggregate_on).std()
        fig = sr_stds.plot(kind='barh', figsize=(16, 16), fontsize=22).get_figure()
        io.save_fig(fig, 'stds_' + time_to_aggregate_on + '.png')


def only_nuremberg_square(df):
    """TODO:What does this method do?

    Args:
        df (DataFrame): DataFrame with trip data
    Returns:
        df_nuremberg (DataFrame): DataFrame with trip data from nuremberg
    """
    # DropTrips outside of Nuremberg, depending on their Start and End Point
    # Information: Nuremberg City Center: Lat: 49.452030, Long: 11.076750
    # --> https://www.laengengrad-breitengrad.de/gps-koordinaten-von-nuernberg
    # Borders of our Data:
    # Latitude North: 49,56 --> ca. 13.6 km
    # Todo: Constants are defined on the top

    north = NUREMBERG_CITY_LONG + DISTANCE
    south = NUREMBERG_CITY_LONG - DISTANCE
    west = NUREMBERG_CITY_LAT + DISTANCE
    east = NUREMBERG_CITY_LAT - DISTANCE
    print("Remove bookings outside nuremberg...")
    # create column "outside" with information:
    # inside --> start and end is inside of our defined square

    bol_start = (df["Latitude_start"] < north) & (df["Latitude_start"] > south) & (df["Longitude_start"] < west) & (
            df["Longitude_start"] > east)

    bol_end = (df["Latitude_end"] < north) & (df["Latitude_end"] > south) & (df["Longitude_end"] < west) & (
            df["Longitude_end"] > east)

    df['inside'] = np.where(bol_end & bol_start, True, False)
    df_nuremberg = df[df["inside"] == True]

    return df_nuremberg


def only_nuremberg_plz(df):
    """TODO:What does this method do?

    Args:
        df (DataFrame): DataFrame with trip data
    Returns:
        df_nuremberg (DataFrame): DataFrame with trip data from nuremberg
    """
    # DropTrips outside of Nuremberg with no PLZ, depending on their Start
    # Information: Nuremberg City Center: Lat: 49.452030, Long: 11.076750
    # --> https://www.laengengrad-breitengrad.de/gps-koordinaten-von-nuernberg

    # adding plz to df
    print("Add PLZ to trip and drop trips without start or end PLZ")
    # Add PLZ to trip and drop trips without start or end PLZ

    # TODO: resolve names of methods or file
    df_plz = plz.plz(df)

    # add start plz
    df_nurem = df_plz.dropna(axis=0)
    # add end plz
    df_nurem = plz.plz_end(df_nurem)

    df_nurem = df_nurem.dropna(axis=0)

    return df_nurem

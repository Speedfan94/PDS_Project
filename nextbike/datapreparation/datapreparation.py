import os
import pandas as pd
import numpy as np
from datetime import datetime
# remove in the end, just for testing the time

NUREMBERG_CITY_LONG = 49.452030
NUREMBERG_CITY_LAT = 11.076750
DISTANCE = 0.07 * 1


# 05--> 191502
# 07--> 196178
# 09-_> 196441
# 10--> 196486
# *1 -> 196505
# *2 -> 197192
# *3 -> 197199
# *4 -> 197202

# ToDo : comment all functions in detail
def datapreparation(df_original):
    # Prepare Columns
    print("Prepare columns...")
    # create new DataFrame from received DF and remove unnecessary columns.
    df_clean = df_original.drop(["Unnamed: 0",
                        "p_spot",
                        "p_place_type",
                        "p_bike",
                        "b_bike_type",
                        "p_bikes",
                        "p_uid",
                        "p_number"], axis=1)

    # Renaming the columns
    df_clean.rename({"p_lat": "Latitude",
                     "p_lng": "Longitude",
                     "p_name": "Place",
                     "b_number": "Bike Number"}, axis=1, inplace=True)

    # Drop Duplicates and creating new df with only unique files
    print("Drop duplicates...")
    df_clean_unique = df_clean.drop_duplicates(subset=df_clean.columns.difference(["Longitude", "Latitude"]))

    # Drop trip first/last
    print("Filter on start/end...")
    df_clean_unique_trip = df_clean_unique[(df_clean_unique["trip"] == "start") | (df_clean_unique["trip"] == "end")]

    df_clean_unique_trip.sort_values(["Bike Number", "datetime"], inplace=True)

    # We do not check, whether every bike has same amount of starts and ends because merge only returns valid entries

    # TODO: check bike number of start and end trips
    # TODO: check first position of df must be a "start"
    # TODO: check last position of df must be a "end"

    # Theoretischer FAll: Special FAll: Die allererste BikeID faengt mit Ende an und das allerletzte Bike enden mit Start
    # 2: Fahrrad x: endet mit Start und Fahrrad x+1 startet mit End
    # 3: Loesungsansaetze: dadurch, dass die Fahrraeder nach Zeit sortiert, ist die wahrscheinlich gering

    # Eliminate Noise
    print("Eliminating Noise")
    # create series with
    # check for multiple start entries
    sr_noise_start = (df_clean_unique_trip['trip'] != df_clean_unique_trip['trip'].shift())
    df_clean_unique_trip['valid_start'] = sr_noise_start

    # check for multiple end entries
    sr_noise_end = (df_clean_unique_trip['trip'] != df_clean_unique_trip['trip'].shift(-1))
    df_clean_unique_trip['valid_end'] = sr_noise_end

    # check if entries are valid
    valid_start_entry = ((df_clean_unique_trip['trip'] == 'start') & (df_clean_unique_trip['valid_start'] == True))
    valid_end_entry = ((df_clean_unique_trip['trip'] == 'end') & (df_clean_unique_trip['valid_end'] == True))
    df_clean_unique_trip['valid_trip'] = valid_start_entry | valid_end_entry

    # only take valid trip entries and drop validation values
    df_final = df_clean_unique_trip[df_clean_unique_trip['valid_trip'] == True]
    df_final.drop(['valid_start', 'valid_end', 'valid_trip'], axis=1, inplace=True)

    print("DONE Noise elimination")
    # split, reindex, merge
    print("Creating Final Trip DataFrame")

    df_start = df_final[df_final["trip"] == "start"]
    df_end = df_final[df_final["trip"] == "end"]

    df_end.reset_index(drop=True, inplace=True)
    df_start.reset_index(drop=True, inplace=True)
    # ToDo: check runtime mit itime --> Niklas
    # df_end["index"] = range(0, len(df_end))
    # df_start["index"] = range(0, len(df_start))

    df_merged = df_start.merge(df_end, left_on=df_start.index, right_on=df_end.index, suffixes=('_start', '_end'))
    df_merged.drop(["key_0",
                    "trip_start",
                    "Bike Number_end",
                    "trip_end"], axis=1, inplace=True)
    df_merged.rename({"datetime_start": "Start Time", "Bike Number_start": "Bike Number", "datetime_end": "End Time"},
                     axis=1, inplace=True)

    print("DONE creating final trip dataframe")
    print(df_merged.head())

    return df_merged


def additional_feature_creation(df_trips):
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
    # TODO: round up to 2 decimals
    df_trips['Duration'] = (df_trips['End Time'] - df_trips['Start Time']).dt.total_seconds() / 60.0
    print("DONE adding 'Duration'")

    return df_trips


def get_aggregate_statistics(df_trips):
    print("Calculating duration mean...")
    # TODO: kick out durations <1min & >2h / 3-5%
    # TODO: stats per month/day/hour
    mean = df_trips['Duration'].mean()
    std = df_trips['Duration'].std()

    # Split into weekends df and weekdays df
    df_weekends = df_trips[df_trips['Weekend'] == True]
    df_weekdays = df_trips[df_trips['Weekend'] == False]

    # Calculate mean and std for weekends
    mean_weekends = df_weekends['Duration'].mean()
    std_weekends = df_weekends['Duration'].std()

    # Calculate mean and std for weekdays
    mean_weekdays = df_weekdays['Duration'].mean()
    std_weekdays = df_weekdays['Duration'].std()

    # package all aggregate statistics into a dictionary to return them
    aggr_stats = {
        "mean": mean,
        "std": std,
        "mean_weekends": mean_weekends,
        "std_weekends": std_weekends,
        "mean_weekdays": mean_weekdays,
        "std_weekdays": std_weekdays
    }
    print("DONE calculating aggregate statistics!")

    return aggr_stats


def onlynuremberg(df):
    # DropTrips outside of Nuremberg, depending on their Start and End Point
    # Information: Nuremberg City Center: Lat: 49.452030, Long: 11.076750
    # --> https://www.laengengrad-breitengrad.de/gps-koordinaten-von-nuernberg
    # Borders of our Data:
    # Latitude North: 49,56 --> ca. 13.6 km
    # Constants are defined on the top

    north = NUREMBERG_CITY_LONG + DISTANCE
    # Latitude South:
    south = NUREMBERG_CITY_LONG - DISTANCE
    # Longitude West:
    west = NUREMBERG_CITY_LAT + DISTANCE
    # Longitude East:
    east = NUREMBERG_CITY_LAT - DISTANCE
    print("Startet OnlyNuremberg for Removing Positions outside Nuremberg")
    # create column "outside" with information:
    # inside --> start and end is inside of our defined square

    bol_start = (df["Latitude_start"] < north) & (df["Latitude_start"] > south) & (df["Longitude_start"] < west) & (
                df["Longitude_start"] > east)

    bol_end = (df["Latitude_end"] < north) & (df["Latitude_end"] > south) & (df["Longitude_end"] < west) & (
            df["Longitude_end"] > east)

    # method 1
    # start = datetime.now()
    df['inside'] = np.where(bol_end & bol_start, True, False)
    # print("Method 1: " + str(datetime.now() - start))
    # --> Method 1: 0:00:00.001999
    # --> Method 2: 0:00:00.005000
    print("This is df before removing NUREMBERG")
    print(df)

    # method 2
    # start = datetime.now()
    # df['inside2'] = False
    # df.loc[(bol_end & bol_start), 'inside2'] = True
    # print("Method 2: " + str(datetime.now() - start))
    # --> method 2 is slower

    df_nuremberg = df[df["inside"] == True]

    return df_nuremberg

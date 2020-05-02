import os
import pandas as pd
import numpy as np
from datetime import datetime
# remove in the end, just for testing the time


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


def onlynuremberg(df):
    # DropTrips outside of Nuremberg, depending on their Start and End Point
    # Information: Nuremberg City Center: 49.460983, 11.061859
    # Borders of our Data:
    # Latitude North: 50 --> ca.
    north = 50
    # Latitude South:
    south = 49
    # Longitude West:
    west = 11.6
    # Longitude East:
    east = 10.5
    print("Startet OnlyNuremberg for Removing Positions outside Nuremberg")
    # create column with information:
    # inside --> start and end is outside of our defined square

    bol_start = (df["Latitude_start"] < north) & (df["Latitude_start"] > south) & (df["Longitude_start"] < west) & (
                df["Longitude_start"] > east)

    bol_end = (df["Latitude_end"] < north) & (df["Latitude_end"] > south) & (df["Longitude_end"] < west) & (
            df["Longitude_end"] > east)

    # method 1
    start = datetime.now()
    df['outside'] = np.where(bol_end & bol_start, 'inside', 'outside')
    print("Method 1: " + str(datetime.now() - start))
    # --> Method 1: 0:00:00.085453
    # --> Method 2: 0:00:00.124790

    print(df)

    # method 2
    # start = datetime.now()
    # df['outside2'] = 'outside'
    # df.loc[(bol_end & bol_start), 'outside3'] = 'inside'
    # print("Method 2: " + str(datetime.now() - start))
    # --> method 2 is slower

    df_nuremberg = df[df["outside"] == "inside"]

    return df_nuremberg

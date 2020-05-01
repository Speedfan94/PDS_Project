import os
import pandas as pd


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

    # TODO: Add Sort by Aufsteigend

    # ToDo: Same amount of start and ends?

    # ToDO: check, whether every bike has same amount of starts and ends
    # ToDo: check Zugehörigkeit zum entsprechenden Bike
    # ToDo: check first position of df must be a "Start"

    # Theoretischer FAll: Special FAll: Die allererste BikeID fängt mit Ende an und das allerletzte Bike enden mit Start
    # 2: Fahrrad x: endet mit Start und Fahrrad x+1 startet mit End
    # 3: Lösungsansätze: dadurch, dass die Fahrräder nach Zeit sortiert, ist die wahrscheinlich gering

    # Eliminate Noise
    print("Eliminate Noise")
    # create series with
    sr_noise = (df_clean_unique_trip['trip'] != df_clean_unique_trip['trip'].shift())
    df_clean_unique_trip['Noisy Entry'] = sr_noise
    df_final = df_clean_unique_trip[df_clean_unique_trip["Noisy Entry"] == True]



    # split, reindex, merge
    print("Create Master DataFrame")

    df_start = df_final[df_final["trip"] == "start"]
    df_end = df_final[df_final["trip"] == "end"]

    df_end.reset_index(drop=True)
    df_start.reset_index(drop=True)
    #ToDo: check runtime mit itime --> Niklas
    #df_end["index"] = range(0, len(df_end))
    #df_start["index"] = range(0, len(df_start))

    df_merged = df_start.merge(df_end, left_on=df_start.index, right_on=df_end.index, suffixes=('_start', '_end'))
    df_merged.drop(["key_0",
                    "trip_start",
                    "Noisy Entry_start",
                    "index_start",
                    #"Bike Number_end",
                    "trip_end",
                    "Noisy Entry_end",
                    "index_end"], axis=1, inplace=True, errors="ignore")
    df_merged.rename({"datetime_start": "Start Time", "Bike Number_start": "Bike Number", "datetime_end": "End Time"},
                     axis=1)


    print("DONE")
    print(df_merged.head())



    return df_merged

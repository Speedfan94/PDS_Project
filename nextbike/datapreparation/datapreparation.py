import os
import pandas as pd

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

    # Eliminate Noise
    print("Eliminate Noise")
    sr_noise = (df_clean_unique_trip['trip'] != df_clean_unique_trip['trip'].shift())
    df_clean_unique_trip['Noisy Entry'] = sr_noise
    df_final = df_clean_unique_trip[df_clean_unique_trip["Noisy Entry"] == True]

    # split, reindex, merge
    print("Create Master DataFrame")
    df_s = df_final[df_final["trip"] == "start"]
    df_e = df_final[df_final["trip"] == "end"]
    df_e["index"] = range(0, len(df_e))
    df_s["index"] = range(0, len(df_s))
    df_merged = df_s.merge(df_e, left_on=df_s["index"], right_on=df_e["index"], suffixes=('_start', '_end'))
    df_merged.drop(["key_0",
                    "trip_start",
                    "Noisy Entry_start",
                    "index_start",
                    "Bike Number_end",
                    "trip_end",
                    "Noisy Entry_end",
                    "index_end"], axis=1, inplace=True, errors="ignore")
    df_merged.rename({"datetime_start": "Start Time", "Bike Number_start": "Bike Number", "datetime_end": "End Time"},
                     axis=1)



    print("DONE")
    print(df_merged.head())

import pandas as pd

NUREMBERG_CITY_LONG = 49.452030
NUREMBERG_CITY_LAT = 11.076750
DISTANCE = 0.07 * 1


def data_cleaning(p_df_original):
    """clean data and create trips from it

    This method drops all duplicates from the raw data.
    It creates trip data for each bike by joining the rows with corresponding start and end.

    Args:
        p_df_original (DataFrame): DataFrame from raw csv
    Returns:
        df_merged (DataFrame): DataFrame of cleaned trip data
    """
    # Drop Duplicates and creating new df with only unique files
    p_df_original = p_df_original.drop_duplicates(subset=p_df_original.columns.difference(["p_lng", "p_lat"]))
    # Drop trip first/last
    df_clean_unique_trip_unsorted = p_df_original[(p_df_original["trip"] == "start") | (p_df_original["trip"] == "end")]
    df_clean_unique_trip = df_clean_unique_trip_unsorted.sort_values(["b_number", "datetime"])
    # We do not check, whether every bike has same amount of starts and ends because merge only returns valid entries
    # compare the trip value of each row with the row above (to check for multiple start entries)
    sr_previous_entry_differs = (df_clean_unique_trip["trip"] != df_clean_unique_trip["trip"].shift())
    # compare the trip value of each row with the row below (to check for multiple end entries)
    sr_next_entry_differs = (df_clean_unique_trip["trip"] != df_clean_unique_trip["trip"].shift(-1))
    # check if entries are valid
    #   just checking if previous or next entry differs does not work!
    #   because otherwise we would pick
    #       either the first start and first end
    #       or the last start and the last end of each trip
    #   but we want to have the first start and the last end of each trip
    #   so for starts only the previous entry is relevant, for ends only next entry is relevant
    sr_valid_start = ((df_clean_unique_trip["trip"] == "start") & sr_previous_entry_differs)
    sr_valid_end = ((df_clean_unique_trip["trip"] == "end") & sr_next_entry_differs)
    # only take valid trip entries (valid starts and valid ends)
    df_final = df_clean_unique_trip[sr_valid_start | sr_valid_end]

    # drop first entry if dataframe does not start with a start entry
    if df_final.iloc[0]["trip"] != "start":
        df_final = df_final.drop(0)
    # drop last entry if dataframe does not end with an end entry
    index_last = len(df_final) - 1
    if df_final.iloc[index_last]["trip"] != "end":
        df_final = df_final.drop(index_last)

    # split dataframe into start and end entries and merge them into trip dataframe
    df_start = df_final[df_final["trip"] == "start"].drop("trip", axis=1)
    df_end = df_final[df_final["trip"] == "end"].drop("trip", axis=1)
    df_end = df_end.reset_index(drop=True)
    df_start = df_start.reset_index(drop=True)
    df_merged = df_start.merge(
        df_end,
        left_on=df_start.index,
        right_on=df_end.index,
        suffixes=("_start", "_end")
    )

    # Only keep trips, which where merged correctly
    df_merged = df_merged[df_merged["b_number_start"] == df_merged["b_number_end"]]

    df_merged = df_merged.drop(
        ["key_0",
         "b_number_end"], axis=1
    )
    df_merged = df_merged.rename(
        {"datetime_start": "Start_Time",
         "b_number_start": "Bike_Number",
         "datetime_end": "End_Time",
         "p_number_start": "Start_Place_id",
         "p_number_end": "End_Place_id",
         "p_lat_start": "Latitude_start",
         "p_lng_start": "Longitude_start",
         "p_lat_end": "Latitude_end",
         "p_lng_end": "Longitude_end",
         "p_name_start": "Place_start",
         "p_name_end": "Place_end"
         },
        axis=1
    )

    return df_merged


def drop_noise(p_df_trips):
    """clean data from noisy entires

    This method cleans the trip data by deleting trips which are shorter than 1 minute.
    Additionally only trips with a duration which is inside the 90% quantile are used.

    Args:
        p_df_trips (DataFrame): DataFrame of trip data
    Returns:
        p_df_trips (DataFrame): DataFrame of trip data without noisy entries
    """
    # Calculate lower and upper bounds for duration (in minutes)
    # Hardcode lower bound because trips of about 2 minutes may be relevant
    lower_duration_bound = 1.0
    upper_duration_bound = p_df_trips["Duration"].quantile(0.90)
    # upper_duration_bound = 5.0
    # Drop values out of duration bounds
    p_df_trips = p_df_trips[
        (p_df_trips["Duration"] > lower_duration_bound) & (p_df_trips["Duration"] < upper_duration_bound)
    ]
    return p_df_trips

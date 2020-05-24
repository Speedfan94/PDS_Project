import pandas as pd

NUREMBERG_CITY_LONG = 49.452030
NUREMBERG_CITY_LAT = 11.076750
DISTANCE = 0.07 * 1

# TODO: Fix that bad style right here:
pd.options.mode.chained_assignment = None


def data_cleaning(p_df_original):
    """clean data and create trips from it

    This method drops all duplicates from the raw data.
    It creates trip data for each bike by joining the rows with corresponding start and end .

    Args:
        p_df_original (DataFrame): DataFrame from raw csv
    Returns:
        df_merged (DataFrame): DataFrame of cleaned trip data
    """
    # Drop Duplicates and creating new df with only unique files
    p_df_original.drop_duplicates(subset=p_df_original.columns.difference(["p_lng", "p_lat"]), inplace=True)
    # Drop trip first/last
    df_clean_unique_trip = p_df_original[(p_df_original["trip"] == "start") | (p_df_original["trip"] == "end")]
    df_clean_unique_trip.sort_values(["b_number", "datetime"], inplace=True)
    # We do not check, whether every bike has same amount of starts and ends because merge only returns valid entries

    # TODO: check bike number of start and end trips
    # TODO: check first position of df must be a "start"
    # TODO: check last position of df must be a "end"

    # Theoretischer FAll: Special FAll: Die allererste BikeID faengt mit Ende an und das allerletzte Bike enden mit
    # Start 2: Fahrrad x: endet mit Start und Fahrrad x+1 startet mit End 3: Loesungsansaetze: dadurch,
    # dass die Fahrraeder nach Zeit sortiert, ist die wahrscheinlich gering

    # Eliminate Noise
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
    sr_valid_start = ((df_clean_unique_trip['trip'] == 'start') & sr_previous_entry_differs)
    sr_valid_end = ((df_clean_unique_trip['trip'] == 'end') & sr_next_entry_differs)
    # only take valid trip entries (valid starts and valid ends)
    df_final = df_clean_unique_trip[sr_valid_start | sr_valid_end]
    df_start = df_final[df_final["trip"] == "start"].drop("trip", axis=1)
    df_end = df_final[df_final["trip"] == "end"].drop("trip", axis=1)
    df_end.reset_index(drop=True, inplace=True)
    df_start.reset_index(drop=True, inplace=True)
    df_merged = df_start.merge(
        df_end,
        left_on=df_start.index,
        right_on=df_end.index,
        suffixes=('_start', '_end')
    )
    df_merged.drop(
        ["key_0",
         "b_number_end",
         "Unnamed: 0_start",
         "Unnamed: 0_end"], axis=1, inplace=True
    )
    df_merged.rename(
        {"datetime_start": "Start Time",
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
        axis=1, inplace=True
    )

    return df_merged


# TODO: Add docstring
def drop_noise(p_df_trips):
    # Calculate lower and upper bounds for duration (in minutes)
    # Hardcode lower bound because trips of about 2 minutes may be relevant
    lower_duration_bound = 1.0
    upper_duration_bound = p_df_trips["Duration"].quantile(0.90)
    # Drop values out of duration bounds
    p_df_trips = p_df_trips[
        (p_df_trips["Duration"] > lower_duration_bound) & (p_df_trips["Duration"] < upper_duration_bound)]
    return p_df_trips

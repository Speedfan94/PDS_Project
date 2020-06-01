import pandas as pd
import geopy.distance as geodis

# Todo: 30 lines for 2 lines of logic?
def additional_feature_creation(p_df_trips):
    """Adds the following additional features to the df:
        - Weekend: boolean whether it was a weekend day (True if it was a saturday or sunday)
        - Duration: describes the trip duration in minutes

    Args:
        p_df_trips (DataFrame): DataFrame with trip data from nuremberg
    Returns:
        df_trips (DataFrame): DataFrame with added columns [Weekend, Duration]
    """
    # Calculating if trip was on a weekend, storing a boolean
    # Check which day of the week the given date is
    # Counting from 0 to 6 (0=monday, 1=tuesday, ...) a 5 or 6 means it was a saturday or sunday
    # So storing if dayofweek is bigger than 4 builds a weekend boolean
    p_df_trips['Weekend'] = (p_df_trips['Start Time'].dt.dayofweek > 4)

    # Calculation trip duration of each trip
    # Calculating simply (end time - start time) for trip duration would
    #   build the duration in the format 'X days HH:MM:SS.sssssssss'
    # So to better calculate with this value in the future,
    #   lets get the total seconds of the duration and
    #   divide it be 60 to manually calculate duration in minutes
    #   and round it to two decimals
    p_df_trips['Duration'] = ((p_df_trips['End Time'] - p_df_trips['Start Time']).dt.total_seconds() / 60.0).round(2)
    # TODO: Fix that bad style right here:
    pd.options.mode.chained_assignment = None
    p_df_trips["month_start"] = p_df_trips["Start Time"].dt.month
    p_df_trips["day_start"] = p_df_trips["Start Time"].dt.day
    p_df_trips["hour_start"] = p_df_trips["Start Time"].dt.hour
    p_df_trips["minute_start"] = p_df_trips["Start Time"].dt.minute
    p_df_trips["dayofyear_start"] = p_df_trips["Start Time"].dt.dayofyear
    p_df_trips["Season"] = (p_df_trips["month_start"] % 12 + 3) // 3  # winter: 1, spring: 2, summer: 3, fall: 4
    p_df_trips["month_end"] = p_df_trips["End Time"].dt.month
    p_df_trips["day_end"] = p_df_trips["End Time"].dt.day
    p_df_trips["hour_end"] = p_df_trips["End Time"].dt.hour
    p_df_trips["minute_end"] = p_df_trips["End Time"].dt.minute
    p_df_trips["dayofyear_end"] = p_df_trips["End Time"].dt.dayofyear


    return p_df_trips


# TODO: Find another place for feature creation
# TODO: Add docstring
def quick_create_dist(p_df):
    uni = (49.452210, 11.079575)
    p_df['Dist_start'] = p_df.apply(lambda row: geodis.distance((row['Latitude_start'],
                                                                 row['Longitude_start']), uni).km, axis=1)
    p_df['Dist_end'] = p_df.apply(lambda row: geodis.distance((row['Latitude_end'],
                                                               row['Longitude_end']), uni).km, axis=1)
    p_df['Direction'] = p_df['Dist_start'] > p_df['Dist_end']  # to uni: True, away: False
    return p_df

import pandas as pd
from datetime import datetime


def cast_datetime(p_df, p_datetime_columns, p_as_numeric_value=False):
    """ changes the format of the given column to a datetime-format.

    Args:
        p_df (dataframe):  dataframe, in which one or more columns
        p_datetime_columns (List of Strings):   List of Strings with the column name, in which the datetime is in
        p_as_numeric_value (Boolean):   Variable, which decides wether it is saved as a numeric value or in datetime
                                        format.
    Returns:
        p_df (dataframe):
    """
    for column in p_datetime_columns:
        p_df[column] = pd.to_datetime(p_df[column], format="%Y-%m-%d %H:%M:%S")

        if p_as_numeric_value:
            p_df[column] = p_df[column].values.astype(int)

    return p_df


def print_time_for_step(p_step_name, p_start_time_step):
    """Calculates time needed for current step and prints it out.
    Returns start time for next step

    Args:
        p_step_name: name/label of the previous step
        p_start_time_step (datetime): start time of current step
    Returns:
        start_time_next_step (float): start time of the next step
    """
    start_time_next_step = datetime.now().replace(microsecond=0)
    print("=== TIME FOR "+p_step_name+":", (start_time_next_step - p_start_time_step))
    return start_time_next_step

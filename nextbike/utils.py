import pandas as pd


def cast_datetime(p_df, p_datetime_columns, p_as_numeric_value=False):
    """ Return the data-path, depending on the location, from where it is called.

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

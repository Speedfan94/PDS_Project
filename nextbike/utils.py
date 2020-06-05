import pandas as pd


# TODO: add docstring
def cast_datetime(p_df, p_datetime_columns, p_as_numeric_value=False):
    for column in p_datetime_columns:
        p_df[column] = pd.to_datetime(p_df[column], format="%Y-%m-%d %H:%M:%S")

        if p_as_numeric_value:
            p_df[column] = p_df[column].values.astype(int)

    return p_df

import pandas as pd

# TODO: add docstring
def cast_datetime(pDf, datetime_columns, as_numeric_value=False):
    for column in datetime_columns:
        pDf[column] = pd.to_datetime(pDf[column], format="%Y-%m-%d %H:%M:%S")

        if as_numeric_value:
            pDf[column] = pDf[column].values.astype(int)

    return pDf

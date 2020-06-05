import pandas as pd
from nextbike import io
from .. import utils


def read_weather():
    # TODO Docstring
    df_weather = io.input.read_csv(p_filename="weather019.csv", p_io_folder="input")
    df_weather = df_weather.drop(["Unnamed: 0"], axis=1)
    utils.cast_datetime(df_weather, ["Date"])
    return df_weather

def add_weather(df_trips):
    # TODO Docstring

    df_weather = read_weather()

    df_trips['rain'] = df_trips.apply(lambda row: geodis.distance((row['Latitude_end'],
                                                               row['Longitude_end']), uni).km, axis=1)
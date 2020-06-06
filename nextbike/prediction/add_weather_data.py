import pandas as pd
from nextbike import io
from nextbike import utils


def read_weather():
    """reads in the weather.csv data file in /data/input/weather.csv

        Args:

        Returns:
            df_weather (DataFrame): it includes weather data with rain and temperature
        """
    df_weather = io.input.read_csv(p_filename="weather2019.csv", p_io_folder="input")
    utils.cast_datetime(df_weather, ["Date"])
    return df_weather


def add_weather(df_trips):
    """Adds the weather data depending on the date and time to the trips DataFrame

        Args:
            df_trips (DataFrame): DataFrame from cli.py which contains already the cleaned trips
        Returns:
            df_trips (DataFrame): DataFrame, which contains the trips with the weather data
        """
    df_weather = read_weather()
    df_trips["Start_Time"] = pd.to_datetime(df_trips["Start_Time"], format="%Y-%m-%d %H:%M:%S")
    df_trips["Date"] = (df_trips["Start_Time"].dt.date.astype(str) + " " + df_trips["Start_Time"].dt.hour.astype(
        str) + ":00:00")
    df_trips["Date"] = pd.to_datetime(df_trips["Date"], format="%Y-%m-%d %H:%M:%S")
    df_trips = pd.merge(df_trips, df_weather, how="left", on=["Date"])
    df_trips = df_trips.drop("Date", axis=1)

    return df_trips

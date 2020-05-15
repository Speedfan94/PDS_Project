import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from .. import io
import folium
from folium import plugins
import json
import vincenty
from shapely.geometry import shape, Point


# import seaborn as sns

def visualize_plz(df):
    # For the summer month (i.e., June, August, or September) with most trips, visualize the number
    # of started trips per PLZ region (you’ll have to find geo data for that yourselves!) in a map.

    print()


def visualize_moment(df):
    # TODO: Motivated BEE <3: Search usefull time for this plot
    # For one moment in time, visualize the number of bikes at fixed stations meaningfully.
    most_bookings = df.groupby(by="Start Time").count()["Bike Number"].sort_values().tail(1).index
    # choose Datetime 3

    # ToDo: Stations with no bikes right now visualize in grey

    for row in most_bookings:
        print("Compute Moment at: " + row + " ...")

        df_moment = df[df["Start Time"] == row]
        # get stations without Start Place_id != 0.0
        df_help = pd.DataFrame(df_moment[df_moment["Start Place_id"] != 0.0]["Start Place_id"])
        # get unique long lat for stations
        df_long_lat = df_moment.drop_duplicates("Start Place_id")[["Start Place_id", "Latitude_start", "Longitude_start"]]
        df_long_lat = df_long_lat[df_long_lat["Start Place_id"] != 0.0]
        df_stations = df_help.merge(df_long_lat, how="left", on="Start Place_id")
        # get bikes with with Start Place_id = 0.0
        df_free = df_moment[df_moment["Start Place_id"] == 0.0][["Start Place_id", "Longitude_start", "Latitude_start"]]
        # get unused stations at given date
        df_helper_unused = df.drop_duplicates("Start Place_id")[["Start Place_id", "Latitude_start", "Longitude_start"]]
        df_helper_unused = df_helper_unused[df_helper_unused["Start Place_id"] != 0.0]
        df_unused = df_helper_unused.append(df_stations).drop_duplicates(keep=False)
        plot_map(df_stations, df_free, df_unused, row)


def plot_map(pDf_stations, pDf_free, pDf_unused, pStr_datetime):
    # Todo: Class with constants
    north = 49.485
    east = 11.13
    south = 49.425
    west = 11.02

    # read img nuremberg
    # https://www.openstreetmap.org/export#map=12/49.4522/11.0770
    # nuremberg_png = plt.imread("nextbike/data/input/nuremberg_v2_hum.png")
    nuremberg_png = plt.imread(io.get_path("nuremberg_v2_hum.png", "input"))

    fig, ax = plt.subplots(figsize=(10, 10))
    free = ax.scatter(pDf_free["Longitude_start"],
                      pDf_free["Latitude_start"],
                      zorder=1, alpha=0.2, c="r", s=14)

    station = ax.scatter(pDf_stations["Longitude_start"],
                         pDf_stations["Latitude_start"],
                         zorder=1, alpha=0.08, c="b", s=30)

    unused = ax.scatter(pDf_unused["Longitude_start"],
                         pDf_unused["Latitude_start"],
                         zorder=1, alpha=0.5, c="grey", s=30)

    ax.set_title('Bikes at ' + str(pStr_datetime))
    ax.set_xlim(west, east)
    ax.set_ylim(north, south)
    plt.legend((station, free, unused), ("Bikes at Station", "Free Bikes", "Unused Stations"), loc="upper left")
    ax.imshow(nuremberg_png, zorder=0, extent=[west, east, north, south], aspect='equal')
    plt.savefig(io.get_path(str(pStr_datetime).replace(":", "-") + ".png", "output"), dpi=300)


def visualize_heatmap(df):
    # Create a heatmap based on an interesting aspect of the data, e.g., end locations of trips shortly
    # before the start of a major public event.
    # https://alysivji.github.io/getting-started-with-folium.html

    stations = df[pd.to_datetime(df["End Time"], format="%Y-%m-%d").dt.date == dt.date(year=2019, month=8, day=24)]

    # ToDo: Maybe filter out 0.0 ids and duplicated places
    # todo: variable time to plot heatmap
    print("Create Heatmap for " + str(dt.date(year=2019, month=12, day=24)) + "...")
    m = folium.Map([49.452030, 11.076750], zoom_start=13)

    # mark each station as a point
    for index, row in stations.iterrows():
        folium.CircleMarker([row['Latitude_end'], row['Longitude_end']],
                            radius=3,
                            popup=row['Place_end'],
                            fill_color="#3db7e4",
                            color="#3db7e4"
                            ).add_to(m)

    # convert to (n, 2) nd-array format for heatmap
    stationArr = stations[['Latitude_end', 'Longitude_end']].values

    # plot heatmap
    m.add_child(plugins.HeatMap(stationArr, radius=20))

    m.save(io.get_path("One-Day-in-Nuremberg.html", "output"))



def visualize_plz(df):
    # Visualizes the number of started trip for each zip code region for the month with the most trips

    # Changing format from object to DateTime
    df["Start Time"] = pd.to_datetime(df["Start Time"])

    # find the month with the most trips:
    df["month"] = df["Start Time"].dt.month

    # finding the month with most trips in the month
    month_most = df.groupby(by="month").count().idxmax()["Start Time"]

    df_biggest_month = df[df["month"] == month_most]
    # prints the number of trips per zip code
    df_map = df_biggest_month.groupby(by="plz_start").count().sort_values(by="Start Time", ascending=True).reset_index()

    print("PLZ Map " + str(month_most))
    m = folium.Map([49.452030, 11.076750], zoom_start=11)

    df_map["plz"] = df_map["plz_start"].astype(str)

    folium.Choropleth(
        # geo_data=f"../nextbike/data/input/postleitzahlen-nuremberg.geojson",
        geo_data=f'{io.get_path("postleitzahlen-nuremberg.geojson", "input")}',
        name="choropleth",
        data=df_map,
        columns=["plz", "month"],
        key_on='feature.properties.plz',
        legend_name='Trips per zip code',
        fill_color='YlGnBu',
        # color="white",
        fill_opacity=0.7,
        line_opacity=0.5,
        # smooth_factor=0
    ).add_to(m)

    df_stations = df.drop_duplicates("Start Place_id", keep="first")

    for index, row in df_stations.iterrows():
        folium.CircleMarker([row['Latitude_start'], row['Longitude_start']],
                            radius=3,
                            popup=[row['Place_start'], row["Latitude_start"]],
                            fill_color="#3db7e4",
                            color="#3db7e4",

                            ).add_to(m)

    folium.LayerControl().add_to(m)

    m.save(io.get_path("Month_Nuremberg.html", "output"))
    print(df_map)


def visualize_distribution(df):
    # Visualize the distribution of trip lengths per month. Compare the distributions to normal
    # distributions with mean and standard deviation as calculated before (1.d))

    print()


def visualize_more(df):
    # These visualizations are the minimum requirement. Use more visualizations wherever it makes
    # sense.

    print()

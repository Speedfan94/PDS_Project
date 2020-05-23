import folium
from folium import plugins
from .. import io
import datetime as dt


def visualize_stations_moment(df):
    """bikes at fixed stations at given point in time

    Args:
        df (DataFrame): DataFrame with trip data from nuremberg
    Returns:
        no return
    """
    print("Visualize Moment...")
    time_moment = "2019-10-01 08:00:00"
    df_relevant = df[df["End Time"] <= time_moment]  # exclude all trips during and after that moment
    df_relevant = df_relevant[df_relevant["End Place_id"] != 0.0]  # exclude free bikes
    # hold for each station just the last entry sorted by datetime
    df_relevant_stations = df_relevant.sort_values(by=["End Place_id", "End Time"]).drop_duplicates("End Place_id", keep="last")
    df_relevant_stations.reset_index(drop=True, inplace=True)
    m = folium.Map(location=[49.452030, 11.076750], zoom_start=13)
    for i in range(len(df_relevant_stations)):
        folium.Marker(
            location=[df_relevant_stations["Latitude_end"].iloc[i], df_relevant_stations["Longitude_end"].iloc[i]],
            radius=5,
            tooltip=str(df_relevant_stations["Place_end"].iloc[i]),
            popup=folium.Popup(
                "<b>Station Name:</b><br>" +
                str(df_relevant_stations["Place_end"].iloc[i]) +
                "<br><b>Bikes at Station: </b>" +
                str(df_relevant_stations["p_bikes_end"].iloc[i]), max_width=400
            ),
            fill_color="red",
            color="red"
        ).add_to(m)
    m.save(io.get_path(filename="Moment.html", io_folder="output", subfolder="data_plots"))


def visualize_heatmap(df):
    """heatmap for the 24th of December by searching for nearby trip ends.

    Args:
        df (DataFrame): DataFrame with trip data from nuremberg
    Returns:
        no return
    """

    # Create a heatmap based on an interesting aspect of the data, e.g., end locations of trips shortly
    # before the start of a major public event.
    # https://alysivji.github.io/getting-started-with-folium.html

    stations = df[df["End Time"].dt.date == dt.date(year=2019, month=12, day=24)]

    # ToDo: Maybe filter out 0.0 ids and duplicated places
    # todo: variable time to plot heatmap
    print("Visualize Heatmap...")
    m = folium.Map([49.452030, 11.076750], zoom_start=13)

    # mark each station as a point
    for index, row in stations.iterrows():
        folium.CircleMarker([row['Latitude_end'], row['Longitude_end']],
                            radius=3,
                            popup=folium.Popup(
                                "<b>Station Name:</b><br>" +
                                str(row["Place_end"]),
                                max_width=400
                            ),
                            fill_color="#3db7e4",
                            color="#3db7e4"
                            ).add_to(m)

    # convert to (n, 2) nd-array format for heatmap
    stationArr = stations[['Latitude_end', 'Longitude_end']].values

    # plot heatmap
    m.add_child(plugins.HeatMap(stationArr, radius=20))

    m.save(io.get_path(filename="One-Day-in-Nuremberg.html", io_folder="output", subfolder="data_plots"))


def visualize_plz(df):
    """Plots a choropleth graph on a map based on the number of started trips in each zip code region.
    This is be done for the month with the most trips

    Args:
        df (DataFrame): DataFrame with trip data from nuremberg
    Returns:
        no return
    """
    print("Visualize Postcode Sectors...")
    # Visualizes the number of started trip for each zip code region for the month with the most trips

    # find the month with the most trips:
    df["month"] = df["Start Time"].dt.month

    # finding the month with most trips in the month
    month_most = df.groupby(by="month").count().idxmax()["Start Time"]

    df_biggest_month = df[df["month"] == month_most]
    # prints the number of trips per zip code
    df_map = df_biggest_month.groupby(by="plz_start").count().sort_values(by="Start Time", ascending=True).reset_index()

    m = folium.Map([49.452030, 11.076750], zoom_start=11)

    df_map["plz"] = df_map["plz_start"].astype(str)

    folium.Choropleth(
        geo_data=f'{io.get_path(filename="postleitzahlen-nuremberg.geojson", io_folder="input")}',
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

    m.save(io.get_path(filename="Month_Nuremberg.html", io_folder="output", subfolder="data_plots"))
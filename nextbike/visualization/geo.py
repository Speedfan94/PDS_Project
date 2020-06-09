import folium
from folium import plugins
from .. import io
import datetime as dt

uni = (49.458395, 11.085980)


def visualize_stations_moment(p_df, p_mode=""):
    """bikes at fixed stations at given point in time

    Args:
        p_df (DataFrame): DataFrame with trip data from nuremberg
        p_mode (str): String which contains modus in which program is running
    Returns:
        no return
    """
    time_moment = "2019-10-01 08:00:00"
    df_relevant = p_df[p_df["End_Time"] <= time_moment]  # exclude all trips during and after that moment
    df_relevant = df_relevant[df_relevant["End_Place_id"] != 0.0]  # exclude free bikes
    # hold for each station just the last entry sorted by datetime
    df_relevant_stations = df_relevant.sort_values(by=["End_Place_id", "End_Time"]).drop_duplicates("End_Place_id",
                                                                                                    keep="last")
    df_relevant_stations = df_relevant_stations.reset_index(drop=True)
    m = folium.Map(location=[49.452030, 11.076750], zoom_start=13)

    for index, row in df_relevant_stations.iterrows():
        folium.CircleMarker(
            location=[row["Latitude_end"], row["Longitude_end"]],
            radius=row["p_bikes_end"],
            tooltip=str(row["Place_end"]),
            popup=folium.Popup(
                "<b>Station Name:</b><br>" +
                str(row["Place_end"]) +
                "<br><b>Bikes at Station: </b>" +
                str(row["p_bikes_end"]), max_width=400
            ),
            fill_color="blue",
            color="blue"
        ).add_to(m)

    folium.Marker(
        location=uni,
        popup=folium.Popup(
            "<b>University:</b><br>", max_width=400
        ),
        tooltip="University",
        icon=folium.Icon(color='black', icon="home"),
    ).add_to(m)
    m.save(
        io.get_path(
            p_filename="One_Moment_In_Time_BikesPerStation_Circle" + p_mode + ".html",
            p_io_folder="output",
            p_sub_folder1="data_plots",
            p_sub_folder2="geo"
        )
    )


def visualize_heatmap(p_df, p_mode=""):
    """heatmap for the 24th of December by searching for nearby trip ends.

    Args:
        p_df (DataFrame): DataFrame with trip data from nuremberg
        p_mode (str): String which contains modus in which program is running
    Returns:
        no return
    """

    # Create a heatmap based on an interesting aspect of the data, e.g., end locations of trips shortly
    # before the start of a major public event.
    # https://alysivji.github.io/getting-started-with-folium.html

    stations = p_df[p_df["End_Time"].dt.date == dt.date(year=2019, month=12, day=24)]

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

    folium.Marker(
        location=uni,
        popup=folium.Popup(
            "<b>University:</b><br>", max_width=400
        ),
        tooltip="University",
        icon=folium.Icon(color="black", icon="home"),
    ).add_to(m)

    m.save(
        io.get_path(
            p_filename="One_Day_in_Nuremberg_EndedTrips_HeatMap" + p_mode + ".html",
            p_io_folder="output",
            p_sub_folder1="data_plots",
            p_sub_folder2="geo"
        )
    )


def visualize_postalcode(p_df, p_mode=""):
    """Plots a choropleth graph on a map based on the number of started trips in each postal code code region.
    This is be done for the month with the most trips

    Args:
        p_df (DataFrame): DataFrame with trip data from nuremberg
        p_mode (str): String which contains modus in which program is running
    Returns:
        no return
    """
    # finding the month with most trips in the month
    month_most = p_df.groupby(by="Month_start").count().idxmax()["Start_Time"]

    df_biggest_month = p_df[p_df["Month_start"] == month_most]
    # prints the number of trips per postal code code
    df_map = df_biggest_month.groupby(
        by="Postalcode_start"
    ).count().sort_values(
        by="Start_Time", ascending=True
    ).reset_index()

    m = folium.Map([49.452030, 11.076750], zoom_start=13)

    df_map["Postalcode"] = df_map["Postalcode_start"].astype(str)

    folium.Choropleth(
        geo_data=f'{io.get_path(p_filename="postleitzahlen-nuremberg.geojson", p_io_folder="input")}',
        name="choropleth",
        data=df_map,
        columns=["Postalcode", "Month_start"],
        key_on='feature.properties.plz',
        legend_name='Trips per postal code',
        fill_color='YlGnBu',
        fill_opacity=0.7,
        line_opacity=0.5,
    ).add_to(m)

    p_df = p_df[p_df["Start_Place_id"] != 0.0]
    # p_df = p_df[p_df["End_Place_id"] != 0.0]
    df_stations = p_df.drop_duplicates("Start_Place_id", keep="first")
    # df_stations = p_df.drop_duplicates("End_Place_id", keep="first")


    for index, row in df_stations.iterrows():
        folium.CircleMarker(
            [row['Latitude_start'], row['Longitude_start']],
            radius=3,
            popup=folium.Popup(
                "<b>Station Name:</b><br>" +
                str(row["Place_start"]), max_width=400),
            fill_color="#3db7e4",
            color="#3db7e4",

        ).add_to(m)

    folium.LayerControl().add_to(m)

    folium.Marker(
        location=uni,
        popup=folium.Popup(
            "<b>University</b><br>", max_width=400
        ),
        tooltip="University",
        icon=folium.Icon(color='black', icon="home"),
    ).add_to(m)

    m.save(
        io.get_path(
            p_filename="Month_Nuremberg_TripsStartet_per_Zip_Code_Included_Stations_choropleth" + p_mode + ".html",
            p_io_folder="output",
            p_sub_folder1="data_plots",
            p_sub_folder2="geo"
        )
    )


def visualize_trips_per_month(p_df, p_mode=""):
    """TODO:Docstring

    Args:
        p_df (DataFrame): DataFrame with trip data from nuremberg
        p_mode (str): String which contains modus in which program is running
    Returns:
        no return
    """

    df_day = p_df[(p_df["Month_start"] == 12) & (p_df["Day_start"] == 24)]

    m = folium.Map([49.452030, 11.076750], zoom_start=14)

    for index, row in df_day.iterrows():
        folium.PolyLine(locations=[(row["Latitude_start"], row["Longitude_start"]),
                                   (row["Latitude_end"], row["Longitude_end"])],
                        line_opacity=0.5, weight=1.5).add_to(m)

    for index, row in df_day.iterrows():
        folium.CircleMarker(
            location=[row["Latitude_start"], row["Longitude_start"]],
            tooltip=str(row["Place_start"]),
            radius=3,
            popup=folium.Popup(
                "<b>Station Name:</b><br>" +
                str(row["Place_end"]) +
                "<br><b>Bikes at Station: </b>" +
                str(row["p_bikes_end"]), max_width=400
            ),
            # icon=folium.Icon(color='green'),
            fill_color="green",
            color="green"
        ).add_to(m)

    for index, row in df_day.iterrows():
        folium.CircleMarker(
            location=(row["Latitude_end"], row["Longitude_end"]),
            tooltip=str(row["Place_end"]),
            radius=3,
            popup=folium.Popup(
                "<b>Station Name:</b><br>" +
                str(row["Place_end"]) +
                "<br><b>Bikes at Station: </b>" +
                str(row["p_bikes_end"]), max_width=400
            ),
            # icon=folium.Icon(color='red'),
            fill_color="red",
            color="red"
        ).add_to(m)

    folium.LayerControl().add_to(m)

    folium.Marker(
        location=uni,
        popup=folium.Popup(
            "<b>University</b><br>", max_width=400
        ),
        tooltip="University",
        icon=folium.Icon(color='black', icon="home"),
    ).add_to(m)

    m.save(
        io.get_path(
            p_filename="Trips_One_Day_In_Time" + p_mode + ".html",
            p_io_folder="output",
            p_sub_folder1="data_plots",
            p_sub_folder2="geo"
        )
    )

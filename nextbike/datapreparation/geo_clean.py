from nextbike import io
import geopandas as gpd

# In which state is the **center** of Germany
# Note reverse notation compared to before
# searchpoint = Point(-0.109970, 51.529163)


def only_nuremberg(p_df):
    """Calculates corresponding zip codes to each data point and
    filters out all data points not in nuremberg (based on zip codes)

    Args:
        p_df (DataFrame): DataFrame with trip data
    Returns:
        df_nuremberg (DataFrame): DataFrame with trip data from nuremberg
    """
    # DropTrips outside of Nuremberg with no PLZ, depending on their Start
    # Information: Nuremberg City Center: Lat: 49.452030, Long: 11.076750
    # --> https://www.laengengrad-breitengrad.de/gps-koordinaten-von-nuernberg

    # adding plz to df
    # Add PLZ to trip and drop trips without start or end PLZ

    # ==========
    # 1. load plz data from geojson file
    path_postalcodes_geojson = io.get_path(p_filename="postleitzahlen-nuremberg.geojson", p_io_folder="input")
    gdf_postalcodes = gpd.read_file(path_postalcodes_geojson)

    # ==========
    # 2. create geometry points of START points (with longitude and latitude)
    gdf_geo_start_pos = gpd.GeoDataFrame(p_df,
                                         geometry=gpd.points_from_xy(p_df["Longitude_start"],
                                                                     p_df["Latitude_start"]))

    # ==========
    # 3. join trips and postalcode dfs on START geo points (to build plz_start)
    gdf_sjoined_start = gpd.sjoin(gdf_geo_start_pos, gdf_postalcodes, how="inner", op="within")
    # clean up unnecessary columns added by sjoin, rename plz to plz_start
    df_with_start_plz = gdf_sjoined_start.drop(["geometry", "index_right", "note"], axis=1)
    df_with_start_plz.rename({"plz": "plz_start"}, axis=1, inplace=True)

    # ==========
    # 4. create geometry points of END points (with longitude and latitude)
    gdf_geo_end_pos = gpd.GeoDataFrame(df_with_start_plz,
                                       geometry=gpd.points_from_xy(df_with_start_plz["Longitude_end"],
                                                                   df_with_start_plz["Latitude_end"]))

    # ==========
    # 5. join trips and postalcode dfs on END geo points (to build plz_end)
    gdf_sjoined_end = gpd.sjoin(gdf_geo_end_pos, gdf_postalcodes, how="inner", op="within")
    # clean up unnecessary columns added by sjoin, rename plz to plz_end
    df_with_all_plz = gdf_sjoined_end.drop(["geometry", "index_right", "note"], axis=1)
    df_with_all_plz.rename({"plz": "plz_end"}, axis=1, inplace=True)

    return df_with_all_plz

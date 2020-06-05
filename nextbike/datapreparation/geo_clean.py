from nextbike import io
import geopandas as gpd

# In which state is the **center** of Germany
# Note reverse notation compared to before
# searchpoint = Point(-0.109970, 51.529163)


def only_nuremberg(p_df):
    """Calculates corresponding postalcodes for start and end points of each trip
    and filters out all data points not in nuremberg (based on postalcodes).
    Merges trip data with postalcode infos based on the position and kicks out
    unnecessary postalcode infos.

    Args:
        p_df (DataFrame): DataFrame with trip data
    Returns:
        df_nuremberg (DataFrame): DataFrame with trip data from nuremberg including postalcodes
    """
    # drop trips outside of Nuremberg with no postalcode
    # add postalcode to trip and drop trips without start or end postalcode
    # information: nuremberg city center: Lat: 49.452030, Long: 11.076750
    # --> https://www.laengengrad-breitengrad.de/gps-koordinaten-von-nuernberg

    # ==========
    # 1. load postalcodes data from geojson file
    path_postalcodes_geojson = io.get_path(p_filename="postleitzahlen-nuremberg.geojson", p_io_folder="input")
    gdf_postalcodes = gpd.read_file(path_postalcodes_geojson)

    # ==========
    # 2. create geometry points of START points (with longitude and latitude)
    gdf_geo_start_pos = gpd.GeoDataFrame(p_df,
                                         geometry=gpd.points_from_xy(p_df["Longitude_start"],
                                                                     p_df["Latitude_start"]))
    # set crs to same as postalcode to fix warnings
    gdf_geo_start_pos.crs = gdf_postalcodes.crs

    # ==========
    # 3. join trips and postalcode dfs on START geo points (to build Postalcode_start)
    gdf_sjoined_start = gpd.sjoin(gdf_geo_start_pos, gdf_postalcodes, how="inner", op="within")
    # clean up unnecessary columns added by sjoin, rename plz to Postalcode_start
    df_with_start_postalcode = gdf_sjoined_start.drop(["geometry", "index_right", "note"], axis=1)
    df_with_start_postalcode = df_with_start_postalcode.rename({"plz": "Postalcode_start"}, axis=1)

    # ==========
    # 4. create geometry points of END points (with longitude and latitude)
    gdf_geo_end_pos = gpd.GeoDataFrame(df_with_start_postalcode,
                                       geometry=gpd.points_from_xy(df_with_start_postalcode["Longitude_end"],
                                                                   df_with_start_postalcode["Latitude_end"]))
    # set crs to same as postalcode to fix warnings
    gdf_geo_end_pos.crs = gdf_postalcodes.crs

    # ==========
    # 5. join trips and postalcode dfs on END geo points (to build Postalcode_end)
    gdf_sjoined_end = gpd.sjoin(gdf_geo_end_pos, gdf_postalcodes, how="inner", op="within")
    # clean up unnecessary columns added by sjoin, rename plz to Postalcode_end
    df_with_all_postalcodes = gdf_sjoined_end.drop(["geometry", "index_right", "note"], axis=1)
    df_with_all_postalcodes = df_with_all_postalcodes.rename({"plz": "Postalcode_end"}, axis=1)

    return df_with_all_postalcodes

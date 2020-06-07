from .. import io
from .. import prediction
from .. import visualization
from .. import utils
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



def create_dummies(p_df):
    """Create dummie values for all usefull non numerical features.

    Args:
        p_df (DataFrame): Dataframe of trips in nuremberg
    Returns:
        df_dummies (DataFrame): Dataframe with dummies instead of booleans
    """
    p_spot_start = p_df["p_spot_start"].astype(int)
    p_bike_start = p_df["p_bike_start"].astype(int)
    weekend = p_df["Weekend"].astype(int)
    p_df = p_df.drop(["p_spot_start", "p_bike_start", "Weekend"], axis=1)
    df_dummies = pd.concat([p_df, p_spot_start, p_bike_start, weekend], axis=1)
    return df_dummies


def create_new_features(p_X, weather):
    """Create new features which are usefull for prediction performance.

    Example methods for feature engineering could be found here:
    https://de.devoteam.com/blog-post/bedeutung-der-feature-engineering-methoden-2/
    Args:
        p_X (DataFrame): Dataframe of existing features (matrix)
    Returns:
        p_X (DataFrame): Dataframe with existing and new added features (matrix)
    """
    # TODO Weather Data adding implement here, by triggering

    if weather:
        print("Adding Weather Data...")
        p_X = prediction.add_weather(p_X)
        p_X["Rain_squared"] = np.power(p_X["rain(mm)"], 3)
        p_X["Temp_squared"] = np.power(p_X["Temp(C)"], 3)
        p_X = p_X.drop(
            ["rain(mm)",
             "Temp(C)"
             ],
            axis=1
        )

    p_X["Hour_squared"] = np.square(p_X["Hour_start"])
    p_X["Day_squared"] = np.square(p_X["Day_start"])
    p_X["Month_squared"] = np.square(p_X["Month_start"])
    p_X["Minute_squared"] = np.square(p_X["Minute_start"])
    # p_X["Minute_squared"] = np.power(p_X["Minute_start"], 3)
    p_X["Weekend_squared"] = np.square(p_X["Weekend"])
    p_X["Latitude_squared"] = np.square(p_X["Latitude_start"])
    p_X["Longitude_squared"] = np.square(p_X["Longitude_start"])
    return p_X


def drop_end_information(p_df, direction_needed=False):
    """Drop all information of end of trips which are not used for duration prediction and model training.

    It does not make any sense to use information on trip ends to predict the duration of a trip.
    The idea is to predict the duration of a trip directly after the start/ booking.
    Args:
        p_df (DataFrame): Dataframe of existing features
    Returns:
        df (DataFrame): Dataframe only including start information
    """
    df = p_df.drop(
        ["p_spot_end",
         "p_place_type_end",
         "End_Time",
         "p_uid_end",
         "p_bikes_end",
         "Latitude_end",
         "b_bike_type_end",
         "Place_end",
         "End_Place_id",
         "Longitude_end",
         "p_bike_end",
         "Postalcode_end",
         "Month_end",
         "Day_end",
         "Hour_end",
         "Minute_end",
         "Day_of_year_end",
         "Dist_end"
         ], axis=1)

    if direction_needed == False:
        df = df.drop("Direction", axis=1)
    return df


def drop_features(p_df):
    """Testing method for feature selection which drops features that should not be included in prediction and
    training of the ML models.

    This method is used to influence the performance of prediction on validation set.
    Args:
        p_df (DataFrame): Dataframe of existing features
    Returns:
        df (DataFrame): Dataframe only including start information
    """
    df = p_df
    do_it = True
    if do_it:
        df = p_df.drop(
            ["p_uid_start",
             # "p_place_type_start",
             # "p_bikes_start",
             # "Month_start",
             # "Day_start",
             # "Start_Place_id",
             # "p_spot_start",
             # "Latitude_start",
             # "Longitude_start",
             # "Bike Number",
             # "b_bike_type_start",
             # "Hour_start",
             # "Day_of_year_start",
             # "Season",
             # "Postalcode_start",
             # "Dist_start",
             # "p_bike_start",
             # "Weekend",
             # "Hour_squared",
             # "Day_squared",
             # "Month_squared",
             # "Weekend_squared",
             # "Minute_squared",
             # "Latitude_squared",
             # "Longitude_squared"
             ],
            axis=1
        )
    return df


def scale(p_X_train, p_scaler_name):
    """Scale all independent variables/ regressors  in DataFrame

    Args:
        p_X_train (DataFrame): DataFrame of independent variables/ regressors (matrix)
        p_scaler_name (str): Name at which the Scaler is saved

    Returns:
        X_train_scaled (DataFrame): DataFrame with scaled independent variables/ regressors (matrix)
    """
    st_scaler = StandardScaler()
    # fit scaler only on training set not on test set
    st_scaler.fit(p_X_train)
    io.save_object(st_scaler, p_scaler_name+".pkl")
    X_train_scaled = st_scaler.transform(p_X_train)
    return X_train_scaled


def do_pca(p_X_scaled_train, p_number_components, p_filename):
    """Do a PCA to analyse which components to take for further predictions.

    PCA creates components from existing features by analysing the equality of variance.
    Multicolinearity will be nearly eliminated. => unbiasedness of models e.g. linear regression
    Args:
        p_X_scaled_train (DataFrame): DataFrame of scaled data
        p_number_components (int): Number of components which PCA should create
        p_filename (str): Name of file of PCA
    Returns:
        X_train_scaled_pca (DataFrame): DataFrame of components
    """
    pca = PCA(n_components=p_number_components)
    pca.fit(p_X_scaled_train)
    pca_explained_variance = pca.explained_variance_ratio_
    # print("Var explained:", pca_explained_variance)
    print("Sum var explained", sum(pca_explained_variance))
    io.save_object(pca, p_filename+".pkl")
    X_train_scaled_pca = pca.transform(p_X_scaled_train)
    visualization.math_descriptive.plot_pca_components(pca_explained_variance, p_filename)
    return X_train_scaled_pca


def add_weather(df_trips):
    """Adds the weather data depending on the date and time to the trips DataFrame

        Args:
            df_trips (DataFrame): DataFrame from cli.py which contains already the cleaned trips
        Returns:
            df_trips (DataFrame): DataFrame, which contains the trips with the weather data
        """
    df_weather = io.input.read_csv(p_filename="weather2019.csv", p_io_folder="input")
    utils.cast_datetime(df_weather, ["Date"])
    df_trips["Start_Time"] = pd.to_datetime(df_trips["Start_Time"], format="%Y-%m-%d %H:%M:%S")
    df_trips["Date"] = (df_trips["Start_Time"].dt.date.astype(str) + " " + df_trips["Start_Time"].dt.hour.astype(
        str) + ":00:00")
    df_trips["Date"] = pd.to_datetime(df_trips["Date"], format="%Y-%m-%d %H:%M:%S")
    df_trips = pd.merge(df_trips, df_weather, how="left", on=["Date"])
    df_trips = df_trips.drop("Date", axis=1)

    return df_trips

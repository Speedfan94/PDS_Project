import click
from . import io
from . import datapreparation
from . import visualization
from . import prediction
from . import utils


@click.command()
@click.option('--train/--no-train', default=True, help="Train the model.")
@click.option('--clean/--no-clean', default=True, help="Clean the data.")
@click.option('--viso/--no-viso', default=True, help="Visualize the data.")
@click.option('--pred/--no-pred', default=True, help="Predict with model.")
@click.option('--show/--no-show', default=False, help="Show the dataframe.")
def main(train, clean, viso, pred, show):
    if clean:
        print("Start Cleaning")
        cleaning()
    if viso:
        print("Start Visualizing")
        visualize()
    if train:
        print("Start Training")
        features()
        training()
    if pred:
        print("Start Predicting")
        predict()
    if show:
        print("Start Show")
        print("Read in trips file...")
        df = io.read_file(pFilename="Trips.csv", pIo_folder="output")
        utils.cast_datetime(df, ["Start Time", "End Time"])
        df.drop("Unnamed: 0", axis=1, inplace=True)
        print(df.info())
        print(df.head())


def cleaning():
    df = io.read_file(pFilename="nuremberg.csv", pIo_folder="input")
    utils.cast_datetime(df, ["datetime"])
    df_trips = datapreparation.data_preparation(df)
    df_trips_add_feat = datapreparation.additional_feature_creation(df_trips)
    df_trips_filter_duration = datapreparation.drop_noise(df_trips_add_feat)
    df_trips_onlynuremberg = datapreparation.only_nuremberg_plz(df_trips_filter_duration)
    datapreparation.calculate_aggregate_statistics(df_trips_onlynuremberg)
    print("Save trip dataframe...")
    io.save_trip(df_trips_onlynuremberg, "Trips.csv")


def visualize():
    df = io.read_file(pFilename="Trips.csv", pIo_folder="output")
    utils.cast_datetime(df, ["Start Time", "End Time"])
    visualization.folium_maps.visualize_stations_moment(df)
    visualization.folium_maps.visualize_heatmap(df)
    visualization.folium_maps.visualize_plz(df)
    #visualization.visualize_plz(df)
    print("Visualizing Complete")


def features():
    df_trips = io.read_file(pFilename="Trips.csv", pIo_folder="output")
    # For calculations, cast datetime into a number (total seconds)
    utils.cast_datetime(df_trips, ["Start Time", "End Time"], as_numeric_value=True)
    df_trips.drop(["Unnamed: 0", "Place_start", "Place_end"], axis=1, inplace=True)
    print("Create dummie variables...")
    df = prediction.create_dummies(df_trips)
    print("Analyse correlations...")
    prediction.corr_analysis(df)
    io.save_trip(df, "Features.csv")


def training():
    df_features = io.read_file(pFilename="Features.csv", pIo_folder="output")
    print("Split data in train and test set...")
    X_train, X_test, y_train, y_test = prediction.simple_split(df_features)
    print("Scale data...")
    X_scaled_train = prediction.scale(X_train)
    print("Do PCA...")
    #X_train_transformed = prediction.do_pca(X_scaled_train)
    # ____________________________________________________________________________________
    print("Train linear regression...")
    prediction.train_linear_regression(X_scaled_train, y_train)
    print("Train SVM regression...")
    prediction.train_svm(X_scaled_train, y_train)
    print("Train NN...")
    prediction.train_neural_network(X_scaled_train, y_train)


def predict():
    df_features = io.read_file(pFilename="Features.csv", pIo_folder="output")
    print("Split data in train and test set...")
    X_train, X_test, y_train, y_test = prediction.simple_split(df_features)
    print("Predict with linear regression")
    prediction.predict_by_regression(X_test, y_test)
    print("Predict with SVM regression")
    prediction.predict_by_svm(X_test, y_test)
    print("Predict with NN")
    prediction.predict_by_nn(X_test, y_test)


if __name__ == '__main__':
    main()

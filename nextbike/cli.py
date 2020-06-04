import click
from datetime import datetime
from . import io
from . import datapreparation
from . import visualization
from . import prediction
from . import utils


@click.command()
@click.option('--clean/--no-clean', default=True, help="Clean the data.")
@click.option('--viso/--no-viso', default=True, help="Visualize the data.")
@click.option('--train/--no-train', default=True, help="Train the model.")
@click.option('--pred/--no-pred', default=True, help="Predict with model.")
def main(clean, viso, train, pred):
    start_time = datetime.now().replace(microsecond=0)
    start_time_step = start_time
    if clean:
        print("START CLEAN")
        cleaning()
        start_time_step = print_time_for_step(start_time_step)
    if viso:
        print("START VISUALIZE")
        # TODO Rename visualization.math / geo to math_plot and geo_plot
        visualize()
        start_time_step = print_time_for_step(start_time_step)
    if train:
        print("START TRAIN")
        features()
        training()
        start_time_step = print_time_for_step(start_time_step)
    if pred:
        print("START PREDICT")
        predict()
        start_time_step = print_time_for_step(start_time_step)
        print("START GEO PREDICT")
        predict_geo()
        start_time_step = print_time_for_step(start_time_step)

    print("TIME FOR RUN:", (datetime.now().replace(microsecond=0) - start_time))


# TODO: Add docstring
def cleaning():
    """Clean the data for further analysis


    Args:
        no Arg
    Returns:
        no Return
    """
    df = io.input.read_csv(p_filename="nuremberg.csv", p_io_folder="input")
    utils.cast_datetime(df, ["datetime"])
    print("Clean Data...")
    df_trips = datapreparation.data_clean.data_cleaning(df)
    print("Add Features...")
    df_trips_add_feat = datapreparation.feature_add.additional_feature_creation(df_trips)
    print("Clean Noise...")
    df_trips_filter_duration = datapreparation.data_clean.drop_noise(df_trips_add_feat)
    print("Clean Postalcodes...")
    df_trips_only_nuremberg = datapreparation.geo_clean.only_nuremberg(df_trips_filter_duration)
    print(df_trips_only_nuremberg.head())
    # print("Add Distances to University...")
    # df_trips_only_nuremberg_dist = datapreparation.feature_add.quick_create_dist(df_trips_only_nuremberg)
    # print("Save trip dataframe...")
    # io.output.save_csv(df_trips_only_nuremberg_dist, "Trips.csv")


# TODO: Add docstring
def visualize():
    df = io.read_csv(p_filename="Trips.csv", p_io_folder="output")
    utils.cast_datetime(df, ["Start Time", "End Time"])
    print("Visualize Aggregate Statistics...")
    visualization.math.calculate_aggregate_statistics(df)
    print("Visualize Stations Map...")
    visualization.geo.visualize_stations_moment(df)
    print("Visualize Heatmap Christmas...")
    visualization.geo.visualize_heatmap(df)
    print("Visualize Postalcode Zones...")
    visualization.geo.visualize_plz(df)
    print("Visualize Distribution Function...")
    visualization.math.plot_distribution(df)
    print("Visualize Mean Duration...")
    visualization.math.plot_mean_duration(df)


# TODO: Add docstring
def features():
    df_trips = io.input.read_csv(p_filename="Trips.csv", p_io_folder="output")
    df_trips.drop(["Place_start", "Start Time"], axis=1, inplace=True)
    print("Drop End Information")
    df_only_start = prediction.math_prepare_feature.drop_end_information(df_trips)
    print("Create Dummie Variables...")
    df_features = prediction.math_prepare_feature.create_dummies(df_only_start)
    print("Do Feature Engineering...")
    df_features_2 = prediction.math_prepare_feature.create_new_features(df_features)
    print("Visualize correlations...")
    df_features_2 = prediction.math_prepare_feature.drop_features(df_features_2)
    visualization.math.corr_analysis(df_features_2)
    io.output.save_csv(df_features_2, "Features.csv")
    # visualization.math.plot_features_influence(df_features_2)


# TODO: Add docstring
def training():
    df_features = io.input.read_csv(p_filename="Features.csv", p_io_folder="output")
    print("Split Data...")
    X_train, X_test, y_train, y_test = prediction.math_split.simple_split(df_features)
    print("Scale Data...")
    X_scaled_train = prediction.math_prepare_feature.scale(X_train)
    print("Do PCA...")
    X_train_transformed = prediction.math_prepare_feature.do_pca(X_scaled_train)
    # ____________________________________________________________________________________
    print("Train Linear Regression...")
    prediction.math_train.train_linear_regression(X_train_transformed, y_train)
    print("Train SVM Regression...")
    prediction.math_train.train_svm(X_train_transformed, y_train)
    print("Train NN...")
    prediction.math_train.train_neural_network(X_train_transformed, y_train)


# TODO: Add docstring
def predict():
    df_features = io.input.read_csv(p_filename="Features.csv", p_io_folder="output")
    print("Split Data...")
    X_train, X_test, y_train, y_test = prediction.math_split.simple_split(df_features)
    print("Predict by Linear Regression...")
    prediction.math_predict.predict_by_regression(X_test, y_test)
    print("Predict by SVM Regression...")
    prediction.math_predict.predict_by_svm(X_test, y_test)
    print("Predict by NN...")
    prediction.math_predict.predict_by_nn(X_test, y_test)


# TODO: Add docstring
def predict_geo():
    df_features = io.input.read_csv(p_filename="Trips.csv", p_io_folder="output")
    print("Predict Trip Direction...")
    prediction.geo_predict.train_pred(df_features)


def print_time_for_step(p_start_time_step):
    """Calculates time needed for current step and prints it out.
    Returns start time for next step

    :param p_start_time_step: start time of current step
    :return: start time of next step
    """
    print("TIME FOR STEP:", (datetime.now().replace(microsecond=0) - p_start_time_step))
    return datetime.now().replace(microsecond=0)


if __name__ == '__main__':
    main()

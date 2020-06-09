import pandas as pd
from nextbike import utils, io, datapreparation, prediction, testing, visualization


def cleaning(p_filename="nuremberg.csv", p_mode=""):
    """Clean the data for further analysis.

    Method which runs the sequential flow of the data cleaning part.

    Args:
        p_filename:     filename of csv input file to clean (Default: nuremberg.csv)
        p_mode:         defines whether called by test set or not
    Returns:
        No return
    """
    df = io.input.read_csv(p_filename=p_filename, p_io_folder="input")
    utils.cast_datetime(p_df=df, p_datetime_columns=["datetime"])
    print("Clean Data...")
    df_trips = datapreparation.data_clean.data_cleaning(p_df_original=df)
    print("Add Features...")
    df_trips_add_feat = datapreparation.feature_add.additional_feature_creation(p_df_trips=df_trips)
    print("Clean Noise...")
    df_trips_filter_duration = datapreparation.data_clean.drop_noise(p_df_trips=df_trips_add_feat)
    print("Clean Postalcodes...")
    df_trips_only_nuremberg = datapreparation.geo_clean.only_nuremberg(p_df=df_trips_filter_duration)
    print("Add Distances to University...")
    df_trips_only_nuremberg_dist = datapreparation.feature_add.quick_create_dist(p_df=df_trips_only_nuremberg)
    print("Save trip dataframe...")
    io.output.save_csv(p_df=df_trips_only_nuremberg_dist, p_filename="Trips" + p_mode + ".csv")


def visualize(p_mode=""):
    """Visualize the data.

    Method which runs the sequential flow of the data visualization part.

    Args:
        No args
    Returns:
        No return
    """
    df = io.input.read_csv(p_filename="Trips" + p_mode + ".csv", p_io_folder="output")
    utils.cast_datetime(p_df=df, p_datetime_columns=["Start_Time", "End_Time"])
    print("Visualize Aggregate Statistics...")
    visualization.math_descriptive.calculate_aggregate_statistics(p_df_trips=df, p_mode=p_mode)
    print("Visualize Duration and count on different subsets...")
    visualization.math_descriptive.plot_all_subet_lines_graphs(p_df_trips=df, p_mode=p_mode)
    print("Visualize Stations Map...")
    visualization.geo.visualize_stations_moment(p_df=df, p_mode=p_mode)
    print("Visualize Heatmap Christmas...")
    visualization.geo.visualize_heatmap(p_df=df, p_mode=p_mode)
    print("Visualize Postalcode Zones...")
    visualization.geo.visualize_postalcode(p_df=df, p_mode=p_mode)
    print("Visualize Trips in Map...")
    visualization.geo.visualize_trips_per_month(p_df=df, p_mode=p_mode)
    print("Visualize Monthly Distribution...")
    visualization.math_descriptive.plot_distribution_monthly(p_df=df, p_mode=p_mode)
    print("Visualize Distribution Function...")
    visualization.math_descriptive.plot_distribution(p_df=df, p_mode=p_mode)
    print("Visualize Mean Duration...")
    visualization.math_descriptive.plot_mean_duration(p_df=df, p_mode=p_mode)


def features_duration(p_weather, p_mode=""):
    """Create and prepare the features before prediction part.

    Method which runs the sequential flow of the feature preparation and creation part.

    Args:
        p_weather       option whether weather data should be included
    Returns:
        No return
    """
    df_trips = io.input.read_csv(p_filename="Trips" + p_mode + ".csv", p_io_folder="output")

    # TODO: Add corr analysis before feature selection be aware of non numerical features
    # visualization.math_descriptive.corr_analysis(df_features_2)
    print("Drop End Information")
    df_only_start = prediction.prepare_feature.drop_end_information(p_df=df_trips)
    print("Create Dummie Variables...")
    df_features = prediction.prepare_feature.create_dummies(p_df=df_only_start)
    print("Do Feature Engineering...")
    df_features_2 = prediction.prepare_feature.create_new_features(p_X=df_features, p_weather=p_weather)
    print("Visualize correlations...")
    df_features_2 = prediction.prepare_feature.drop_features(p_df=df_features_2)
    df_features_2 = df_features_2.drop(["Place_start", "Start_Time"], axis=1)
    visualization.math_descriptive.corr_analysis(p_df=df_features_2, p_weather=p_weather)
    io.output.save_csv(p_df=df_features_2, p_filename="Features_Duration" + p_mode + p_weather + ".csv")
    # visualization.math.plot_features_influence(df_features_2)


def training_duration_models(p_weather, p_mode=""):
    """Train the different machine learning models.

    Method which runs the sequential flow on training the ML models.

    Args:
        p_weather (str): Fileending for generated files (include weather or not)
        p_mode (str):     defines whether called by test set or not
    Returns:
        No return
    """
    # Prepare
    df_features = io.input.read_csv(p_filename="Features_Duration" + p_mode + p_weather + ".csv", p_io_folder="output")
    print("Split Data...")
    X_train, X_test, y_train, y_test = prediction.split.simple_split_duration(p_df=df_features)
    print("Scale Data...")
    X_scaled_train = prediction.prepare_feature.scale(p_X_train=X_train,
                                                      p_scaler_name="Standard_Scaler_Duration" + p_weather)
    print("Do PCA...")
    components = 17
    # if len(p_weather) > 0:
    #     components = 21

    X_train_transformed = prediction.prepare_feature.do_pca(p_X_scaled_train=X_scaled_train,
                                                            p_number_components=components,
                                                            p_filename="PCA_Duration" + p_weather)
    df_components = pd.DataFrame(X_train_transformed)
    io.output.save_csv(p_df=y_train, p_filename="y_train_Duration" + p_weather + ".csv")
    io.output.save_csv(p_df=df_components, p_filename="Components_Duration" + p_weather + ".csv")
    # Train
    # The sets are in order: y_train, y_val, y_prediction_train, y_prediction_val
    print("Train Dummy Mean Regression...")
    d_mean_sets = prediction.math_train.train_dummy_regression_mean(p_X_train_scaled=X_train_transformed,
                                                                    p_y_train=y_train, p_weather=p_weather)
    print("Train Dummy Median Regression...")
    dummy_med_sets = prediction.math_train.train_dummy_regression_median(p_X_train_scaled=X_train_transformed,
                                                                         p_y_train=y_train, p_weather=p_weather)
    print("Train Linear Regression...")
    lin_regr_sets = prediction.math_train.train_linear_regression(p_X_train_scaled=X_train_transformed,
                                                                  p_y_train=y_train, p_weather=p_weather)
    print("Train SVM Regression...")
    svm_regr_sets = prediction.math_train.train_svm(p_X_train_scaled=X_train_transformed,
                                                    p_y_train=y_train, p_weather=p_weather)
    print("Train NN...")
    nn_regr_sets = prediction.math_train.train_neural_network(p_X_train_scaled=X_train_transformed,
                                                              p_y_train=y_train, p_weather=p_weather)
    # Evaluate Training
    # Dummy Regression Mean
    prediction.evaluate.duration_error_metrics(p_y_true=d_mean_sets[0],
                                               p_y_predictions=d_mean_sets[2],
                                               p_filename="Dummy_Mean_Regression_Training" + p_weather)
    prediction.evaluate.duration_error_metrics(p_y_true=d_mean_sets[1],
                                               p_y_predictions=d_mean_sets[3],
                                               p_filename="Dummy_Mean_Regression_Validation" + p_weather,
                                               p_status="Validation")
    # Dummy Regression Median
    prediction.evaluate.duration_error_metrics(p_y_true=dummy_med_sets[0],
                                               p_y_predictions=dummy_med_sets[2],
                                               p_filename="Dummy_Med_Regression_Training" + p_weather)
    prediction.evaluate.duration_error_metrics(p_y_true=dummy_med_sets[1],
                                               p_y_predictions=dummy_med_sets[3],
                                               p_filename="Dummy_Med_Regression_Validation" + p_weather,
                                               p_status="Validation")
    # Linear Regression
    prediction.evaluate.duration_error_metrics(p_y_true=lin_regr_sets[0],
                                               p_y_predictions=lin_regr_sets[2],
                                               p_filename="Linear_Regression_Training" + p_weather)
    prediction.evaluate.duration_error_metrics(p_y_true=lin_regr_sets[1],
                                               p_y_predictions=lin_regr_sets[3],
                                               p_filename="Linear_Regression_Validation" + p_weather,
                                               p_status="Validation")
    # SVM Regression
    prediction.evaluate.duration_error_metrics(p_y_true=svm_regr_sets[0],
                                               p_y_predictions=svm_regr_sets[2],
                                               p_filename="SVM_Regression_Training" + p_weather)
    prediction.evaluate.duration_error_metrics(p_y_true=svm_regr_sets[1],
                                               p_y_predictions=svm_regr_sets[3],
                                               p_filename="SVM_Regression_Validation" + p_weather,
                                               p_status="Validation")
    # NN Regression
    prediction.evaluate.duration_error_metrics(p_y_true=nn_regr_sets[0],
                                               p_y_predictions=nn_regr_sets[2],
                                               p_filename="NN_Regression_Training" + p_weather)
    prediction.evaluate.duration_error_metrics(p_y_true=nn_regr_sets[1],
                                               p_y_predictions=nn_regr_sets[3],
                                               p_filename="NN_Regression_Validation" + p_weather,
                                               p_status="Validation")


def testing_robust_scaler():
    """Tests robust scaler on duration features.

    Returns:
        No return
    """
    df_features = io.input.read_csv(p_filename="Features_Duration.csv", p_io_folder="output")
    testing.robust_scaler_testing.test_robust_scaler(p_df=df_features)


def testing_duration_models():
    """Tests duration models on duration component and true y values.

    Returns:
        No return
    """
    df_components = io.input.read_csv(p_filename="Components_Duration.csv", p_io_folder="output").reset_index(drop=True)
    y_true = io.input.read_csv(p_filename="y_train_Duration.csv", p_io_folder="output")
    # testing.nn_testing.test_neuralnetwork_model(p_components=df_components, p_y_train=y_true)
    testing.linear_regression_testing.test_regression_model(p_components=df_components, p_y_train=y_true)


def predict_duration_models(p_weather):
    """Predict the duration of trips by different models.

    Method which runs the sequential flow of the duration prediction by different trained ML models.

    Args:
        p_weather       option whether weather data should be included
    Returns:
        No return
    """
    # Prepare
    df_features = io.input.read_csv(p_filename="Features_Duration" + p_weather + ".csv", p_io_folder="output")
    print("Split Data...")
    X_train, X_test, y_train, y_test = prediction.split.simple_split_duration(p_df=df_features)
    # Predict
    print("Predict by Dummy Mean Regression...")
    dummy_mean_reg_y_prediction = prediction.math_predict.predict_by_dummy_mean(p_X_test=X_test, p_weather=p_weather)
    print("Predict by Median Regression...")
    dummy_med_reg_y_prediction = prediction.math_predict.predict_by_dummy_median(p_X_test=X_test, p_weather=p_weather)
    print("Predict by Linear Regression...")
    lin_regr_y_prediction = prediction.math_predict.predict_by_regression(p_X_test=X_test, p_weather=p_weather)
    print("Predict by SVM Regression...")
    svm_y_prediction = prediction.math_predict.predict_by_svm(p_X_test=X_test, p_weather=p_weather)
    print("Predict by NN...")
    nn_y_prediction = prediction.math_predict.predict_by_nn(p_X_test=X_test, p_weather=p_weather)
    # Evaluate Prediction
    prediction.evaluate.duration_error_metrics(p_y_true=y_test,
                                               p_y_predictions=dummy_mean_reg_y_prediction,
                                               p_filename="Dummy_Mean_Regression" + p_weather,
                                               p_status="Testing")
    prediction.evaluate.duration_error_metrics(p_y_true=y_test,
                                               p_y_predictions=dummy_med_reg_y_prediction,
                                               p_filename="Dummy_Median_Regression" + p_weather,
                                               p_status="Testing")
    prediction.evaluate.duration_error_metrics(p_y_true=y_test,
                                               p_y_predictions=lin_regr_y_prediction,
                                               p_filename="Linear_Regression" + p_weather,
                                               p_status="Testing")
    prediction.evaluate.duration_error_metrics(p_y_true=y_test,
                                               p_y_predictions=svm_y_prediction,
                                               p_filename="SVM_Regression" + p_weather,
                                               p_status="Testing")
    prediction.evaluate.duration_error_metrics(p_y_true=y_test,
                                               p_y_predictions=nn_y_prediction,
                                               p_filename="NN_Regression" + p_weather,
                                               p_status="Testing")


def features_direction(p_weather, p_mode=""):

    """Prepares features for direction prediction out of trip data.
    Drops end information and unneeded features, creates new features
    and saves them as csv file.

    Args:
        p_weather:  file ending when weather data is included
        p_mode:     defines whether called by test set or not
    Returns:
        No return
    """
    df_features = io.input.read_csv(p_filename="Trips" + p_mode + ".csv", p_io_folder="output")

    print("Drop End Information")
    df_features = prediction.prepare_feature.drop_end_information(p_df=df_features, direction_needed=True)
    print("Create Dummie Variables...")
    df_features = prediction.prepare_feature.create_dummies(p_df=df_features)
    print("Do Feature Engineering...")
    df_features = prediction.prepare_feature.create_new_features(p_X=df_features, p_weather=p_weather)
    print("Drop Unneeded Features...")
    df_features = prediction.prepare_feature.drop_features(p_df=df_features)
    df_features = df_features.drop(["Place_start", "Start_Time"], axis=1)
    io.output.save_csv(p_df=df_features, p_filename="Features_Direction" + p_mode + p_weather + ".csv")


def train_direction_models(p_weather, p_mode=""):
    """Train the direction models for trip classification (towards or away from university).

    Method which runs the sequential flow of the direction classification training.

    Args:
        p_weather:  file ending when weather data is included
        p_mode:     defines whether called by test set or not
    Returns:
        No return
    """
    # TODO: Feature selection etc...
    # Prepare
    df_features = io.input.read_csv(p_filename="Features_Direction" + p_mode + p_weather + ".csv", p_io_folder="output")
    print("Split Data...")
    X_train, X_test, y_train, y_test = prediction.split.simple_split_direction(p_df=df_features)
    print("Scale Data...")
    X_scaled_train = prediction.prepare_feature.scale(p_X_train=X_train,
                                                      p_scaler_name="Standard_Scaler_Direction" + p_weather)
    print("Do PCA...")
    # TODO: fit number of components
    X_train_transformed = prediction.prepare_feature.do_pca(p_X_scaled_train=X_scaled_train,
                                                            p_number_components=17,
                                                            p_filename="PCA_Direction" + p_weather)
    # Train
    print("Train Dummy Classifier...")
    dummy_sets = prediction.geo_train.train_classification_dummy(p_X_train_scaled=X_train_transformed,
                                                                 p_y_train=y_train, p_weather=p_weather)
    print("Train KNeighbors Classifier...")
    kn_sets = prediction.geo_train.train_classification_k_neighbors(p_X_train_scaled=X_train_transformed,
                                                                    p_y_train=y_train, p_weather=p_weather)
    print("Train Decision Tree Classifier...")
    dt_sets = prediction.geo_train.train_classification_decision_tree(p_X_train_scaled=X_train_transformed,
                                                                      p_y_train=y_train, p_weather=p_weather)
    print("Train Random Forest Classifier...")
    rf_sets = prediction.geo_train.train_classification_random_forest(p_X_train_scaled=X_train_transformed,
                                                                      p_y_train=y_train, p_weather=p_weather)
    print("Train NN Classifier...")
    nn_sets = prediction.geo_train.train_classification_neural_network(p_X_train_scaled=X_train_transformed,
                                                                       p_y_train=y_train, p_weather=p_weather)
    # Evaluate Training
    # Dummy_Classifier
    prediction.evaluate.direction_error_metrics(p_y_true=dummy_sets[0],
                                                p_y_predictions=dummy_sets[2],
                                                p_filename="Dummy_Classifier" + p_weather)
    prediction.evaluate.direction_error_metrics(p_y_true=dummy_sets[1],
                                                p_y_predictions=dummy_sets[3],
                                                p_filename="Dummy_Classifier" + p_weather, p_status="Validation")
    # KNeighbors_Classifier
    prediction.evaluate.direction_error_metrics(p_y_true=kn_sets[0],
                                                p_y_predictions=kn_sets[2],
                                                p_filename="KNeighbors_Classifier" + p_weather)
    prediction.evaluate.direction_error_metrics(p_y_true=kn_sets[1],
                                                p_y_predictions=kn_sets[3],
                                                p_filename="KNeighbors_Classifier" + p_weather, p_status="Validation")
    # Decision_Tree_Classifier
    prediction.evaluate.direction_error_metrics(p_y_true=dt_sets[0],
                                                p_y_predictions=dt_sets[2],
                                                p_filename="Decision_Tree_Classifier" + p_weather)
    prediction.evaluate.direction_error_metrics(p_y_true=dt_sets[1],
                                                p_y_predictions=dt_sets[3],
                                                p_filename="Decision_Tree_Classifier" + p_weather,
                                                p_status="Validation")
    # Random_Forest_Classifier
    prediction.evaluate.direction_error_metrics(p_y_true=rf_sets[0],
                                                p_y_predictions=rf_sets[2],
                                                p_filename="Random_Forest_Classifier" + p_weather)
    prediction.evaluate.direction_error_metrics(p_y_true=rf_sets[1],
                                                p_y_predictions=rf_sets[3],
                                                p_filename="Random_Forest_Classifier" + p_weather,
                                                p_status="Validation")
    # Neural Network Classifier
    prediction.evaluate.direction_error_metrics(p_y_true=nn_sets[0],
                                                p_y_predictions=nn_sets[2],
                                                p_filename="NN_Classifier" + p_weather)
    prediction.evaluate.direction_error_metrics(p_y_true=nn_sets[1],
                                                p_y_predictions=nn_sets[3],
                                                p_filename="NN_Classifier" + p_weather, p_status="Validation")


def testing_direction_subsets():
    """Tests direction classification on different subsets (like month).

    Returns:
        No return
    """
    # data
    df_features = io.input.read_csv(p_filename="Features_Direction.csv", p_io_folder="output")
    testing.direction_classification_subsets_testing.filter_subsets(p_df=df_features)


def predict_direction_models(p_weather):
    """Starts prediction of direction based on different models and
    prints out the performances and error metrcis of the different models.

    Args:
        p_weather:  file ending when weather data is included
    Returns:
        No return
    """
    # Prepare
    df_features = io.input.read_csv(p_filename="Features_Direction" + p_weather + ".csv", p_io_folder="output")
    print("Split Data...")
    X_train, X_test, y_train, y_test = prediction.split.simple_split_direction(p_df=df_features)
    # Predict
    print("Predict by Dummy Classification...")
    dummy_y_prediction = prediction.geo_predict.predict_by_dummy_classificaion(p_X_test=X_test, p_weather=p_weather)
    print("Predict by KNeighbors Classificaion...")
    kn_y_prediction = prediction.geo_predict.predict_by_k_neighbors_classificaion(p_X_test=X_test, p_weather=p_weather)
    print("predict_by_decision_tree_classificaion...")
    dt_y_prediction = prediction.geo_predict.predict_by_decision_tree_classificaion(p_X_test=X_test,
                                                                                    p_weather=p_weather)
    print("predict_by_random_forest_classificaion")
    rf_y_prediction = prediction.geo_predict.predict_by_random_forest_classificaion(p_X_test=X_test,
                                                                                    p_weather=p_weather)
    print("predict_by_neural_network_classificaion")
    nn_y_prediction = prediction.geo_predict.predict_by_neural_network_classificaion(p_X_test=X_test,
                                                                                     p_weather=p_weather)
    # Evaluate Prediction
    prediction.evaluate.direction_error_metrics(p_y_true=y_test,
                                                p_y_predictions=dummy_y_prediction,
                                                p_filename="Dummy_Classifier" + p_weather,
                                                p_status="Testing")
    prediction.evaluate.direction_error_metrics(p_y_true=y_test,
                                                p_y_predictions=kn_y_prediction,
                                                p_filename="KNeighbors_Classifier" + p_weather,
                                                p_status="Testing")
    prediction.evaluate.direction_error_metrics(p_y_true=y_test,
                                                p_y_predictions=dt_y_prediction,
                                                p_filename="Decision_Tree_Classifier" + p_weather,
                                                p_status="Testing")
    prediction.evaluate.direction_error_metrics(p_y_true=y_test,
                                                p_y_predictions=rf_y_prediction,
                                                p_filename="Random_Forest_Classifier" + p_weather,
                                                p_status="Testing")
    prediction.evaluate.direction_error_metrics(p_y_true=y_test,
                                                p_y_predictions=nn_y_prediction,
                                                p_filename="NN_Classifier" + p_weather,
                                                p_status="Testing")


def train_best_regression_model(p_weather, p_mode=""):
    """Train the best model for regression on duration (Neural Network).

    Method which runs the sequential flow on training the Neural Network.

    Args:
        p_weather (str): Fileending for generated files (include weather or not)
        p_mode (str):    Defines whether called by test set or not
    Returns:
        No return
    """
    # Prepare
    df_features = io.input.read_csv(p_filename="Features_Duration" + p_mode + p_weather + ".csv", p_io_folder="output")
    print("Split Data...")
    X_train, X_test, y_train, y_test = prediction.split.simple_split_duration(p_df=df_features)
    print("Scale Data...")
    X_scaled_train = prediction.prepare_feature.scale(p_X_train=X_train,
                                                      p_scaler_name="Standard_Scaler_Duration" + p_weather)
    print("Do PCA...")
    X_train_transformed = prediction.prepare_feature.do_pca(p_X_scaled_train=X_scaled_train,
                                                            p_number_components=17,
                                                            p_filename="PCA_Duration" + p_weather)
    df_components = pd.DataFrame(X_train_transformed)
    io.output.save_csv(p_df=y_train, p_filename="y_train_Duration" + p_weather + ".csv")
    io.output.save_csv(p_df=df_components, p_filename="Components_Duration" + p_weather + ".csv")
    # Train
    # The sets are in order: y_train, y_val, y_prediction_train, y_prediction_val
    print("Train NN...")
    nn_regr_sets = prediction.math_train.train_neural_network(p_X_train_scaled=X_train_transformed,
                                                              p_y_train=y_train, p_weather=p_weather)
    # Evaluate Training
    # NN Regression
    prediction.evaluate.duration_error_metrics(p_y_true=nn_regr_sets[0],
                                               p_y_predictions=nn_regr_sets[2],
                                               p_filename="NN_Regression_Training" + p_weather)
    prediction.evaluate.duration_error_metrics(p_y_true=nn_regr_sets[1],
                                               p_y_predictions=nn_regr_sets[3],
                                               p_filename="NN_Regression_Validation" + p_weather,
                                               p_status="Validation")


def train_best_classification_model(p_weather, p_mode=""):
    """Train the best direction classification (towards or away from university) model (Neural Network).

    Method which runs the sequential flow of the direction classification training.

    Args:
        p_weather:  file ending when weather data is included
        p_mode:     defines whether called by test set or not (optional)
    Returns:
        No return
    """
    # Prepare
    df_features = io.input.read_csv(p_filename="Features_Direction" + p_mode + p_weather + ".csv", p_io_folder="output")
    print("Split Data...")
    X_train, X_test, y_train, y_test = prediction.split.simple_split_direction(p_df=df_features)
    print("Scale Data...")
    X_scaled_train = prediction.prepare_feature.scale(p_X_train=X_train,
                                                      p_scaler_name="Standard_Scaler_Direction" + p_weather)
    print("Do PCA...")
    X_train_transformed = prediction.prepare_feature.do_pca(p_X_scaled_train=X_scaled_train,
                                                            p_number_components=17,
                                                            p_filename="PCA_Direction" + p_weather)
    # Train
    print("Train KNeighbors Classifier...")
    kn_sets = prediction.geo_train.train_classification_k_neighbors(p_X_train_scaled=X_train_transformed,
                                                                    p_y_train=y_train, p_weather=p_weather)
    print("Train NN Classifier...")
    nn_sets = prediction.geo_train.train_classification_neural_network(p_X_train_scaled=X_train_transformed,
                                                                       p_y_train=y_train, p_weather=p_weather)
    # Evaluate Training
    # KNeighbors_Classifier
    prediction.evaluate.direction_error_metrics(p_y_true=kn_sets[0],
                                                p_y_predictions=kn_sets[2],
                                                p_filename="KNeighbors_Classifier" + p_weather)
    prediction.evaluate.direction_error_metrics(p_y_true=kn_sets[1],
                                                p_y_predictions=kn_sets[3],
                                                p_filename="KNeighbors_Classifier" + p_weather, p_status="Validation")

    # Neural Network Classifier
    prediction.evaluate.direction_error_metrics(p_y_true=nn_sets[0],
                                                p_y_predictions=nn_sets[2],
                                                p_filename="NN_Classifier" + p_weather)
    prediction.evaluate.direction_error_metrics(p_y_true=nn_sets[1],
                                                p_y_predictions=nn_sets[3],
                                                p_filename="NN_Classifier" + p_weather, p_status="Validation")


def pred_by_best_reression_model(p_weather):
    """regress on the duration of trips by the best model (Neural Network).

    Method which runs the sequential flow of the duration regression by the neural network.

    Args:
        p_weather (str):    option whether weather data should be included
    Returns:
        No return
    """
    # Prepare
    df_features = io.input.read_csv(p_filename="Features_Duration" + p_weather + ".csv", p_io_folder="output")
    print("Split Data...")
    X_train, X_test, y_train, y_test = prediction.split.simple_split_duration(p_df=df_features)
    # Predict
    print("Predict by NN...")
    nn_y_prediction = prediction.math_predict.predict_by_nn(p_X_test=X_test, p_weather=p_weather)
    # Evaluate Prediction
    prediction.evaluate.duration_error_metrics(p_y_true=y_test,
                                               p_y_predictions=nn_y_prediction,
                                               p_filename="NN_Regression" + p_weather,
                                               p_status="Testing")


def pred_by_best_classification_model(p_weather):
    """classify the direction of trips by the best model (Neural Network).

    Method which runs the sequential flow of the direction classification by the neural network.

    Args:
        p_weather (str):    option whether weather data should be included
    Returns:
        No return
    """
    # Prepare
    df_features = io.input.read_csv(p_filename="Features_Direction" + p_weather + ".csv", p_io_folder="output")
    print("Split Data...")
    X_train, X_test, y_train, y_test = prediction.split.simple_split_direction(p_df=df_features)
    # Predict
    print("predict_by_neural_network_classificaion")
    nn_y_prediction = prediction.geo_predict.predict_by_neural_network_classificaion(p_X_test=X_test,
                                                                                     p_weather=p_weather)
    # Evaluate Prediction
    prediction.evaluate.direction_error_metrics(p_y_true=y_test,
                                                p_y_predictions=nn_y_prediction,
                                                p_filename="NN_Classifier" + p_weather,
                                                p_status="Testing")

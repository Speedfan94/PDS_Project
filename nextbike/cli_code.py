import pandas as pd
from nextbike import utils, io, datapreparation, prediction, testing, visualization


def cleaning(filename="nuremberg.csv"):
    """Clean the data for further analysis.

    Method which runs the sequential flow of the data cleaning part.

    Args:
        no Arg
    Returns:
        no Return
    """
    df = io.input.read_csv(p_filename=filename, p_io_folder="input")
    utils.cast_datetime(df, ["datetime"])
    print("Clean Data...")
    df_trips = datapreparation.data_clean.data_cleaning(df)
    print("Add Features...")
    df_trips_add_feat = datapreparation.feature_add.additional_feature_creation(df_trips)
    print("Clean Noise...")
    df_trips_filter_duration = datapreparation.data_clean.drop_noise(df_trips_add_feat)
    print("Clean Postalcodes...")
    df_trips_only_nuremberg = datapreparation.geo_clean.only_nuremberg(df_trips_filter_duration)
    print("Add Distances to University...")
    df_trips_only_nuremberg_dist = datapreparation.feature_add.quick_create_dist(df_trips_only_nuremberg)
    print("Save trip dataframe...")
    io.output.save_csv(df_trips_only_nuremberg_dist, "Trips.csv")


def visualize():
    """Visualize the data.

    Method which runs the sequential flow of the data visualization part.

    Args:
        no Arg
    Returns:
        no Return
    """
    df = io.read_csv(p_filename="Trips.csv", p_io_folder="output")
    utils.cast_datetime(df, ["Start_Time", "End_Time"])
    print("Visualize Aggregate Statistics...")
    visualization.math_descriptive.calculate_aggregate_statistics(df)
    print("Visualize Stations Map...")
    visualization.geo.visualize_stations_moment(df)
    print("Visualize Heatmap Christmas...")
    visualization.geo.visualize_heatmap(df)
    print("Visualize Postalcode Zones...")
    visualization.geo.visualize_postalcode(df)
    print("Visualize Monthly Distribution...")
    visualization.math_descriptive.plot_distribution_monthly(df)
    print("Visualize Distribution Function...")
    visualization.math_descriptive.plot_distribution(df)
    print("Visualize Mean Duration...")
    visualization.math_descriptive.plot_mean_duration(df)


def features_duration(trips_file="Trips.csv", weather=False):
    """Create and prepare the features before prediction part.

    Method which runs the sequential flow of the feature preparation and creation part.

    Args:
        no Arg
    Returns:
        no Return
    """
    df_trips = io.input.read_csv(p_filename=trips_file, p_io_folder="output")

    # TODO: Add corr analysis before feature selection be aware of non numerical features
    # visualization.math_descriptive.corr_analysis(df_features_2)
    print("Drop End Information")
    df_only_start = prediction.prepare_feature.drop_end_information(df_trips)
    print("Create Dummie Variables...")
    df_features = prediction.prepare_feature.create_dummies(df_only_start)
    print("Do Feature Engineering...")
    df_features_2 = prediction.prepare_feature.create_new_features(df_features, weather)
    print("Visualize correlations...")
    df_features_2 = prediction.prepare_feature.drop_features(df_features_2)
    df_features_2 = df_features_2.drop(["Place_start", "Start_Time"], axis=1)
    visualization.math_descriptive.corr_analysis(df_features_2)
    io.output.save_csv(df_features_2, "Features_Duration.csv")
    # visualization.math.plot_features_influence(df_features_2)


def training_duration_models():
    """Train the different machine learning models.

    Method which runs the sequential flow on training the ML models.

    Args:
        no Arg
    Returns:
        no Return
    """
    # Prepare
    df_features = io.input.read_csv(p_filename="Features_Duration.csv", p_io_folder="output")
    print("Split Data...")
    X_train, X_test, y_train, y_test = prediction.split.simple_split_duration(df_features)
    print("Scale Data...")
    X_scaled_train = prediction.prepare_feature.scale(X_train, "Standard_Scaler_Duration")
    print("Do PCA...")
    X_train_transformed = prediction.prepare_feature.do_pca(X_scaled_train, 21, "PCA_Duration")
    df_components = pd.DataFrame(X_train_transformed)
    io.output.save_csv(y_train, p_filename="y_train_Duration.csv")
    io.output.save_csv(df_components, p_filename="Components_Duration.csv")
    # Train
    # The sets are in order: y_train, y_val, y_prediction_train, y_prediction_val
    print("Train Dummy Mean Regression...")
    d_mean_sets = prediction.math_train.train_dummy_regression_mean(X_train_transformed, y_train)
    print("Train Dummy Median Regression...")
    dummy_med_sets = prediction.math_train.train_dummy_regression_median(X_train_transformed, y_train)
    print("Train Linear Regression...")
    lin_regr_sets = prediction.math_train.train_linear_regression(X_train_transformed, y_train)
    print("Train SVM Regression...")
    svm_regr_sets = prediction.math_train.train_svm(X_train_transformed, y_train)
    print("Train NN...")
    nn_regr_sets = prediction.math_train.train_neural_network(X_train_transformed, y_train)
    # Evaluate Training
    # Dummy Regression Mean
    prediction.evaluate.duration_error_metrics(d_mean_sets[0], d_mean_sets[2], "Dummy_Mean_Regression_Training")
    prediction.evaluate.duration_error_metrics(d_mean_sets[1], d_mean_sets[3], "Dummy_Mean_Regression_Validation", "Validation")
    # Dummy Regression Median
    prediction.evaluate.duration_error_metrics(dummy_med_sets[0], dummy_med_sets[2], "Dummy_Med_Regression_Training")
    prediction.evaluate.duration_error_metrics(dummy_med_sets[1], dummy_med_sets[3], "Dummy_Med_Regression_Validation", "Validation")
    # Linear Regression
    prediction.evaluate.duration_error_metrics(lin_regr_sets[0], lin_regr_sets[2], "Linear_Regression_Training")
    prediction.evaluate.duration_error_metrics(lin_regr_sets[1], lin_regr_sets[3], "Linear_Regression_Validation", "Validation")
    # SVM Regression
    prediction.evaluate.duration_error_metrics(svm_regr_sets[0], svm_regr_sets[2], "SVM_Regression_Training")
    prediction.evaluate.duration_error_metrics(svm_regr_sets[1], svm_regr_sets[3], "SVM_Regression_Validation", "Validation")
    # NN Regression
    prediction.evaluate.duration_error_metrics(nn_regr_sets[0], nn_regr_sets[2], "NN_Regression_Training")
    prediction.evaluate.duration_error_metrics(nn_regr_sets[1], nn_regr_sets[3], "NN_Regression_Validation", "Validation")


def testing_robust_scaler():
    # TODO: Docstring
    df_features = io.input.read_csv("Features_Duration.csv", p_io_folder="output")
    testing.robust_scaler_testing.test_robust_scaler(df_features)


def testing_duration_models():
    # TODO: add docstring
    df_components = io.input.read_csv("Components_Duration.csv", p_io_folder="output").reset_index(drop=True)
    y_true = io.input.read_csv("y_train_Duration.csv", p_io_folder="output")
    # testing.nn_testing.test_neuralnetwork_model(df_components, y_true)
    testing.linear_regression_testing.test_regression_model(df_components, y_true)


def predict_duration_models(trips_file, weather=False):
    """Predict the duration of trips by different models.

    Method which runs the sequential flow of the duration prediction by different trained ML models.

    Args:
        no Arg
    Returns:
        no Return
    """
    # Prepare
    if trips_file:
        # first create features for duration prediction out of trips data
        features_duration(trips_file, weather)
    df_features = io.input.read_csv(p_filename="Features_Duration.csv", p_io_folder="output")
    print("Split Data...")
    X_train, X_test, y_train, y_test = prediction.split.simple_split_duration(df_features)
    # Predict
    print("Predict by Dummy Mean Regression...")
    dummy_mean_reg_y_prediction = prediction.math_predict.predict_by_dummy_mean(X_test)
    print("Predict by Median Regression...")
    dummy_med_reg_y_prediction = prediction.math_predict.predict_by_dummy_median(X_test)
    print("Predict by Linear Regression...")
    lin_regr_y_prediction = prediction.math_predict.predict_by_regression(X_test)
    print("Predict by SVM Regression...")
    svm_y_prediction = prediction.math_predict.predict_by_svm(X_test)
    print("Predict by NN...")
    nn_y_prediction = prediction.math_predict.predict_by_nn(X_test)
    # Evaluate Prediction
    prediction.evaluate.duration_error_metrics(y_test, dummy_mean_reg_y_prediction, "Dummy_Mean_Regression", "Testing")
    prediction.evaluate.duration_error_metrics(y_test, dummy_med_reg_y_prediction, "Dummy_Median_Regression", "Testing")
    prediction.evaluate.duration_error_metrics(y_test, lin_regr_y_prediction, "Linear_Regression", "Testing")
    prediction.evaluate.duration_error_metrics(y_test, svm_y_prediction, "SVM_Regression", "Testing")
    prediction.evaluate.duration_error_metrics(y_test, nn_y_prediction, "NN_Regression", "Testing")


def features_direction(trips_file="Trips.csv", weather=False):
    # TODO:Docstring
    df_features = io.input.read_csv(p_filename=trips_file, p_io_folder="output")
    print("Drop End Information")
    df_features = prediction.prepare_feature.drop_end_information(df_features, direction_needed=True)
    print("Create Dummie Variables...")
    df_features = prediction.prepare_feature.create_dummies(df_features)
    print("Do Feature Engineering...")
    df_features = prediction.prepare_feature.create_new_features(df_features, weather)
    print("Drop Unneeded Features...")
    df_features = prediction.prepare_feature.drop_features(df_features)
    df_features = df_features.drop(["Place_start", "Start_Time"], axis=1)
    io.output.save_csv(df_features, "Features_Direction.csv")


def train_direction_models():
    """Predict the direction of a trip (towards or away from university).

    Method which runs the sequential flow of the direction prediction.

    Args:
        no Arg
    Returns:
        no Return
    """
    # TODO: Feature selection etc...
    # Prepare
    df_features = io.input.read_csv(p_filename="Features_Direction.csv", p_io_folder="output")
    print("Split Data...")
    X_train, X_test, y_train, y_test = prediction.split.simple_split_direction(df_features)
    print("Scale Data...")
    X_scaled_train = prediction.prepare_feature.scale(X_train, "Standard_Scaler_Direction")
    print("Do PCA...")
    # TODO: fit number of components
    X_train_transformed = prediction.prepare_feature.do_pca(X_scaled_train, 15, "PCA_Direction")
    # Train
    print("Train Dummy Classifier...")
    dummy_sets = prediction.geo_train.train_classification_dummy(X_train_transformed, y_train)
    print("Train KNeighbors Classifier...")
    kn_sets = prediction.geo_train.train_classification_k_neighbors(X_train_transformed, y_train)
    print("Train Decision Tree Classifier...")
    dt_sets = prediction.geo_train.train_classification_decision_tree(X_train_transformed, y_train)
    print("Train Random Forest Classifier...")
    rf_sets = prediction.geo_train.train_classification_random_forest(X_train_transformed, y_train)
    print("Train NN Classifier...")
    nn_sets = prediction.geo_train.train_classification_neural_network(X_train_transformed, y_train)
    # Evaluate Training
    # Dummy_Classifier
    prediction.evaluate.direction_error_metrics(dummy_sets[0], dummy_sets[2], "Dummy_Classifier")
    prediction.evaluate.direction_error_metrics(dummy_sets[1], dummy_sets[3], "Dummy_Classifier")
    # KNeighbors_Classifier
    prediction.evaluate.direction_error_metrics(kn_sets[0], kn_sets[2], "KNeighbors_Classifier")
    prediction.evaluate.direction_error_metrics(kn_sets[1], kn_sets[3], "KNeighbors_Classifier")
    # Decision_Tree_Classifier
    prediction.evaluate.direction_error_metrics(dt_sets[0], dt_sets[2], "Decision_Tree_Classifier")
    prediction.evaluate.direction_error_metrics(dt_sets[1], dt_sets[3], "Decision_Tree_Classifier")
    # Random_Forest_Classifier
    prediction.evaluate.direction_error_metrics(rf_sets[0], rf_sets[2], "Random_Forest_Classifier")
    prediction.evaluate.direction_error_metrics(rf_sets[1], rf_sets[3], "Random_Forest_Classifier")
    # Neural Network Classifier
    prediction.evaluate.direction_error_metrics(nn_sets[0], nn_sets[2], "NN_Classifier")
    prediction.evaluate.direction_error_metrics(nn_sets[1], nn_sets[3], "NN_Classifier")


def testing_direction_subsets():
    # TODO: Docstring
    # data
    df_features = io.input.read_csv(p_filename="Features_Direction.csv", p_io_folder="output")
    testing.direction_classification_subsets_testing.filter_subsets(df_features)


def predict_direction_models(trips_file, weather=False):
    # TODO: Docstring
    # Prepare
    if trips_file:
        # first create features for direction prediction out of trips data
        features_direction(trips_file, weather)
    df_features = io.input.read_csv(p_filename="Features_Direction.csv", p_io_folder="output")
    print("Split Data...")
    X_train, X_test, y_train, y_test = prediction.split.simple_split_direction(df_features)
    # Predict
    print("Predict by Dummy Classification...")
    dummy_y_prediction = prediction.geo_predict.predict_by_dummy_classificaion(X_test)
    print("Predict by KNeighbors Classificaion...")
    kn_y_prediction = prediction.geo_predict.predict_by_k_neighbors_classificaion(X_test)
    print("predict_by_decision_tree_classificaion...")
    dt_y_prediction = prediction.geo_predict.predict_by_decision_tree_classificaion(X_test)
    print("predict_by_random_forest_classificaion")
    rf_y_prediction = prediction.geo_predict.predict_by_random_forest_classificaion(X_test)
    print("predict_by_neural_network_classificaion")
    nn_y_prediction = prediction.geo_predict.predict_by_neural_network_classificaion(X_test)
    # Evaluate Prediction
    prediction.evaluate.direction_error_metrics(y_test, dummy_y_prediction, "Dummy_Classifier", "Testing")
    prediction.evaluate.direction_error_metrics(y_test, kn_y_prediction, "KNeighbors_Classifier", "Testing")
    prediction.evaluate.direction_error_metrics(y_test, dt_y_prediction, "Decision_Tree_Classifier", "Testing")
    prediction.evaluate.direction_error_metrics(y_test, rf_y_prediction, "Random_Forest_Classifier", "Testing")
    prediction.evaluate.direction_error_metrics(y_test, nn_y_prediction, "NN_Classifier", "Testing")
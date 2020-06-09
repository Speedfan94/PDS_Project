from nextbike import prediction
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA


def test_robust_scaler(p_df, p_weather):
    """Tests robust scaler to not kick out noise hardcoded. Does PCA and trains neural networks with scaled data.
    Evaluates performance of robust scaler.

    For us, the robust scaler did not perform that good

    Args:
        p_df:       Whole data set
        p_weather:  file ending when weather data is included
    Returns:
        No return
    """
    # Prepare Train
    print("Split Data...")
    X_train, X_test, y_train, y_test = prediction.split.simple_split_duration(p_df)
    print("Robust Scale Data...")
    robust_scaler = RobustScaler()
    X_scaled_train = robust_scaler.fit_transform(X_train)
    print("Do PCA...")
    pca = PCA(n_components=21)
    pca.fit(X_scaled_train)
    X_train_transformed = pca.transform(X_scaled_train)
    # Train
    print("Train NN...")
    nn_regr_sets = prediction.math_train.train_neural_network(X_train_transformed, y_train, p_testing=True, p_weather=p_weather)
    # Evaluate Training
    prediction.evaluate.duration_error_metrics(nn_regr_sets[0], nn_regr_sets[2], "NN_Regression_Training"+p_weather)
    prediction.evaluate.duration_error_metrics(nn_regr_sets[1], nn_regr_sets[3], "NN_Regression_Validation"+p_weather)

    # Prepare Predict
    X_scaled_test = robust_scaler.transform(X_test)
    X_test_transformed = pca.transform(X_scaled_test)
    # Predict
    print("Predict by NN...")
    nn_y_prediction = prediction.math_predict.predict_by_nn(X_test_transformed, p_testing=True)
    # Evaluate Prediction
    prediction.evaluate.duration_error_metrics(y_test, nn_y_prediction, "NN_Regression", "Testing")

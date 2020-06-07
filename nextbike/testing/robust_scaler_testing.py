from .. import prediction
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA


def test_robust_scaler(p_df):
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
    nn_y_prediction = prediction.math_train.train_neural_network(X_train_transformed, y_train, p_testing=True)
    # Evaluate Training
    prediction.evaluate.duration_error_metrics(y_train, nn_y_prediction, "NN_Regression")
    # Prepare Predict
    X_scaled_test = robust_scaler.transform(X_test)
    X_test_transformed = pca.transform(X_scaled_test)
    # Predict
    print("Predict by NN...")
    nn_y_prediction = prediction.math_predict.predict_by_nn(X_test_transformed, p_testing=True)
    # Evaluate Prediction
    prediction.evaluate.duration_error_metrics(y_test, nn_y_prediction, "NN_Regression", "Testing")

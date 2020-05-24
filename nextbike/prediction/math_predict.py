from .. import io
from sklearn import metrics
from tensorflow.keras.models import load_model
import numpy as np


# TODO: Add docstring
def predict_by_regression(p_X_test, p_y_test):
    scaler = io.read_object("Standard_Scaler.pkl")
    model = io.read_object("Linear_Regression_Model.pkl")
    pca = io.read_object("PCA.pkl")
    X_test_scaled = scaler.transform(p_X_test)
    X_test_transformed = pca.transform(X_test_scaled)
    y_predictions = model.predict(X_test_transformed)
    # TODO: Is it fine to call this method from here or should I return to cli.py and then evaluate metrics?
    show_errors_metrics(p_y_test, y_predictions, "Linear_Regression_Prediction.png")


# TODO: Add docstring
def predict_by_nn(p_X_test, p_y_test):
    scaler = io.read_object("Standard_Scaler.pkl")
    model = load_model(io.get_path("Neural_Network_Model", "output", "models"))
    pca = io.read_object("PCA.pkl")
    X_test_scaled = scaler.transform(p_X_test)
    X_test_transformed = pca.transform(X_test_scaled)
    y_predictions = model.predict(X_test_transformed)
    show_errors_metrics(p_y_test, y_predictions, "NN_Regression_Model")


# TODO: Add docstring
def predict_by_svm(p_X_test, p_y_test):
    scaler = io.read_object("Standard_Scaler.pkl")
    pca = io.read_object("PCA.pkl")
    X_test_scaled = scaler.transform(p_X_test)
    X_test_transformed = pca.transform(X_test_scaled)
    model = io.read_object("SVM_Regression_Model_"+str(3)+".pkl")
    y_predictions = model.predict(X_test_transformed)
    show_errors_metrics(p_y_test, y_predictions, "SVM_Regression_Prediction_" + str(3))


# TODO: Add docstring
def show_errors_metrics(p_y_true, p_y_predictions, p_filename):
    print(p_filename)
    print("RMSE:", np.sqrt(metrics.mean_squared_error(p_y_true, p_y_predictions)))
    print("MAE", metrics.mean_absolute_error(p_y_true, p_y_predictions))

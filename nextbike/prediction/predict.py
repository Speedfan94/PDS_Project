from .. import io
from sklearn import metrics
import numpy as np


def predict_by_regression(pX_test, py_test):
    scaler = io.read_object("Standard_Scaler.pkl")
    model = io.read_object("Linear_Regression_Model.pkl")
    pca = io.read_object("PCA.pkl")
    X_test_scaled = scaler.transform(pX_test)
    X_test_transformed = pca.transform(X_test_scaled)
    y_predictions = model.predict(X_test_transformed)
    # TODO: Is it fine to call this method from here or should I return to cli.py and then evaluate metrics?
    show_errors_metrics(py_test, y_predictions, "Linear_Regression_Prediction.png")


def show_errors_metrics(py_true, py_predictions, pFilename):
    print(py_true.max())
    print(py_predictions.max())

    print("RMSE:", np.sqrt(metrics.mean_squared_error(py_true, py_predictions)))
    print("MAE", metrics.mean_absolute_error(py_true, py_predictions))

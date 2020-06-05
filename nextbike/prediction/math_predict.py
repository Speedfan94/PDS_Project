from .. import io
from .. import visualization
from sklearn import metrics
from tensorflow.keras.models import load_model
import numpy as np


def predict_by_regression(p_X_test, p_y_test):
    """Predicts the duration of a trip by trained Linear Regression model.

    This method uses the trained scaler, pca and linear regression model to predict the given test set duration.
    Then a method to evaluate the performance is called.
    Args:
        p_X_test (DataFrame): Dataframe of input features for prediction (matrix)
        p_y_test (Series): Series of true output durations (vector)
    Returns:
        no Return
    """
    # Read pickle objects
    scaler = io.read_object("Standard_Scaler.pkl")
    pca = io.read_object("PCA.pkl")
    model = io.read_object("Linear_Regression_Model.pkl")
    # Use trained pickle objects
    X_test_scaled = scaler.transform(p_X_test)
    X_test_transformed = pca.transform(X_test_scaled)
    y_predictions = model.predict(X_test_transformed)
    # Evaluate prediction
    show_errors_metrics(p_y_test, y_predictions, "01_Linear_Regression_Prediction")


def predict_by_nn(p_X_test, p_y_test):
    """Predicts the duration of a trip by trained neural network model.

    This method uses the trained scaler, pca and Neural Network model to predict the given test set duration.
    Then a method to evaluate the performance is called.
    Args:
        p_X_test (DataFrame): Dataframe of input features for prediction (matrix)
        p_y_test (Series): Series of true output durations (vector)
    Returns:
        no Return
    """
    # Read pickle objects
    scaler = io.read_object("Standard_Scaler.pkl")
    pca = io.read_object("PCA.pkl")
    model = load_model(io.get_path("Neural_Network_Model", "output", "models"))
    # Use trained pickle objects
    X_test_scaled = scaler.transform(p_X_test)
    X_test_transformed = pca.transform(X_test_scaled)
    y_predictions = model.predict(X_test_transformed)
    # Evaluate prediction
    show_errors_metrics(p_y_test, y_predictions, "03_NN_Regression_Model")


def predict_by_svm(p_X_test, p_y_test):
    """Predicts the duration of a trip by trained Support Vector Machine model.

    This method uses the trained scaler, pca and Support Vector Machine model to predict the given test set duration.
    Then a method to evaluate the performance is called.
    Args:
        p_X_test (DataFrame): Dataframe of input features for prediction (matrix)
        p_y_test (Series): Series of true output durations (vector)
    Returns:
        no Return
    """
    # Read pickle objects
    scaler = io.read_object("Standard_Scaler.pkl")
    pca = io.read_object("PCA.pkl")
    model = io.read_object("SVM_Regression_Model_3.pkl")
    # Use trained pickle objects
    X_test_scaled = scaler.transform(p_X_test)
    X_test_transformed = pca.transform(X_test_scaled)
    y_predictions = model.predict(X_test_transformed)
    # Evaluate prediction
    show_errors_metrics(p_y_test, y_predictions, "02_SVM_Regression_Prediction_3")


def show_errors_metrics(p_y_true, p_y_predictions, p_filename):
    """Evaluate the error metrics for a prediction by a model.

    This method is called after the prediction of a given model. It evaluates the performance of the prediction by
        - Root Mean Squared Error
        - Mean Absolute Error
    Then a visualization method is called to visualize the performance of the prediction of the given model.
    Args:
        p_y_true (Series): Series of true output durations (vector)
        p_y_predictions (Series): Series of predicted output durations (vector)
        p_filename (str): String with the model name
    Returns:
        no Return
    """
    print(p_filename, "Test loss - Error Metrics:")
    print("RMSE:", np.sqrt(metrics.mean_squared_error(p_y_true, p_y_predictions)), end=" ")
    print("MAE", metrics.mean_absolute_error(p_y_true, p_y_predictions))
    visualization.math_predictive.plot_true_vs_predicted(p_y_true, p_y_predictions, p_filename)

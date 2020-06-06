from .. import visualization
import numpy as np
from sklearn import metrics


def duration_error_metrics(p_y_true, p_y_predictions, p_filename, p_status="Training"):
    """Evaluate the trained models by error metrics.

    Print for the given model the following error metrics:
        - Root Mean Squared Error
        - Mean Absolute Error
        - R^2
    Args:
        p_y_true (Series): True values of duration (vector)
        p_y_predictions (Series): Predicted values for train set of duration (vector)
        p_filename (str): Name of the used model
        p_status (str): string which states if metrics are for training or testing
    Returns:
        no return
    """
    print(p_filename, p_status, "loss - Error Metrics:")
    print("RMSE:", np.sqrt(metrics.mean_squared_error(p_y_true, p_y_predictions)), end=" ")
    print("MAE:", metrics.mean_absolute_error(p_y_true, p_y_predictions), end=" ")
    # The coefficient of determination: 1 is perfect prediction
    print("R^2:", metrics.r2_score(p_y_true, p_y_predictions))
    visualization.math_predictive.plot_true_vs_predicted(p_y_true, p_y_predictions, p_filename)


def direction_error_metrics(p_y_true, p_y_predictions, p_filename, p_status="Training"):
    # TODO: docstring
    print(p_filename, p_status, "loss - Error Metrics:")
    print("Accuracy:", metrics.accuracy_score(p_y_true, p_y_predictions), end=" ")
    print("Confusion Matrix:", metrics.confusion_matrix(p_y_true, p_y_predictions).flatten(), end=" ")
    # zero_division states if dive by 0 error should be raised. With 0 only the value 0 is printed.
    print("Precision:", metrics.precision_score(p_y_true, p_y_predictions, zero_division=0), end=" ")
    print("Recall:", metrics.recall_score(p_y_true, p_y_predictions), end=" ")
    print("F1 score", metrics.f1_score(p_y_true, p_y_predictions))

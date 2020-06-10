from nextbike import visualization
import numpy as np
from sklearn import metrics


def duration_error_metrics(p_y_true, p_y_predictions, p_filename, p_status="Training"):
    """Evaluate the trained models by error metrics.

    Print for the given model the following error metrics:
        - Root Mean Squared Error
        - Mean Absolute Error
        - R^2

    Args:
        p_y_true (Series):          True values of duration (vector)
        p_y_predictions (Series):   Predicted values for train set of duration (vector)
        p_filename (str):           Name of the used model
        p_status (str):             string which states if metrics are for training or validation
    Returns:
        No return
    """
    print(p_filename, p_status, "loss - Error Metrics:")
    print("RMSE:", np.sqrt(metrics.mean_squared_error(p_y_true, p_y_predictions)), end=" ")
    print("MAE:", metrics.mean_absolute_error(p_y_true, p_y_predictions), end=" ")
    # The coefficient of determination: 1 is perfect prediction
    print("R^2:", metrics.r2_score(p_y_true, p_y_predictions))
    visualization.math_predictive.plot_true_vs_predicted(p_y_true, p_y_predictions, p_filename, p_status)


def direction_error_metrics(p_y_true, p_y_predictions, p_filename, p_status="Training"):
    """Evaluate the trained models by error metrics.

    Print for the given model the following error metrics:
        - Accuracy
        - Confusion matrix
        - Precision
        - Recall
        - F1 score

    Args:
        p_y_true:           True values of direction (vector)
        p_y_predictions:    Predicted values for train set of direction (vector)
        p_filename:         Name of the used model
        p_status:           string which states if metrics are for training or testing
    Returns:
        No return
    """

    accuracy = metrics.accuracy_score(p_y_true, p_y_predictions)
    confusion_matrix = metrics.confusion_matrix(p_y_true, p_y_predictions).flatten() # ohne Flatten: [[1, 4] [2, 5]] mit Flatten:[1, 4, 2, 5]
    precision = metrics.precision_score(p_y_true, p_y_predictions, zero_division=0)

    recall = metrics.recall_score(p_y_true, p_y_predictions)
    f1_score = metrics.f1_score(p_y_true, p_y_predictions)

    print(p_filename, p_status, "Metrics:", end =" ")
    print("Accuracy:", accuracy, end=" ")
    print("Confusion Matrix:", confusion_matrix, end=" ")
    # zero_division states if dive by 0 error should be raised. With 0 only the value 0 is printed.
    print("Precision:", precision, end=" ")
    print("Recall:", recall, end=" ")
    print("F1 score", f1_score)


from nextbike import io
from tensorflow.keras.models import load_model


def predict_by_dummy_mean(p_X_test, p_weather=""):
    """Predicts the duration of a trip by trained Dummy Mean model.

    This method uses the trained scaler, pca and Dummy Mean Regression model to predict the given test set duration.
    Then a method to evaluate the performance is called.

    Args:
        p_X_test:       X_test set
        p_weather:      file ending when weather data is included
    Returns:
        y_predictions:  predicted y values for duration
    """
    # Read pickle objects
    scaler = io.read_object("Standard_Scaler_Duration"+p_weather+".pkl")
    pca = io.read_object("PCA_Duration"+p_weather+".pkl")
    model = io.read_object("Dummy_Mean_Regression_Model"+p_weather+".pkl")
    # Use trained pickle objects
    X_test_scaled = scaler.transform(p_X_test)
    X_test_transformed = pca.transform(X_test_scaled)
    y_predictions = model.predict(X_test_transformed)
    return y_predictions


def predict_by_dummy_median(p_X_test, p_weather=""):
    """Predicts the duration of a trip by trained Dummy Median model.

    This method uses the trained scaler, pca and Dummy Median Regression model to predict the given test set duration.
    Then a method to evaluate the performance is called.

    Args:
        p_X_test:       X_test set
        p_weather:      file ending when weather data is included
    Returns:
        y_predictions:  predicted y values for duration
    """
    # Read pickle objects
    scaler = io.read_object("Standard_Scaler_Duration"+p_weather+".pkl")
    pca = io.read_object("PCA_Duration"+p_weather+".pkl")
    model = io.read_object("Dummy_Median_Regression_Model"+p_weather+".pkl")
    # Use trained pickle objects
    X_test_scaled = scaler.transform(p_X_test)
    X_test_transformed = pca.transform(X_test_scaled)
    y_predictions = model.predict(X_test_transformed)
    return y_predictions


def predict_by_regression(p_X_test, p_weather=""):
    """Predicts the duration of a trip by trained Linear Regression model.

    This method uses the trained scaler, pca and linear regression model to predict the given test set duration.
    Then a method to evaluate the performance is called.

    Args:
        p_X_test (DataFrame):   Dataframe of input features for prediction (matrix)
        p_weather:              file ending when weather data is included
    Returns:
        y_predictions:          predicted y values for durations
    """
    # Read pickle objects
    scaler = io.read_object("Standard_Scaler_Duration"+p_weather+".pkl")
    pca = io.read_object("PCA_Duration"+p_weather+".pkl")
    model = io.read_object("Linear_Regression_Model"+p_weather+".pkl")
    # Use trained pickle objects
    X_test_scaled = scaler.transform(p_X_test)
    X_test_transformed = pca.transform(X_test_scaled)
    y_predictions = model.predict(X_test_transformed)
    return y_predictions


def predict_by_nn(p_X_test, p_testing=False, p_weather=""):
    """Predicts the duration of a trip by trained neural network model.

    This method uses the trained scaler, pca and Neural Network model to predict the given test set duration.
    Then a method to evaluate the performance is called.

    Args:
        p_X_test (DataFrame):   Dataframe of input features for prediction (matrix)
        p_testing (Boolean):    States if Testing is done
        p_weather:              file ending when weather data is included
    Returns:
        y_predictions           predicted y values for durations
    """
    # Read pickle objects
    if p_testing:
        model = load_model(io.get_path("TEST_Neural_Network_Regression_Model"+p_weather+"", "output", "models"))
        y_predictions = model.predict(p_X_test)
    else:
        scaler = io.read_object("Standard_Scaler_Duration"+p_weather+".pkl")
        pca = io.read_object("PCA_Duration"+p_weather+".pkl")
        model = load_model(io.get_path("Neural_Network_Regression_Model"+p_weather+"", "output", "models"))
        # Use trained pickle objects
        X_test_scaled = scaler.transform(p_X_test)
        X_test_transformed = pca.transform(X_test_scaled)
        y_predictions = model.predict(X_test_transformed)
    return y_predictions


def predict_by_svm(p_X_test, p_weather=""):
    """Predicts the duration of a trip by trained Support Vector Machine model.

    This method uses the trained scaler, pca and Support Vector Machine model to predict the given test set duration.
    Then a method to evaluate the performance is called.

    Args:
        p_X_test (DataFrame):   Dataframe of input features for prediction (matrix)
        p_weather:              file ending when weather data is included
    Returns:
        y_predictions           predicted y values for duration
    """
    # Read pickle objects
    scaler = io.read_object("Standard_Scaler_Duration"+p_weather+".pkl")
    pca = io.read_object("PCA_Duration"+p_weather+".pkl")
    model = io.read_object("SVM_Regression_Model_3"+p_weather+".pkl")
    # Use trained pickle objects
    X_test_scaled = scaler.transform(p_X_test)
    X_test_transformed = pca.transform(X_test_scaled)
    y_predictions = model.predict(X_test_transformed)
    return y_predictions

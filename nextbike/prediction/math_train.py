from nextbike import io
from nextbike import visualization
import warnings

from sklearn.linear_model import LinearRegression
from sklearn import svm
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor


def train_dummy_regression_median(p_X_train_scaled, p_y_train, p_weather):
    """Train Dummy Regression Median Model

    Train and save a Dummy Regression Median model. Then evaluate the error metrics by another method.

    Args:
        p_X_train_scaled:   scaled X train data
        p_y_train:          y train data
        p_weather:          file ending when weather data is included
    Return:
        lin_regr_sets:      linear regression sets
    """
    # create a validation set which is 20% of the whole dataset. Therefore use formula to receive ca. 0.2857.
    X_train, X_val, y_train, y_val = train_test_split(p_X_train_scaled, p_y_train, random_state=42, test_size=0.2 / 0.7)
    dummy_reg = DummyRegressor(strategy="median")
    dummy_reg.fit(X_train, y_train)
    io.save_object(dummy_reg, "Dummy_Median_Regression_Model"+p_weather+".pkl")
    y_prediction_train = dummy_reg.predict(X_train)
    y_prediction_val = dummy_reg.predict(X_val)
    lin_regr_sets = [y_train, y_val, y_prediction_train, y_prediction_val]
    return lin_regr_sets


def train_dummy_regression_mean(p_X_train_scaled, p_y_train, p_weather):
    """Train Dummy Regression Mean Model

    Train and save a Dummy Regression Mean model. Then evaluate the error metrics by another method.

    Args:
        p_X_train_scaled:   scaled X train data
        p_y_train:          y train data
        p_weather:          file ending when weather data is included
    Return:
        lin_regr_sets:      linear regression sets
    """
    # create a validation set which is 20% of the whole dataset. Therefore use formula to receive ca. 0.2857.
    X_train, X_val, y_train, y_val = train_test_split(p_X_train_scaled, p_y_train, random_state=42, test_size=0.2 / 0.7)
    dummy_reg = DummyRegressor(strategy="mean")
    dummy_reg.fit(X_train, y_train)
    io.save_object(dummy_reg, "Dummy_Mean_Regression_Model"+p_weather+".pkl")
    y_prediction_train = dummy_reg.predict(X_train)
    y_prediction_val = dummy_reg.predict(X_val)
    lin_regr_sets = [y_train, y_val, y_prediction_train, y_prediction_val]
    return lin_regr_sets


def train_linear_regression(p_X_train_scaled, p_y_train, p_weather):
    """Train Linear Regression Model

    Train and save a Linear Regression model. Then evaluate the error metrics by another method.

    Args:
        p_X_train_scaled (DataFrame): Scaled X input of train set (matrix)
        p_y_train (Series): y output to train on (vector)
    Returns:
        No return
    """
    # create a validation set which is 20% of the whole dataset. Therefore use formula to receive ca. 0.2857.
    X_train, X_val, y_train, y_val = train_test_split(p_X_train_scaled, p_y_train, random_state=42, test_size=0.2/0.7)
    lin = LinearRegression()
    lin.fit(X_train, y_train)
    # print("Coefficients: ", lin.coef_)
    io.save_object(lin, "Linear_Regression_Model"+p_weather+".pkl")
    y_prediction_train = lin.predict(X_train)
    y_prediction_val = lin.predict(X_val)
    lin_regr_sets = [y_train, y_val, y_prediction_train, y_prediction_val]
    return lin_regr_sets


def train_neural_network(p_X_train_scaled, p_y_train, p_weather, p_testing=False):
    """Train Neural Network Model

    Train and save a Neural Network model.
    The network has the following properties:
        - one hidden layer
        - 10 epochs
        - activation function is relu
        - dimension of input and hidden layer is 36
        - dimension of output layer is 1
        - dropout is not used
    Then evaluate the error metrics by another method.

    Args:
        p_X_train_scaled (DataFrame): Scaled X input of train set (matrix)
        p_y_train (Series): y output to train on (vector)
        p_testing (Boolean): states if testing should be done
    Returns:
        No return
    """
    # create a validation set which is 20% of the whole dataset. Therefore use formula to receive ca. 0.2857.
    X_train, X_val, y_train, y_val = train_test_split(p_X_train_scaled, p_y_train, random_state=42, test_size=0.2 / 0.7)
    neural_network = keras.Sequential(
        [layers.Dense(36, activation="relu", input_shape=[p_X_train_scaled.shape[1]], kernel_initializer="random_normal"),
         # layers.Dropout(0.2),
         layers.Dense(36, activation="relu", kernel_initializer="random_normal"),
         layers.Dense(36, activation="relu", kernel_initializer="random_normal"),
         layers.Dense(36, activation="relu", kernel_initializer="random_normal"),
         #layers.Dense(36, activation="relu", kernel_initializer="random_normal"),
         #layers.Dense(36, activation="relu", kernel_initializer="random_normal"),
         # layers.Dense(36, activation="softmax"),
         # layers.Dense(36, activation="softmax"),
         # layers.Dropout(0.2),
         layers.Dense(1)])
    optimizer = keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)
    neural_network.compile(loss="mse",
                           optimizer=optimizer,
                           metrics=["mae", "mse"])
    epochs = 50
    # create a validation set which is 20% of the whole dataset. Therefore use formula to receive ca. 0.2857.
    history = neural_network.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val))
    if p_testing:
        neural_network.save(io.get_path("TEST_Neural_Network_Regression_Model"+p_weather, "output", "models"))
    else:
        neural_network.save(io.get_path("Neural_Network_Regression_Model"+p_weather, "output", "models"))
    y_prediction_train = neural_network.predict(X_train)
    y_prediction_val = neural_network.predict(X_val)
    visualization.math_predictive.plot_train_loss(history, p_weather=p_weather)
    nn_regression_sets = [y_train, y_val, y_prediction_train, y_prediction_val]
    return nn_regression_sets


def train_svm(p_X_train_scaled, p_y_train, p_weather):
    """Train Support Vector Machine Model

    Train and save a Support Vector Machine model.
    SVM is not the best model. Optimal would be max iteration of -1,
    but we decided to leave it at 1000 as it would run forever and we assume
    that it would not deliver that much better results
    The properties of the SVM are:
        - max iterations are 1000
        - degree is 1
        - kernel is linear
        - cache_size is 2000 kb
        - gamma regularization
    Then evaluate the error metrics by another method.

    Args:
        p_X_train_scaled (DataFrame): Scaled X input of train set (matrix)
        p_y_train (Series): y output to train on (vector)
    Returns:
        svm_regression_sets (List): list of y_train, y_val, y_prediction_train, y_prediction_val
    """
    # create a validation set which is 20% of the whole dataset. Therefore use formula to receive ca. 0.2857.
    X_train, X_val, y_train, y_val = train_test_split(p_X_train_scaled, p_y_train, random_state=42, test_size=0.2 / 0.7)
    warnings.filterwarnings("ignore", "Solver terminated early.*")
    # specify kernel{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}, default=’rbf’
    # degreeint, default=3
    # max_iterint, default=-1 => no limit
    # verbose=1
    svm_regr = svm.SVR(kernel="poly", max_iter=1000, cache_size=2000, degree=3, gamma="auto")
    svm_regr.fit(X_train, y_train)
    io.save_object(svm_regr, "SVM_Regression_Model_3"+p_weather+".pkl")
    y_prediction_train = svm_regr.predict(X_train)
    y_prediction_val = svm_regr.predict(X_val)
    svm_regression_sets = [y_train, y_val, y_prediction_train, y_prediction_val]
    return svm_regression_sets

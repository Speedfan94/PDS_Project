import pandas as pd
from .. import io
import warnings

from sklearn.linear_model import LinearRegression
from sklearn import svm
from tensorflow import keras
from tensorflow.keras import layers


def train_linear_regression(pX_train_scaled, py_train):
    lin = LinearRegression()
    lin.fit(pX_train_scaled, py_train)
    io.save_object(lin, "Linear_Regression_Model.pkl")


def train_neural_network(pX_train_scaled, py_train):
    """train neural network

    Args:
        pX_train_scaled (DataFrame): Scaled x input of train set
        py_train (Series): y output to train on
    Returns:
        no return
    """
    neural_network = keras.Sequential(
        [layers.Dense(36, activation="relu", input_shape=[pX_train_scaled.shape[1]]),
         # layers.Dropout(0.2),
         layers.Dense(36, activation="relu"),
         # layers.Dropout(0.2),
         layers.Dense(1)])
    optimizer = keras.optimizers.RMSprop(0.001)
    neural_network.compile(loss="mse",
                           optimizer=optimizer,
                           metrics=["mae", "mse"])
    epochs = 10
    batch_size = 200  # right now not used but should be tried
    neural_network.fit(pX_train_scaled, py_train.values, epochs=epochs, validation_split=0.2)
    neural_network.save(io.get_path("Neural_Network_Model", "output", "models"))


def train_svm(pX_train_scaled, py_train):
    """Trains a svm

    Args:
        pX_train_scaled (DataFrame): Scaled x input of train set
        py_train (Series): y output to train on
    Returns:
        no return
    """
    warnings.filterwarnings('ignore', 'Solver terminated early.*')
    # specify kernel{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}, default=’rbf’
    # degreeint, default=3
    # max_iterint, default=-1 => no limit
    # verbose=1
    regr = svm.SVR(max_iter=1000, cache_size=2000, degree=3)
    regr.fit(pX_train_scaled, py_train)
    io.save_object(regr, "SVM_Regression_Model_" + str(3) + ".pkl")

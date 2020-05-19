from sklearn.linear_model import LinearRegression
from .. import io


def train_linear_regression(pX_train_scaled, py_train):
    lin = LinearRegression()
    lin.fit(pX_train_scaled, py_train)
    io.save_object(lin, "Linear_Regression_Model.pkl")
    # Uncommend for showing
    # print("Intercept=", lin.intercept_, "Coefficients=", lin.coef_)


def train_neural_network(X_train, y_train):
    print()


def train_svm(X_train, y_train):
    print()

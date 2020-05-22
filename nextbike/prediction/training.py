from sklearn.linear_model import LinearRegression
from sklearn import svm
from .. import io
import warnings

def train_linear_regression(pX_train_scaled, py_train):
    lin = LinearRegression()
    lin.fit(pX_train_scaled, py_train)
    io.save_object(lin, "Linear_Regression_Model.pkl")


def train_neural_network(X_train, y_train):
    print()


def train_svm(pX_train_scaled, py_train):
    warnings.filterwarnings('ignore', 'Solver terminated early.*')
    # specify kernel{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}, default=’rbf’
    # degreeint, default=3
    # max_iterint, default=-1 => no limit
    # verbose=1
    for i in range(4):
        print("Train SVM of degree "+str(i))
        regr = svm.SVR(max_iter=1000, cache_size=2000, degree=i)
        regr.fit(pX_train_scaled, py_train)
        io.save_object(regr, "SVM_Regression_Model_"+str(i)+".pkl")



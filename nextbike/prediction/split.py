# splitting
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

# measures
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics


def simple_split(df):
    X = df.drop(["Duration"], axis=1)
    y = df["Duration"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


def k_fold_split(df):
    X = df.drop(["Duration"], axis=1)
    y = df["Duration"]

    # the more folds we have, we will be reducing the error due the bias but increasing the error due to variance
    kf = KFold(n_splits=3)  # folds
    kf.get_n_splits(X)
    show_fold(kf, X, y)


def show_fold(pKf, pX, py):    # prints the index of the data split
    for train_index, test_index in pKf.split(pX):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = pX.loc[train_index], pX.loc[test_index]
        y_train, y_test = py.loc[train_index], py.loc[test_index]


def cross_validation(pModel, pX, py):
    # Perform 3-fold cross validation
    scores = cross_val_score(pModel, pX, py, cv=3)
    print("Cross - validated scores:", scores)



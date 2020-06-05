# splitting
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

# measures
from sklearn.model_selection import cross_val_score, cross_val_predict


def simple_split(p_df):
    """Splits the data in train and test set.

    Create train and the matrices X and train and test vector y.
        - X are the independet variables
        - y are the dependent variables
    The test set size is set to 30% of the data.
    The random state 42 is used such that the split can be replicated later on.
    Args:
        p_df (DataFrame): Dataframe of independent variables (matrix)
    Returns:
        X_train (DataFrame): Dataframe of independent variables for train set (matrix)
        X_test (DataFrame): Dataframe of independent variables for test set (matrix)
        y_train (Series): Series of dependent variables for train set (vector)
        y_test (Series): Series of dependent variables for test set (vector)
    """
    X = p_df.drop(["Duration"], axis=1)
    y = p_df["Duration"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


# TODO: Add docstring
def k_fold_split(p_df):
    X = p_df.drop(["Duration"], axis=1)
    y = p_df["Duration"]

    # the more folds we have, we will be reducing the error due the bias but increasing the error due to variance
    kf = KFold(n_splits=3)  # folds
    kf.get_n_splits(X)
    show_fold(kf, X, y)


# TODO: Add docstring
def show_fold(p_kf, p_X, p_y):    # prints the index of the data split
    for train_index, test_index in p_kf.split(p_X):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = p_X.loc[train_index], p_X.loc[test_index]
        y_train, y_test = p_y.loc[train_index], p_y.loc[test_index]


# TODO: Add docstring
def cross_validation(p_model, p_X, p_y):
    # Perform 3-fold cross validation
    scores = cross_val_score(p_model, p_X, p_y, cv=3)
    print("Cross - validated scores:", scores)

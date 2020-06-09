# splitting
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

# measures
from sklearn.model_selection import cross_val_score, cross_val_predict


def simple_split_duration(p_df):
    """Splits the data in train and test set for regression on trip duration.

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


def simple_split_direction(p_df):
    """Splits the data in train and test set for classification of trip direction.

    Create train and test matrices X and train and test vector y.
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
    X = p_df.drop(["Direction"], axis=1)
    y = p_df["Direction"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

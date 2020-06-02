import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def train_pred(p_df):
    """Train and predict KNN Classifier for trip direction.

    Train and predict if a trips end is nearer the university than the start.
    Print the accuracy of the prdiction.
    Args:
        p_df (DataFrame): Dataframe of trip data
    Returns:
        no Return
    """
    X = p_df[["Duration", "Dist_start", "Bike Number"]]
    y = p_df["Direction"]
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = KNeighborsClassifier()
    clf.fit(X_train, y_train)
    print("Test score (mean accuracy)", np.round(clf.score(X_test, y_test)*100, 2), "%")


# TODO: add additional methods for prediction

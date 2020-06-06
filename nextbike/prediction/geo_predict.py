from .. import visualization
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB


def train_pred(p_df):
    """Train and predict different classification models for trip direction.

    Train and predict if trip's end is nearer the university than the start.
    Print the accuracy of the prediction.
    Args:
        p_df (DataFrame): Dataframe of trip data
    Returns:
        no Return
    """
    # data
    X = p_df.drop("Direction", axis=1)
    y = p_df["Direction"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    # train
    # KNN
    print("Train KNN")
    clf = KNeighborsClassifier(n_neighbors=20, weights="distance")
    clf.fit(X_train_scaled, y_train)
    # Decision Tree
    print("Train Decision Tree")
    dt = DecisionTreeClassifier(max_depth=5)
    dt.fit(X_train_scaled, y_train)
    # RF
    print("Train Random Forest")
    rf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
    rf.fit(X_train_scaled, y_train)
    # NN
    print("Train Neural Network")
    nn = MLPClassifier(alpha=1, max_iter=1000)
    nn.fit(X_train_scaled, y_train)
    # GNB
    #print("Gaussian Naive Bayes")
    #gnb = GaussianNB()
    #gnb.fit(X_train_scaled, y_train)
    # predict
    X_test_scaled = scaler.transform(X_test)
    print("Predict KNN")
    print("KNN: Test score (mean accuracy)", np.round(clf.score(X_test_scaled, y_test) * 100, 2), "%")
    print("Predict DT")
    print("DT: Test score (mean accuracy)", np.round(dt.score(X_test_scaled, y_test) * 100, 2), "%")
    print("Predict RF")
    print("RF: Test score (mean accuracy)", np.round(rf.score(X_test_scaled, y_test) * 100, 2), "%")
    print("Predict NN")
    print("NN: Test score (mean accuracy)", np.round(nn.score(X_test_scaled, y_test) * 100, 2), "%")
    #print("GNB: Test score (mean accuracy)", np.round(gnb.score(X_test_scaled, y_test) * 100, 2), "%")
    # TODO: Visualization of 1-dimension?
    #visualization.math_predictive.plot_direction_classification(X, y)


# TODO: add additional methods for prediction

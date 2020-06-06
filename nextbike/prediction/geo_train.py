from .. import io
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
# TODO: Docstrings


def train_classification_dummy(p_X_train_scaled, p_y_train):
    dc = DummyClassifier(strategy="most_frequent")
    dc.fit(p_X_train_scaled, p_y_train)
    y_predictions = dc.predict(p_X_train_scaled)
    io.save_object(dc, "Dummy_classifier_model.pkl")
    return y_predictions


def train_classification_k_neighbors(p_X_train_scaled, p_y_train):
    clf = KNeighborsClassifier(n_neighbors=20, weights="distance")
    clf.fit(p_X_train_scaled, p_y_train)
    y_predictions = clf.predict(p_X_train_scaled)
    io.save_object(clf, "KNearestNeighbours_classifier_model.pkl")
    return y_predictions


def train_classification_decision_tree(p_X_train_scaled, p_y_train):
    dt = DecisionTreeClassifier(max_depth=5)
    dt.fit(p_X_train_scaled, p_y_train)
    y_predictions = dt.predict(p_X_train_scaled)
    io.save_object(dt, "DecisionTree_classifier_model.pkl")
    return y_predictions


def train_classification_random_forest(p_X_train_scaled, p_y_train):
    rf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
    rf.fit(p_X_train_scaled, p_y_train)
    y_predictions = rf.predict(p_X_train_scaled)
    io.save_object(rf, "RandomForest_classifier_model.pkl")
    return y_predictions


def train_classification_neural_network(p_X_train_scaled, p_y_train):
    nn = MLPClassifier(alpha=1, max_iter=1000)
    nn.fit(p_X_train_scaled, p_y_train)
    y_predictions = nn.predict(p_X_train_scaled)
    io.save_object(nn, "NeuralNetwork_classifier_model.pkl")
    return y_predictions

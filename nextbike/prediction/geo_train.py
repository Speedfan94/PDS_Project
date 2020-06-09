from nextbike import io
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier


def train_classification_dummy(p_X_train_scaled, p_y_train, p_weather):
    """Trains simple dummy classification model with X and y train data

    Args:
        p_X_train_scaled:           scaled X train data
        p_y_train:                  y train data
        p_weather:                  file ending when weather data is included
    Returns:
        dummy_classification_sets   dummy classification sets
    """
    X_train, X_val, y_train, y_val = train_test_split(p_X_train_scaled, p_y_train, random_state=42, test_size=0.2 / 0.7)
    dc = DummyClassifier(strategy="most_frequent")
    dc.fit(X_train, y_train)
    y_prediction_train = dc.predict(X_train)
    y_prediction_val = dc.predict(X_val)
    io.save_object(dc, "Dummy_classifier_model"+p_weather+".pkl")
    dummy_classification_sets = [y_train, y_val, y_prediction_train, y_prediction_val]
    return dummy_classification_sets


def train_classification_k_neighbors(p_X_train_scaled, p_y_train, p_weather):
    """Trains k neighbors classification model with X and y train data

    Args:
        p_X_train_scaled:           scaled X train data
        p_y_train:                  y train data
        p_weather:                  file ending when weather data is included
    Returns:
        kn_classification_sets      k neighbors classification sets
    """
    X_train, X_val, y_train, y_val = train_test_split(p_X_train_scaled, p_y_train, random_state=42, test_size=0.2 / 0.7)
    clf = KNeighborsClassifier(n_neighbors=10, weights="distance")
    clf.fit(X_train, y_train)
    y_prediction_train = clf.predict(X_train)
    y_prediction_val = clf.predict(X_val)
    io.save_object(clf, "KNearestNeighbours_classifier_model"+p_weather+".pkl")
    kn_classification_sets = [y_train, y_val, y_prediction_train, y_prediction_val]
    return kn_classification_sets


def train_classification_decision_tree(p_X_train_scaled, p_y_train, p_weather):
    """Trains decision tree classification model with X and y train data

    Args:
        p_X_train_scaled:           scaled X train data
        p_y_train:                  y train data
        p_weather:                  file ending when weather data is included
    Returns:
        dt_classification_sets      decision tree classification sets
    """
    X_train, X_val, y_train, y_val = train_test_split(p_X_train_scaled, p_y_train, random_state=42, test_size=0.2 / 0.7)
    dt = DecisionTreeClassifier(max_depth=10)
    dt.fit(X_train, y_train)
    y_prediction_train = dt.predict(X_train)
    y_prediction_val = dt.predict(X_val)
    io.save_object(dt, "DecisionTree_classifier_model"+p_weather+".pkl")
    dt_classification_sets = [y_train, y_val, y_prediction_train, y_prediction_val]
    return dt_classification_sets


def train_classification_random_forest(p_X_train_scaled, p_y_train, p_weather):
    """Trains random forest classification model with X and y train data

    Args:
        p_X_train_scaled:           scaled X train data
        p_y_train:                  y train data
        p_weather:                  file ending when weather data is included
    Returns:
        rf_classification_sets      random forest classification sets
    """
    X_train, X_val, y_train, y_val = train_test_split(p_X_train_scaled, p_y_train, random_state=42, test_size=0.2 / 0.7)
    rf = RandomForestClassifier(max_depth=10, n_estimators=15, max_features=3)
    rf.fit(X_train, y_train)
    y_prediction_train = rf.predict(X_train)
    y_prediction_val = rf.predict(X_val)
    io.save_object(rf, "RandomForest_classifier_model"+p_weather+".pkl")
    rf_classification_sets = [y_train, y_val, y_prediction_train, y_prediction_val]
    return rf_classification_sets


def train_classification_neural_network(p_X_train_scaled, p_y_train, p_weather):
    """Trains neural network classification model with X and y train data

    Args:
        p_X_train_scaled:           scaled X train data
        p_y_train:                  y train data
        p_weather:                  file ending when weather data is included
    Returns:
        nn_classification_sets      neural network classification sets
    """
    X_train, X_val, y_train, y_val = train_test_split(p_X_train_scaled, p_y_train, random_state=42, test_size=0.2 / 0.7)
    nn = MLPClassifier(alpha=0.0001, max_iter=1000)
    nn.fit(X_train, y_train)
    y_prediction_train = nn.predict(X_train)
    y_prediction_val = nn.predict(X_val)
    io.save_object(nn, "NeuralNetwork_classifier_model"+p_weather+".pkl")
    nn_classification_sets = [y_train, y_val, y_prediction_train, y_prediction_val]
    return nn_classification_sets

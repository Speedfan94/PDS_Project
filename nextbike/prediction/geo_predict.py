from nextbike import io


def predict_by_dummy_classificaion(p_X_test, p_weather):
    # TODO: Docstring
    # Read pickle objects
    scaler = io.read_object("Standard_Scaler_Direction"+p_weather+".pkl")
    pca = io.read_object("PCA_Direction"+p_weather+".pkl")
    model = io.read_object("Dummy_classifier_model"+p_weather+".pkl")
    # Use trained pickle objects
    X_test_scaled = scaler.transform(p_X_test)
    X_test_transformed = pca.transform(X_test_scaled)
    y_predictions = model.predict(X_test_transformed)
    return y_predictions


def predict_by_k_neighbors_classificaion(p_X_test, p_weather):
    # TODO: Docstring
    # Read pickle objects
    scaler = io.read_object("Standard_Scaler_Direction"+p_weather+".pkl")
    pca = io.read_object("PCA_Direction"+p_weather+".pkl")
    model = io.read_object("KNearestNeighbours_classifier_model"+p_weather+".pkl")
    # Use trained pickle objects
    X_test_scaled = scaler.transform(p_X_test)
    X_test_transformed = pca.transform(X_test_scaled)
    y_predictions = model.predict(X_test_transformed)
    return y_predictions


def predict_by_decision_tree_classificaion(p_X_test, p_weather):
    # TODO: Docstring
    # Read pickle objects
    scaler = io.read_object("Standard_Scaler_Direction"+p_weather+".pkl")
    pca = io.read_object("PCA_Direction"+p_weather+".pkl")
    model = io.read_object("DecisionTree_classifier_model"+p_weather+".pkl")
    # Use trained pickle objects
    X_test_scaled = scaler.transform(p_X_test)
    X_test_transformed = pca.transform(X_test_scaled)
    y_predictions = model.predict(X_test_transformed)
    return y_predictions


def predict_by_random_forest_classificaion(p_X_test, p_weather):
    # TODO: Docstring
    # Read pickle objects
    scaler = io.read_object("Standard_Scaler_Direction"+p_weather+".pkl")
    pca = io.read_object("PCA_Direction"+p_weather+".pkl")
    model = io.read_object("RandomForest_classifier_model"+p_weather+".pkl")
    # Use trained pickle objects
    X_test_scaled = scaler.transform(p_X_test)
    X_test_transformed = pca.transform(X_test_scaled)
    y_predictions = model.predict(X_test_transformed)
    return y_predictions


def predict_by_neural_network_classificaion(p_X_test, p_weather):
    # TODO: Docstring
    # Read pickle objects
    scaler = io.read_object("Standard_Scaler_Direction"+p_weather+".pkl")
    pca = io.read_object("PCA_Direction"+p_weather+".pkl")
    model = io.read_object("NeuralNetwork_classifier_model"+p_weather+".pkl")
    # Use trained pickle objects
    X_test_scaled = scaler.transform(p_X_test)
    X_test_transformed = pca.transform(X_test_scaled)
    y_predictions = model.predict(X_test_transformed)
    return y_predictions

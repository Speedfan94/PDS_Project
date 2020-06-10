from nextbike import io, prediction
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def filter_subsets(p_df):
    """Filters data on different months and iterates over each month
    to test with subset classification.

    Args:
        p_df:   Whole data set
    Returns:
        No return
    """
    months = p_df["Month_start"].unique()
    months.sort()
    for month in months:
        print(
            "====================================\n"+"      "
            "==========="+str(month)+"===========\n"+
            "===================================="
        )
        df_subset = p_df[p_df["Month_start"] == month]
        test_subset_classification(df_subset)


def test_subset_classification(p_df_subset, p_weather):
    """For each month, tries to classify direction with
    dummy classifiers and k nearest neighbor classifiers.

    Args:
        p_df_subset:    subset of the data set to classify on
        p_weather:      file ending when weather data is included
    Returns:
        No return
    """
    X = p_df_subset.drop("Direction", axis=1)
    y = p_df_subset["Direction"]
    print("Scale Data...")
    scaler = StandardScaler()
    X_scaled_train = scaler.fit_transform(X)
    print("Do PCA...")
    pca = PCA(n_components=15)
    pca.fit(X_scaled_train)
    X_train_transformed = pca.transform(X_scaled_train)
    # Read Models
    dummy = io.read_object("Dummy_classifier_model"+p_weather+".pkl")
    clf = io.read_object("KNearestNeighbours_classifier_model"+p_weather+".pkl")
    # Predict
    print("Predict...")
    dummy_y_prediction = dummy.predict(X_train_transformed)
    kn_y_prediction = clf.predict(X_train_transformed)
    # Evaluate Predict
    prediction.evaluate.direction_error_metrics(y, dummy_y_prediction, "Dummy_Classifier"+p_weather, "Testing")
    prediction.evaluate.direction_error_metrics(y, kn_y_prediction, "KNeighbors_Classifier"+p_weather, "Testing")


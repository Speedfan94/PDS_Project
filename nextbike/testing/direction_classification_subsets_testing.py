from nextbike import io, prediction
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def filter_subsets(p_df):
    # TODO: Docstring
    # TODO: What happens if month start is dropped from features?
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
    # TODO: Docstring
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
    prediction.evaluate.direction_error_metrics(y, dummy_y_prediction, "Dummy_Classifier", "Testing")
    prediction.evaluate.direction_error_metrics(y, kn_y_prediction, "KNeighbors_Classifier", "Testing")


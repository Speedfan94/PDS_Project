from .. import io
from .. import prediction
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier


def filter_subsets(p_df):
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


def test_subset_classification(p_df_subset):

    print("Split Data...")
    X_train, X_test, y_train, y_test = prediction.split.simple_split_direction(p_df_subset)
    print("Scale Data...")
    scaler = StandardScaler()
    X_scaled_train = scaler.fit_transform(X_train)
    print("Do PCA...")
    pca = PCA(n_components=15)
    pca.fit(X_scaled_train)
    X_train_transformed = pca.transform(X_scaled_train)
    # Train
    print("Train Dummy Classifier...")
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train_transformed, y_train)
    dummy_y_prediction = dummy.predict(X_train_transformed)
    print("Train KNeighbors Classifier...")
    clf = KNeighborsClassifier(n_neighbors=20, weights="distance")
    clf.fit(X_train_transformed, y_train)
    kn_y_prediction = clf.predict(X_train_transformed)
    # Evaluate Training
    # prediction.evaluate.direction_error_metrics(y_train, dummy_y_prediction, "Dummy_Classifier")
    # prediction.evaluate.direction_error_metrics(y_train, kn_y_prediction, "KNeighbors_Classifier")
    # Prepare
    X_test_scaled = scaler.transform(X_test)
    X_test_transformed = pca.transform(X_test_scaled)
    # Predict
    print("Predict")
    dummy_y_prediction = dummy.predict(X_test_transformed)
    kn_y_prediction = clf.predict(X_test_transformed)
    # Evaluate Predict
    prediction.evaluate.direction_error_metrics(y_test, dummy_y_prediction, "Dummy_Classifier", "Testing")
    prediction.evaluate.direction_error_metrics(y_test, kn_y_prediction, "KNeighbors_Classifier", "Testing")


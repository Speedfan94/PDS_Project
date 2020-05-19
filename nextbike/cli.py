import click
from . import io
from . import datapreparation
from . import visualization
from . import prediction


@click.command()
@click.option('--train/--no-train', default=True, help="Train the model.")
@click.option('--clean/--no-clean', default=True, help="Clean the data.")
@click.option('--viso/--no-viso', default=True, help="Visualize the data.")
@click.option('--show/--no-show', default=False, help="Show the dataframe.")
def main(train, clean, viso, show):
    if clean:
        print("Do cleaning")
        cleaning()
    if viso:
        print("Do visualizing")
        visualize()
    if train:
        print("Do training")
        predict()
    if show:
        print("Do show")
        print("Read in trips file...")
        df = io.read_trips()
        df.drop("Unnamed: 0", axis=1, inplace=True)
        print(df.info())
        print(df.head())


def cleaning():
    print("Read in nuremberg file...")
    df = io.read_file()
    df_trips = datapreparation.data_preparation(df)
    df_trips_add_feat = datapreparation.additional_feature_creation(df_trips)
    df_trips_filter_duration = datapreparation.drop_noise(df_trips_add_feat)
    df_trips_onlynuremberg = datapreparation.only_nuremberg_plz(df_trips_filter_duration)
    datapreparation.get_aggregate_statistics(df_trips_onlynuremberg)
    print("Save trip dataframe...")
    io.saveTrip(df_trips_onlynuremberg)


def visualize():
    print("Read in trips file...")
    df = io.read_trips()
    print("DONE reading in trips file")

    print("Visualize specific moment and heatmap...")
    visualization.visualize_moment(df)
    visualization.visualize_heatmap(df)
    visualization.visualize_plz(df)
    print("DONE visualizing")


def predict():
    print("Read in trips file...")
    df_trips = io.read_trips()
    df_trips.drop(["Unnamed: 0", "Place_start", "Place_end"], axis=1, inplace=True)
    print("Create dummie variables...")
    df = prediction.create_dummies(df_trips)
    print("Cast datetime...")
    df_fin = prediction.cast_datetime(df)
    print("Analyse correlations...")
    prediction.corr_analysis(df)
    # ___________________________________________________________________________________
    print("Split data in train and test set...")
    X_train, X_test, y_train, y_test = prediction.simple_split(df_fin)
    print("Scale data...")
    X_scaled_train = prediction.scale(X_train)
    print("Do PCA...")
    X_train_transformed = prediction.do_pca(X_scaled_train)
    # ____________________________________________________________________________________
    print("Train regression...")
    prediction.train_linear_regression(X_train_transformed, y_train)
    print("Predict with regression")
    prediction.predict_by_regression(X_test, y_test)


def test(df):
    time = 0
    a = df[df["End Time"] <= time]
    # - 0 places
    b=a.sort_by(by=["place_id", "datetime_end"])
    c = b.drop_duplicates(["place_id"], keep="last")
    print(c["p_bikes_end"])




if __name__ == '__main__':
    main()

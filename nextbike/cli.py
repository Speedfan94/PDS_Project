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
    df_trips_onlynuremberg = datapreparation.only_nuremberg_plz(df_trips)
    df_final = datapreparation.additional_feature_creation(df_trips_onlynuremberg)
    datapreparation.get_aggregate_statistics(df_final)

    print("Save trip dataframe...")
    io.saveTrip(df_final)


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
    df_trips.drop("Unnamed: 0", axis=1, inplace=True)
    X_train, X_test, y_train, y_test = prediction.simple_split(df_trips)
    #prediction.train(X_train, y_train)


if __name__ == '__main__':
    main()

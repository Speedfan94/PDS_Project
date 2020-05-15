import click
from . import io
from . import model
from . import datapreparation
from . import visualization
from . import prediction


@click.command()
@click.option('--train/--no-train', default=False, help="Train the model.")
@click.option('--clean/--no-clean', default=False, help="Clean the data.")
@click.option('--viso/--no-viso', default=False, help="Visualize the data.")
@click.option('--show/--no-show', default=False, help="Show the dataframe.")
def main(train, clean, viso, show):
    if clean:
        print("Do cleaning")
        cleaning()
    elif viso:
        print("Do visualizing")
        visualize()
    elif train:
        print("Do training")
        predict()
    elif show:
        print("Do show")
        print("Read in trips file...")
        df = io.read_trips()
        df.drop("Unnamed: 0", axis=1, inplace=True)
        print(df.info())
        print(df.head())

    else:
        print("Do all")
        cleaning()
        visualize()


def cleaning():
    print("Read in nuremberg file...")
    df = io.read_file()
    df_trips = datapreparation.datapreparation(df)
    df_trips_onlynuremberg = datapreparation.onlynuremberg(df_trips)
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
    print("DONE visualizing")


def predict():
    print("Read in trips file...")
    df = io.read_trips()
    df.drop("Unnamed: 0", axis=1, inplace=True)
    prediction.k_fold_split(df)


if __name__ == '__main__':
    main()

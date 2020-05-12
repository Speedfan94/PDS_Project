import click
from . import io
from . import model
from . import datapreparation
from . import visualization


@click.command()
@click.option('--train/--no-train', default=False, help="Train the model.")
@click.option('--clean/--no-clean', default=False, help="Clean the data.")
@click.option('--viso/--no-viso', default=False, help="Visualize the data.")
def main(train, clean, viso):
    if clean:
        print("Do cleaning")
        cleaning()
    elif viso:
        print("Do visualizing")
        visualize()
    elif train:
        print("Do training")
        model.train()
    else:
        print("Do all")
        cleaning()
        visualize()


def cleaning():
    print("Read in nuremberg file...")
    df = io.read_file()

    df_trips = datapreparation.datapreparation(df)

    df_trips_onlynuremberg = datapreparation.onlynuremberg(df_trips)
    #print(df_trips.head())

    print("Save trip dataframe...")
    io.saveTrip(df_trips_onlynuremberg)


def visualize():
    print("Read in trips file...")
    df = io.read_trips()

    visualization.visualize_moment(df)
    visualization.visualize_heatmap(df)


if __name__ == '__main__':
    main()

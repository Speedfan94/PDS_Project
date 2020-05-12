import click
from . import io
from . import model
from . import datapreparation
from . import visualization


@click.command()
@click.option('--train/--no-train', default=False, help="Train the model.")
def main(train):
    cleaning()

    # visualize()

    if train:
        model.train()
    else:
        print("You don't do anything.")


def cleaning():
    print("Read in nuremberg file...")
    df = io.read_file()

    df_trips = datapreparation.datapreparation(df)

    df_trips_onlynuremberg = datapreparation.onlynuremberg_plz(df_trips)
    print(df_trips_onlynuremberg)

    print("Save trip dataframe...")
    io.saveTrip(df_trips_onlynuremberg)


def visualize():
    print("Read in trips file...")
    df = io.read_trips()

    visualization.visualize_moment(df)
    visualization.visualize_heatmap(df)


if __name__ == '__main__':
    main()

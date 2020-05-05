import click
from . import io
from . import model
from . import datapreparation
from . import visualization


@click.command()
@click.option('--train/--no-train', default=False, help="Train the model.")
def main(train):
    cleaning()

    df = io.read_trips()
    visualize(df)





    if train:
        model.train()
    else:
        print("You don't do anything.")


def cleaning():
    # read in nuremberg file
    print("Read in nuremberg file...")
    df = io.read_file()

    df_trips = datapreparation.datapreparation(df)

    df_trips_onlynuremberg = datapreparation.onlynuremberg(df_trips)

    print("Save trip dataframe...")
    io.saveTrip(df_trips)

def visualize(df):
    visualization.visualize_moment(df)

if __name__ == '__main__':
    main()


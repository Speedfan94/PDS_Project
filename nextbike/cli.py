import click
from . import io
from . import model
from . import datapreparation


@click.command()
@click.option('--train/--no-train', default=False, help="Train the model.")
def main(train):

    # read in nuremberg file
    print("Reading in nuremberg file...")
    df = io.read_file()
    print("Done!")
    print(df)

    df_trips = datapreparation.datapreparation(df)

    print("Dropping trips outside Nuremberg")
    df_trips_onlynuremberg = datapreparation.onlynuremberg(df_trips)
    # print(df_trips_onlynuremberg)

    aggr_stats = datapreparation.get_aggregate_statistics(df_trips_onlynuremberg['Duration'])
    print("======== Aggregate Statistics ========")
    print("Mean:               ", aggr_stats['mean'])
    print("Standard deviation: ", aggr_stats['std'])
    print("======================================")

    print("Saving trip dataframe")
    io.saveTrip(df_trips)


    if train:
        model.train()
    else:
        print("You don't do anything.")


if __name__ == '__main__':
    main()


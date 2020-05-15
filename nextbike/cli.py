import click
from . import io
from . import model
from . import datapreparation
from . import visualization


@click.command()
@click.option('--train/--no-train', default=False, help="Train the model.")
def main(train):
    print("Start cleaning process...")
    cleaning()
    print("DONE cleaning")

    print("Start visualizing process...")
    visualize()
    print("DONE visualizing")

    if train:
        model.train()
    else:
        print("You don't do any training.")


def cleaning():
    print("Read in nuremberg file...")
    df = io.read_file()
    print("DONE reading file")

    print("Build trips dataframe...")
    df_trips = datapreparation.datapreparation(df)
    print("DONE building trips dataframe")

    print("Kick out entries not from nuremberg...")
    df_trips_onlynuremberg = datapreparation.onlynuremberg(df_trips)
    print("DONE kicking out non-nuremberg entries")

    print("Add additional features...")
    df_with_additional_columns = datapreparation.additional_feature_creation(df_trips_onlynuremberg)
    print("DONE adding additional features")

    print("Calculate aggregation statistics...")
    aggr_stats = datapreparation.get_aggregate_statistics(df_with_additional_columns)
    print()
    print("============ Aggregate Statistics ============")
    print("Mean:                      ", aggr_stats['mean'])
    print("Std. deviation:            ", aggr_stats['std'])
    print("Mean           (weekends): ", aggr_stats['mean_weekends'])
    print("Std. deviation (weekends): ", aggr_stats['std_weekends'])
    print("Mean           (weekdays): ", aggr_stats['mean_weekdays'])
    print("Std. deviation (weekdays): ", aggr_stats['std_weekdays'])
    print("==============================================")
    print()

    print("Save trip dataframe as csv...")
    io.saveTrip(df_with_additional_columns)
    print("DONE saving as csv")


def visualize():
    print("Read in trips file...")
    df = io.read_trips()
    print("DONE reading in trips file")

    print("Visualize specific moment and heatmap...")
    visualization.visualize_moment(df)
    visualization.visualize_heatmap(df)
    print("DONE visualizing")


if __name__ == '__main__':
    main()

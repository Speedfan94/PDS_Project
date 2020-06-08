import click
from datetime import datetime
from nextbike import visualization, cli_code, utils


# TODO: Usefull help strings
@click.group("test-group")
@click.option("--test/--no-test", default=False, help="Testing mode") # TODO: add parameter option for different tests
@click.option("--clean/--no-clean", default=False, help="Clean the data.")
@click.option("--viso/--no-viso", default=False, help="Visualize the data.")
@click.option("--train/--no-train", default=False, help="Train duration models.")
@click.option("--pred/--no-pred", default=False, help="Predict duration with models.")
@click.option("--traingeo/--no-traingeo", default=False, help="Train direction models.")
@click.option("--predgeo/--no-predgeo", default=False, help="Predict direction with models.")
@click.option("--weather/--no-weather", default=False, help="Decide, whether to include weather data or not.")
def main(test, clean, viso, train, pred, traingeo, predgeo, weather):
    # TODO: Docstring // Only for testing
    if test:
        # testing_duration_models()
        # testing_robust_scaler()
        # testing_direction_subsets()
        visualization.main_test()
    else:
        start_time = datetime.now().replace(microsecond=0)
        start_time_step = start_time

        if clean:
            print("START CLEAN")
            cli_code.cleaning()
            start_time_step = utils.print_time_for_step("STEP CLEAN", start_time_step)
        if viso:
            print("START VISUALIZE")
            # TODO Rename visualization.math / geo to math_plot and geo_plot
            cli_code.visualize()
            start_time_step = utils.print_time_for_step("STEP VISUALIZATION", start_time_step)
        if train:
            print("START TRAIN")
            cli_code.features_duration(weather)
            cli_code.training_duration_models()
            start_time_step = utils.print_time_for_step("STEP TRAIN", start_time_step)
        if pred:
            print("START PREDICT")
            cli_code.predict_duration_models(weather)
            start_time_step = utils.print_time_for_step("STEP PREDICT", start_time_step)
        if traingeo:
            print("START GEO TRAIN")
            cli_code.features_direction(weather)
            cli_code.train_direction_models()
            start_time_step = utils.print_time_for_step("STEP GEO TRAIN", start_time_step)
        if predgeo:
            print("START GEO PREDICT")
            cli_code.predict_direction_models(weather)
            start_time_step = utils.print_time_for_step("STEP GEO PREDICT", start_time_step)

        # TODO: can we use start_time_step instead of datetime.now().replace(microsecond=0) ???
        # print("TIME FOR RUN:", (datetime.now().replace(microsecond=0) - start_time))


# TODO: Help
@main.command(name="train", help="Trains a ML model based on nuremberg.csv in data/input")
@click.option("--regress/--no-regress", default=True, help="")
@click.option("--classify/--no-classify", default=False, help="")
@click.option("--weather/--no-weather", default=False, help="Activate to include weather data."
                                                            "Be sure to insert weather data for given time period"
                                                            "into 'data/input' directory first by yourself.")
def train(regress, classify, weather):
    # TODO: Docstring
    start_time_train = datetime.now().replace(microsecond=0)
    start_time_step = start_time_train
    if regress:
        print("START TRAIN")
        cli_code.features_duration(weather)
        cli_code.training_duration_models()
        start_time_step = utils.print_time_for_step("STEP TRAIN", start_time_step)

    if classify:
        print("START GEO TRAIN")
        cli_code.features_direction(weather)
        cli_code.train_direction_models()
        utils.print_time_for_step("STEP GEO TRAIN", start_time_step)
    utils.print_time_for_step("COMMAND TRAIN", start_time_train)


@main.command(name="predict", help="Loads a previously trained model and predicts journey duration and direction"
                                   "based on start information given in provided file."
                                   "File has to be inserted into 'data/input' directory")
@click.option("--regress/--no-regress", default=True, help="Determines whether duration should be predicted. Default: True")
@click.option("--classify/--no-classify", default=False, help="Determines whether direction should be predicted, too."
                                                                "Default: False.")
@click.option("--weather/--no-weather", default=False, help="Activate to include weather data."
                                                            "Be sure to insert weather data for given time period"
                                                            "into 'data/input' directory first by yourself.")
@click.argument("filename", default="nuremberg.csv")
def predict(filename, regress, classify, weather):
    # TODO: Docstring
    start_time_predict = datetime.now().replace(microsecond=0)
    start_time_step = start_time_predict
    if regress:
        print("START PREDICT")
        cli_code.predict_duration_models(filename, weather)
        start_time_step = utils.print_time_for_step("STEP PREDICT", start_time_step)
    if classify:
        print("START GEO PREDICT")
        cli_code.predict_direction_models(filename, weather)
        utils.print_time_for_step("STEP GEO PREDICT", start_time_step)
    utils.print_time_for_step("COMMAND PREDICT", start_time_predict)


@main.command(name="transform", help="Transforms a data set in the provided format to a data set in trip format"
                                     "and saves it as 'Trips.csv' into 'data/output' directory."
                                     "New data set has to be inserted into 'data/input' directory.")
@click.argument("filename", default="nuremberg.csv")
def transform(filename):
    # TODO: Docstring
    start_time_transform = datetime.now().replace(microsecond=0)
    print("START CLEAN")
    cli_code.cleaning(filename)
    utils.print_time_for_step("COMMAND TRANSFORM", start_time_transform)


# TODO: Help
@main.command(name="descriptive_analysis", help="")
@click.argument("filename", default="nuremberg.csv")
@click.option("--clean/--no-clean", default=True, help="")
def descriptive_analysis(filename, clean):
    # TODO: docstring
    start_time_desc_analysis = datetime.now().replace(microsecond=0)
    start_time_step = start_time_desc_analysis
    if clean:
        print("START CLEAN")
        cli_code.cleaning(filename)
        start_time_step = utils.print_time_for_step("STEP CLEAN", start_time_step)
    print("START VISUALIZATION")
    cli_code.visualize()
    utils.print_time_for_step("STEP VISUALIZATION", start_time_step)
    utils.print_time_for_step("COMMAND DESCRIPTIVE ANALYSIS", start_time_desc_analysis)


if __name__ == "__main__":
    main()

import click
from datetime import datetime
from nextbike import cli_code, utils, io, visualization


@click.group(invoke_without_command=True)
@click.option("--test/--no-test", default=False,
              help="Activate to test some alternative algorithms and their performance. Default: False")
@click.option("--clean/--no-clean", default=False,
              help="Clean the data, create trips dataset and save it as Trips.csv. Default: False")
@click.option("--viso/--no-viso", default=False,
              help="Do descriptive visualizations and plot aggregate statistics and different moments in time."
                   "Default: False")
@click.option("--traindur/--no-traindur", default=False,
              help="Train ML models to later on predict trip duration. Save them as pkl files to use again."
                   "Default: False")
@click.option("--pred/--no-pred", default=False,
              help="Predict the trip duration based on given start data by using the built ML models. Default: False")
@click.option("--traingeo/--no-traingeo", default=False,
              help="Train ML models to later on predict direction of trips."
                   "Save them as pkl files to use again. Default: False")
@click.option("--predgeo/--no-predgeo", default=False,
              help="Predict the direction of trips (towards or away from university)"
                   "based on given start data by using the built ML models. Default: False")
@click.option("--weather/--no-weather", default=False,
              help="Decide, whether to include weather data or not."
                   "Beware that weather data has to be manually added to 'data/input' directory"
                   "for the given time period by the user before running this. Default: False")
def main(test, clean, viso, traindur, pred, traingeo, predgeo, weather):
    """Initial entry point when running 'nextbike' in console.
    Runs everytime, nextbike is called. Therefore all steps are set off on default.

    This method should only be used to test specific stages/steps of our pipeline.

    Args:
        test:       option whether to run test stage (Default: False)
        clean:      option whether to run clean stage (Default: False)
        viso:       option whether to run viso stage (Default: False)
        traindur:      option whether to run train stage (Default: False)
        pred:       option whether to run pred stage (Default: False)
        traingeo:   option whether to run traingeo stage (Default: False)
        predgeo:    option whether to run predgeo stage (Default: False)
        weather:    option whether to include weather data (Default: False)
    Returns:
        No return
    """
    if not (test | clean | viso | traindur | pred | traingeo | predgeo):
        # No option activated, nothing to do in main
        return

    print("=== START COMMAND: MAIN")
    print()
    start_time_main = datetime.now().replace(microsecond=0)
    start_time_step = start_time_main

    # initialize file endings if weather or custom csv file is used
    weather_data = ""
    if weather:
        weather_data = "_weather"

    if test:
        print("=== START STEP: TEST")
        print("Testing alternative models/scaler")
        df = io.read_csv("Trips.csv", "output")
        # cli_code.testing_duration_models(p_weather=weather_data)
        # cli_code.testing_robust_scaler(p_weather=weather_data)
        # cli_code.testing_direction_subsets(p_weather=weather_data)
        visualization.visualize_trips_per_month(p_df=df)
        visualization.visualize_stations_moment(p_df=df)
        visualization.visualize_postalcode(p_df=df)
        utils.print_time_for_step(p_step_name="STEP TEST", p_start_time_step=start_time_step)
    else:
        if clean:
            print("=== START STEP: CLEAN")
            cli_code.cleaning()
            start_time_step = utils.print_time_for_step(p_step_name="STEP CLEAN", p_start_time_step=start_time_step)
        if viso:
            print("=== START STEP: VISUALIZATION")
            cli_code.visualize()
            start_time_step = utils.print_time_for_step(
                p_step_name="STEP VISUALIZATION", p_start_time_step=start_time_step)
        if traindur:
            print("=== START STEP: TRAIN DURATION")
            cli_code.features_duration(p_weather=weather_data)
            cli_code.train_all_duration_models(p_weather=weather_data)
            start_time_step = utils.print_time_for_step(p_step_name="STEP TRAIN DURATION", p_start_time_step=start_time_step)
        if pred:
            print("=== START STEP: PREDICT DURATION")
            cli_code.predict_by_all_duration_models(p_weather=weather_data)
            start_time_step = utils.print_time_for_step(p_step_name="STEP PREDICT DURATION", p_start_time_step=start_time_step)
        if traingeo:
            print("=== START STEP: TRAIN DIRECTION")
            cli_code.features_direction(p_weather=weather_data)
            cli_code.train_all_direction_models(weather_data)
            start_time_step = utils.print_time_for_step(p_step_name="STEP TRAIN DIRECTION", p_start_time_step=start_time_step)
        if predgeo:
            print("=== START STEP: PREDICT DIRECTION")
            cli_code.predict_by_all_direction_models(p_weather=weather_data)
            utils.print_time_for_step(p_step_name="STEP PREDICT DIRECTION", p_start_time_step=start_time_step)

    utils.print_time_for_step("COMMAND MAIN", start_time_main)


@main.command(name="train", help="Train ML models based on nuremberg.csv and save them as pkl files in 'data/output'."
                                 "BEWARE: This method will overwrite the existing models!!!")
@click.option("--regress/--no-regress", default=True,
              help="Deactivate to skip training regression models used for trip duration prediction. Default: True.")
@click.option("--classify/--no-classify", default=True,
              help="Activate to also train classification models used for direction prediction. Default: False")
@click.option("--weather/--no-weather", default=False,
              help="Activate to include weather data. Be sure to insert weather data for given time period"
                   "into 'data/input' directory first by yourself.")
def train(regress, classify, weather):
    """Train ML models based on nuremberg.csv and save them as pkl files in 'data/output'.
    BEWARE: This method will overwrite the existing models!!!

    Args:
        regress:    option whether to train regression models for duration prediction (Default: True)
        classify:   option whether to train classification models for direction prediction (Default: False)
        weather:    option whether to include weather data (Default: False)
    Returns:
        No return
    """
    print("=== START COMMAND: TRAIN")
    print()
    start_time_train = datetime.now().replace(microsecond=0)
    start_time_step = start_time_train

    # initialize file endings if weather or custom csv file is used
    weather_data = ""
    if weather:
        weather_data = "_weather"

    print("=== START STEP: CLEAN")
    cli_code.cleaning(p_filename="nuremberg.csv")
    start_time_step = utils.print_time_for_step(p_step_name="STEP CLEAN", p_start_time_step=start_time_step)

    if regress:
        print("=== START STEP: TRAIN DURATION")
        cli_code.features_duration(p_weather=weather_data)
        cli_code.train_best_regression_model(p_weather=weather_data)
        start_time_step = utils.print_time_for_step(p_step_name="STEP TRAIN DURATION", p_start_time_step=start_time_step)

    if classify:
        print("=== START STEP: TRAIN DIRECTION")
        cli_code.features_direction(p_weather=weather_data)
        cli_code.train_best_classification_model(p_weather=weather_data)
        utils.print_time_for_step(p_step_name="STEP TRAIN DIRECTION", p_start_time_step=start_time_step)

    utils.print_time_for_step(p_step_name="COMMAND TRAIN", p_start_time_step=start_time_train)


@main.command(name="predict", help="Loads a previously trained model and predicts journey duration and direction"
                                   "based on start information given in provided file."
                                   "File has to be inserted into 'data/input' directory")
@click.option("--regress/--no-regress", default=True,
              help="Deactivate to skip training regression models used for trip duration prediction. Default: True.")
@click.option("--classify/--no-classify", default=True,
              help="Activate to also train classification models used for direction prediction. Default: False")
@click.option("--weather/--no-weather", default=False,
              help="Activate to include weather data. Be sure to insert weather data for given time period"
                   "into 'data/input' directory first by yourself.")
@click.argument("filename", default="nuremberg.csv")
def predict(filename, regress, classify, weather):
    """Loads a previously trained model and predicts journey duration and direction
    based on start information given in provided file. File has to be inserted into 'data/input' directory.

    Args:
        filename:   filename of new data csv file. File has to be located in 'data/input'
        regress:    option whether to predict by regression model for duration regression (Default: True)
        classify:   option whether to predict by classification model for direction classification (Default: False)
        weather:    option whether to include weather data (Default: False)
    Returns:
        No return
    """
    print("=== START COMMAND: PREDICT")
    print()
    start_time_predict = datetime.now().replace(microsecond=0)
    start_time_step = start_time_predict

    # initialize file endings if weather or custom csv file is used
    run_on_original_dataset = ""
    if filename != "nuremberg.csv":
        run_on_original_dataset = "_testing"
    weather_data = ""
    if weather:
        weather_data = "_weather"

    print("=== START STEP: CLEAN")
    cli_code.cleaning(p_filename=filename, p_mode=run_on_original_dataset)
    start_time_step = utils.print_time_for_step(p_step_name="STEP CLEAN", p_start_time_step=start_time_step)

    if regress:
        print("=== START STEP: PREDICT DURATION")
        cli_code.features_duration(p_weather=weather_data, p_mode=run_on_original_dataset)
        cli_code.predict_by_best_regression_model(p_weather=weather_data, p_mode=run_on_original_dataset)
        start_time_step = utils.print_time_for_step(p_step_name="STEP PREDICT DURATION", p_start_time_step=start_time_step)

    if classify:
        print("=== START STEP: PREDICT DIRECTION")
        cli_code.features_direction(p_weather=weather_data, p_mode=run_on_original_dataset)
        cli_code.predict_by_best_classification_model(p_weather=weather_data, p_mode=run_on_original_dataset)
        utils.print_time_for_step(p_step_name="STEP PREDICT DIRECTION", p_start_time_step=start_time_step)

    utils.print_time_for_step(p_step_name="COMMAND PREDICT", p_start_time_step=start_time_predict)


@main.command(name="transform", help="Transforms a data set in the provided format to a data set in trip format"
                                     "and saves it as 'Trips.csv' into 'data/output' directory."
                                     "New data set has to be inserted into 'data/input' directory.")
@click.argument("filename", default="nuremberg.csv")
def transform(filename):
    """Transforms a data set in the provided format to a data set in trip format
    and saves it as 'Trips.csv' into 'data/output' directory. New data set has to
    be inserted into 'data/input' directory.

    Args:
        filename:   filename of new data csv file. File has to be located in 'data/input'
    Returns:
        No return
    """
    print("=== START COMMAND: TRANSFORM")
    print()
    start_time_transform = datetime.now().replace(microsecond=0)
    start_time_step = start_time_transform

    # initialize file endings if weather or custom csv file is used
    run_on_original_dataset = ""
    if filename != "nuremberg.csv":
        run_on_original_dataset = "_testing"

    print("=== START STEP: CLEAN")
    cli_code.cleaning(p_filename=filename, p_mode=run_on_original_dataset)
    utils.print_time_for_step(p_step_name="STEP CLEAN", p_start_time_step=start_time_step)

    utils.print_time_for_step(p_step_name="COMMAND TRANSFORM", p_start_time_step=start_time_transform)


@main.command(name="descriptive_analysis",
              help="Start a descriptive analysis on the given data set. Clean and transform data into trip data."
                   "Afterwards start plotting descriptive statistics and visualizations.")
@click.argument("filename", default="nuremberg.csv")
def descriptive_analysis(filename):
    """Start a descriptive analysis on the given data set. Clean and transform data into trip data.
    Afterwards start plotting descriptive statistics and visualizations.

    Args:
        filename:   filename of new data csv file. File has to be located in 'data/input'
    Returns:
        No return
    """
    print("=== START COMMAND: DESCRIPTIVE ANALYSIS")
    print()
    start_time_desc_analysis = datetime.now().replace(microsecond=0)
    start_time_step = start_time_desc_analysis

    # initialize file endings if weather or custom csv file is used
    run_on_original_dataset = ""
    if filename != "nuremberg.csv":
        run_on_original_dataset = "_testing"

    print("=== START STEP: CLEAN")
    cli_code.cleaning(p_filename=filename, p_mode=run_on_original_dataset)
    start_time_step = utils.print_time_for_step(p_step_name="STEP CLEAN", p_start_time_step=start_time_step)

    print("=== START STEP: VISUALIZATION")
    cli_code.visualize(p_mode=run_on_original_dataset)
    utils.print_time_for_step(p_step_name="STEP VISUALIZATION", p_start_time_step=start_time_step)

    utils.print_time_for_step(p_step_name="COMMAND DESCRIPTIVE ANALYSIS", p_start_time_step=start_time_desc_analysis)


if __name__ == "__main__":
    main()

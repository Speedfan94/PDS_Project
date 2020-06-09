import click
from datetime import datetime
from nextbike import cli_code, utils, testing, io, visualization


# TODO: OPTIONAL: add parameter option for different tests
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
    weather_data = ""
    if weather:
        weather_data = "_weather"
    if test:
        df = io.read_csv("Trips.csv", "output")
        # df = io.read_csv("Features_Duration.csv", "output")
        # testing_duration_models()
        # testing.robust_scaler_testing.test_robust_scaler(p_df=df, p_weather)
        # testing_direction_subsets()
        visualization.visualize_trips_per_month(p_df=df)
        visualization.visualize_stations_moment(p_df=df)
        visualization.visualize_postalcode(p_df=df)
        print("This can be used for testing-purposes")
    else:
        start_time_main = datetime.now().replace(microsecond=0)
        start_time_step = start_time_main

        if clean:
            print("START CLEAN")
            cli_code.cleaning()
            start_time_step = utils.print_time_for_step(p_step_name="STEP CLEAN", p_start_time_step=start_time_step)
        if viso:
            print("START VISUALIZE")
            cli_code.visualize()
            start_time_step = utils.print_time_for_step(
                p_step_name="STEP VISUALIZATION", p_start_time_step=start_time_step)
        if traindur:
            print("START TRAIN")
            cli_code.features_duration(p_weather=weather_data)
            cli_code.training_duration_models(p_weather=weather_data)
            start_time_step = utils.print_time_for_step(p_step_name="STEP TRAIN", p_start_time_step=start_time_step)
        if pred:
            print("START PREDICT")
            cli_code.predict_duration_models(p_weather=weather_data)
            start_time_step = utils.print_time_for_step(p_step_name="STEP PREDICT", p_start_time_step=start_time_step)
        if traingeo:
            print("START GEO TRAIN")
            cli_code.features_direction(p_weather=weather_data)
            cli_code.train_direction_models(weather_data)
            start_time_step = utils.print_time_for_step(p_step_name="STEP GEO TRAIN", p_start_time_step=start_time_step)
        if predgeo:
            print("START GEO PREDICT")
            cli_code.predict_direction_models(p_weather=weather_data)
            utils.print_time_for_step(p_step_name="STEP GEO PREDICT", p_start_time_step=start_time_step)

        # As we in general do not run any of these main command stages, the print might confuse users
        # utils.print_time_for_step("COMMAND MAIN", start_time_main)


@main.command(name="train", help="Train ML models based on nuremberg.csv and save them as pkl files in 'data/output'.")
@click.option("--regress/--no-regress", default=True,
              help="Deactivate to skip training regression models used for trip duration prediction. Default: True.")
@click.option("--classify/--no-classify", default=False,
              help="Activate to also train classification models used for direction prediction. Default: False")
@click.option("--weather/--no-weather", default=False,
              help="Activate to include weather data. Be sure to insert weather data for given time period"
                   "into 'data/input' directory first by yourself.")
def train(regress, classify, weather):
    """Train ML models based on nuremberg.csv and save them as pkl files in 'data/output'

    Args:
        regress:    option whether to train regression models for duration prediction (Default: True)
        classify:   option whether to train classification models for direction prediction (Default: False)
        weather:    option whether to include weather data (Default: False)
    Returns:
        No return
    """
    start_time_train = datetime.now().replace(microsecond=0)
    start_time_step = start_time_train
    weather_data = ""
    print("START CLEAN")
    cli_code.cleaning(p_filename="nuremberg.csv")
    if weather:
        weather_data = "_weather"
    if regress:
        print("START TRAIN")
        cli_code.features_duration(p_weather=weather_data, p_mode="_testing")
        cli_code.train_best_regression_model(p_weather=weather_data, p_mode="_testing")
        start_time_step = utils.print_time_for_step(p_step_name="STEP TRAIN", p_start_time_step=start_time_step)
    if classify:
        print("START GEO TRAIN")
        cli_code.features_direction(p_weather=weather_data, p_mode="_testing")
        cli_code.train_best_classification_model(p_weather=weather_data, p_mode="_testing")
        utils.print_time_for_step(p_step_name="STEP GEO TRAIN", p_start_time_step=start_time_step)
    utils.print_time_for_step(p_step_name="COMMAND TRAIN", p_start_time_step=start_time_train)


@main.command(name="predict", help="Loads a previously trained model and predicts journey duration and direction"
                                   "based on start information given in provided file."
                                   "File has to be inserted into 'data/input' directory")
@click.option("--regress/--no-regress", default=True,
              help="Deactivate to skip training regression models used for trip duration prediction. Default: True.")
@click.option("--classify/--no-classify", default=False,
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
    start_time_predict = datetime.now().replace(microsecond=0)
    start_time_step = start_time_predict
    weather_data = ""
    if weather:
        weather_data = "_weather"
    print("START CLEAN")
    cli_code.cleaning(p_filename=filename, p_mode="_testing")
    if regress:
        print("START PREDICT DURATION")
        cli_code.features_duration(p_weather=weather_data, p_mode="_testing")
        cli_code.pred_by_best_reression_model(p_weather=weather_data)
        start_time_step = utils.print_time_for_step(p_step_name="STEP PREDICT", p_start_time_step=start_time_step)
    if classify:
        print("START GEO PREDICT")
        cli_code.features_direction(p_weather=weather_data, p_mode="_testing")
        cli_code.pred_by_best_classification_model(p_weather=weather_data)
        utils.print_time_for_step(p_step_name="STEP GEO PREDICT", p_start_time_step=start_time_step)
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

    start_time_transform = datetime.now().replace(microsecond=0)
    print("START CLEAN")
    cli_code.cleaning(p_filename=filename, p_mode="_testing")
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
    start_time_desc_analysis = datetime.now().replace(microsecond=0)
    start_time_step = start_time_desc_analysis
    print("START CLEAN")
    cli_code.cleaning(p_filename=filename, p_mode="_testing")
    start_time_step = utils.print_time_for_step(p_step_name="STEP CLEAN", p_start_time_step=start_time_step)
    print("START VISUALIZATION")
    cli_code.visualize(p_mode="_testing")
    utils.print_time_for_step(p_step_name="STEP VISUALIZATION", p_start_time_step=start_time_step)
    utils.print_time_for_step(p_step_name="COMMAND DESCRIPTIVE ANALYSIS", p_start_time_step=start_time_desc_analysis)


if __name__ == "__main__":
    main()

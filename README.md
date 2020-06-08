# PDS_Project
This package implements a command line interface.

## 1. Installation Guide
Within the same directory as ```setup.py``` run the following commands in your cmd / terminal:

1. ```conda install rtree```
2. ```conda install geopandas```
3. ```conda install tensorflow```
4. ```pip install .```
5. Copy initial csv file of original dataset into ```data/input``` directory

Great! You are now ready to use our nextbike analytics & prediction package!

### 1.1. Additional Installation Notes
- When in subdirectory ```nextbike``` or ```notebooks```, run ```pip install ..``` in step 4
- Use flag ```-e``` on ```pip install``` to install in development mode
- Accept installation processes by typing ```y``` when asked

## 2. Usage
This package runs through different steps:
- cleaning
- viasualization
- training
- prediction

General terminal command to run the project:

```nextbike [OPTIONS] [COMMAND] [ARGS] [CMD-OPTIONS]```

### 2.1 Options

| Option | Name | Description |
|--------|------|-------------|
| ```nextbike --test``` | algorithm testing | Tests some alternative algorithms and their performance |
| ```nextbike --clean``` | data cleaning | Prepares trips data, saves them as "Trips.csv" |
| ```nextbike --viso``` | visualization | Plots graphs & interactive maps in "data/output" |
| ```nextbike --train``` | training | Trains models for duration prediction using different algorithms, analyzing & scaling features and components |
| ```nextbike --pred``` | prediction | Predicts trip duration with different algorithms, measures prediction performance |
| ```nextbike --traingeo``` | geo training | Trains models for direction prediction using different algorithms, analyzing & scaling features and components |
| ```nextbike --predgeo``` | geo prediction | Predicts trip direction with different algorithms, measures prediction performance |
| ```nextbike --weather``` | weather data | Include weather data into training and prediction, data has to be fetched manually before |

By default, all these options are deactivated.

To run one or multiple of these steps, add the corresponding option when running nextbike (e.g.: ```nextbike --clean --viso```).

### 2.2 Commands

| Command | Options | Description |
|---------|---------|-------------|
| ```nextbike transform new_data.csv``` |  | Cleans the given data set & creates trip data. File has to be located inside ```data/input``` directory. If no file is given, runs with initial ```nuremberg.csv``` dataset. |
| ```nextbike descriptive_analysis new_data.csv``` | ```--no-clean``` | Runs a descriptive analysis, including data cleaning and visualization. Able to skip cleaning with option. |
| ```nextbike train``` | ```--no-regress``` ```--classify``` | Trains models for prediction. By default, only trains regression model for trip duration prediction. With options, activate classification model for direction prediction, too. |
| ```nextbike predict new_data.csv``` | ```--no-regress``` ```--classify``` | Runs predictions based on built models. By default, only runs regression prediction for duration. With option, also direction prediction is possible. |

Referenced csv files have to be already copied into ```data/input``` directory before running these commands!
Make sure these files exist.

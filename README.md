# PDS_Project
This package implements a command line interface.

## 1. Installation Guide
Within the same directory as ```setup.py``` run the following commands in your cmd / terminal:

1. ```conda install rtree```
2. ```conda install geopandas```
3. ```conda install tensorflow```
4. ```pip install .```

#### 1.1. Additional Installation Notes
- When in subdirectory ```nextbike``` or ```notebooks```, run ```pip install ..``` in step 4
- Use flag ```-e``` on ```pip install``` to install in development mode
- Accept installation processes by typing ```y``` when asked

## 2. Usage
This package has different stages:

| Step | Name | Description |
|------|------|-------------|
| ```clean``` | data cleaning | prepares trips data, saves them as "Trips.csv" |
| ```viso``` | visualization | plots graphs & interactive maps in "data/output" |
| ```train``` | training | trains models for different algorithms, analyzes & scales features and components |
| ```pred``` | prediction | tries to predict trip duration and direction with different algorithms, measures prediction performance |

By default, all these steps are activated. So to run the project, simply call ```nextbike```.

To skip one or multiple step(s) add ```--no-[STEP]``` option. (e.g.: ```nextbike --no-clean```)

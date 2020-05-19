from .utils import get_data_path
import pandas as pd
import os
import pickle


def read_file(path=os.path.join(get_data_path(), "input/nuremberg.csv")):
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        print("Data file not found. Path was " + path)


def read_object(pFilename):
    path = os.path.join(get_data_path(), "output", "models", pFilename)
    with open(path, "rb") as f:
        my_object = pickle.load(f)
    return my_object


def read_trips():
    path = os.path.join(get_data_path(), "output/Trips.csv")
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        print("Data file not found. Path was " + path)



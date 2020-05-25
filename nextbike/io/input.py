from .utils import get_data_path
import pandas as pd
import os
import pickle


# TODO: Add docstring
def read_csv(p_filename, p_io_folder, p_sub_folder=""):
    path = os.path.join(get_data_path(), p_io_folder, p_sub_folder, p_filename)
    try:

        df = pd.read_csv(path)
        print("Read:", path)
        return df
    except FileNotFoundError:
        print("Data file not found. Path was " + path)


# TODO: Add docstring
def read_object(p_filename):
    path = os.path.join(get_data_path(), "output", "models", p_filename)
    with open(path, "rb") as f:
        my_object = pickle.load(f)
        print("Read:", path)
    return my_object

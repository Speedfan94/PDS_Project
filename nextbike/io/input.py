from .utils import get_data_path
import pandas as pd
import os
import pickle


def read_csv(p_filename, p_io_folder, p_sub_folder=""):
    """Read a csv file under given path which is created from params.

    e.g. Used for nuremberg.csv, trips.csv, features.csv
    Return the file or an error.
    Args:
        p_filename (str): String of the filename of csv
        p_io_folder (str): String that differentiates between input and output folder
        p_sub_folder (str): String that contains the subfolder after input or output folder => if not given none
    Returns:
        df (DataFrame): DataFrame which is created from the read csv
    """
    path = os.path.join(get_data_path(), p_io_folder, p_sub_folder, p_filename)
    try:
        # index_col=0 throws numpy warning,
        # so we set it manually after reading in the file
        df = pd.read_csv(path)
        df.set_index(df.columns[0], inplace=True)
        print("Read:", path)
        return df
    except FileNotFoundError:
        print("Data file not found. Path was " + path)


def read_object(p_filename):
    """Read a pickle file which contains an object and returns it.

    e.g. Used for ML models, PCA, Scaler
    Return the file in data/output/models/*p_filename.pkl* or an error.
    Args:
        p_filename (str): String of the filename of pickle
    Returns:
        my_object (Object): Object which is read by pickle
    """
    path = os.path.join(get_data_path(), "output", "models", p_filename)
    with open(path, "rb") as f:
        my_object = pickle.load(f)
        print("Read:", path)
    return my_object

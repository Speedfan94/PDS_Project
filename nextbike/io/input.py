from .utils import get_data_path
import pandas as pd
import os
import pickle


def read_file(pFilename, pIo_folder, pSub_folder=""):
    path = os.path.join(get_data_path(), pIo_folder, pSub_folder, pFilename)
    try:
        print("Read", path)
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        print("Data file not found. Path was " + path)


def read_object(pFilename):
    path = os.path.join(get_data_path(), "output", "models", pFilename)
    with open(path, "rb") as f:
        my_object = pickle.load(f)
    return my_object

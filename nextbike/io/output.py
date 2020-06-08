from .utils import get_data_path
import os
import pickle


def save_object(p_object, p_filename):
    """Save an object as a pickle file

    Saves the given object under the given filename under data/output/models/*p_filename*.pkl
    e.g. Used for ML models, PCA, Scaler

    Args:
        p_object (Object): Object which should be saved as pickle file
        p_filename (str): String of the filename of pickle file
    Returns:
        No return
    """
    path = os.path.join(get_data_path(), "output", "models", p_filename)
    pickle.dump(p_object, open(path, "wb"))
    print("Saved:", path)


def save_csv(p_df, p_filename, p_subfolder=""):
    """Save an dataframe as a csv file

    Saves the given dataframe under the given filename under data/output/*p_filename*.csv
    e.g. Used for trips.csv, features.csv

    Args:
        p_df (DataFrame): Dataframe which should be saved as csv file
        p_filename (str): String of the filename of csv file
        p_subfolder (str): String with name of subfolder to save in (optional)
    Returns:
        No return
    """
    path = os.path.join(get_data_path(), "output", p_subfolder, p_filename)
    p_df_reindexed = p_df.reset_index(drop=True)
    p_df_reindexed.to_csv(path, index_label="index")
    print("Saved:", path)


def save_fig(p_fig, p_filename, p_io_folder="output", p_sub_folder1="data_plots", p_sub_folder2=""):
    """Save an figure as a png file

    Saves the given figure under the given filename under
    data/*p_io_folder*/*p_sub_folder1*/*p_sub_folder2*/*p_filename*.png
    Where the standard path to save is data/output/data_plots/*p_filename*.png
    e.g. Used for geographicall visualizations and mathematical visualizations

    Args:
        p_fig (Figure): Figure which should be saved as png file
        p_filename (str): String of the filename of png file
        p_io_folder (str): String which is input or output
        p_sub_folder1 (str): String of first subfolder
        p_sub_folder2 (str): String of second subfolder
    Returns:
        No return
    """
    path = os.path.join(get_data_path(), p_io_folder, p_sub_folder1, p_sub_folder2, p_filename)
    p_fig.savefig(path)
    print("Saved:", path)

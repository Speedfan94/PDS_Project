from .utils import get_data_path
import os
import pickle


# TODO: Add docstring
def save_object(p_object, p_filename):
    path = os.path.join(get_data_path(), "output", "models", p_filename)
    pickle.dump(p_object, open(path, 'wb'))
    print("Saved:", path)


# TODO: Add docstring
def save_csv(p_df, p_filename):
    path = os.path.join(get_data_path(), "output", p_filename)
    p_df.to_csv(path)
    print("Saved:", path)
    print("______________________________________________________")


# TODO: Add docstring
def save_fig(p_fig, p_filename, p_io_folder="output", p_sub_folder1="data_plots", p_sub_folder2=""):
    path = os.path.join(get_data_path(), p_io_folder, p_sub_folder1, p_sub_folder2, p_filename)
    p_fig.savefig(path)
    print("Saved:", path)

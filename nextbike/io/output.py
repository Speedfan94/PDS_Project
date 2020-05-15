from .utils import get_data_path
import os
import pickle


def save_model(model):
    pickle.dump(model, open(os.path.join(get_data_path(), "output/model.pkl"), 'wb'))


def saveTrip(df):
    # Save final df
    df.to_csv(os.path.join(get_data_path(), "output/Trips.csv"))


def save_fig(fig, file_name):
    fig.savefig(os.path.join(get_data_path(), "output/", file_name))

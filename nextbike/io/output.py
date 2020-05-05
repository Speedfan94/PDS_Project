from .utils import get_data_path
import os
import pickle


def save_model(model):
    pickle.dump(model, open(os.path.join(get_data_path(), "output/model.pkl"), 'wb'))


def saveTrip(df):
    # Save final df
    print("Save csv")
    df.to_csv(os.path.join(get_data_path(), "output/Trips.csv"))
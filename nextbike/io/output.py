from .utils import get_data_path
import os
import pickle


def save_object(pObject, pFilename):
    pickle.dump(pObject, open(os.path.join(get_data_path(), "output", "models", pFilename), 'wb'))
    print("Saved", pFilename)


# TODO: No need for save_fig and saveTrip => Create just one method
def saveTrip(df):
    df.to_csv(os.path.join(get_data_path(), "output/Trips.csv"))
    print("Saved Trips.csv")


def save_fig(fig, file_name):
    fig.savefig(os.path.join(get_data_path(), "output/", file_name))
    print("Saved", file_name)

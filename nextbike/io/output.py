from .utils import get_data_path
import os
import pickle


def save_object(pObject, pFilename):
    pickle.dump(pObject, open(os.path.join(get_data_path(), "output", "models", pFilename), 'wb'))
    print("Saved", pFilename)


# TODO: No need for save_fig and saveTrip => Create just one method
def save_trip(pDf, pFilename):
    pDf.to_csv(os.path.join(get_data_path(), "output", pFilename))
    print("Saved Trips.csv")


def save_fig(pFig, pFile_name, pIo_folder="output", pSub_folder="data_plots"):
    pFig.savefig(os.path.join(get_data_path(), pIo_folder, pSub_folder, pFile_name))
    print("Saved", pFile_name)

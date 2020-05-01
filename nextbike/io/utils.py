import os


def get_data_path():

    if os.path.isdir(os.path.join(os.getcwd(), 'data')):
        return os.path.join(os.getcwd(), 'data')
    elif os.path.isdir(os.path.join(os.getcwd(), "../data")):
        return os.path.join(os.getcwd(), "../data")
    elif os.path.isdir(os.path.join(os.getcwd(), "nextbike/data")):
        return os.path.join(os.getcwd(), "nextbike/data")
    else:
        raise FileNotFoundError

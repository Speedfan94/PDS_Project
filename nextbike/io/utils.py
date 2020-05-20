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


def get_path(filename, io_folder, subfolder=""):
    """ get you the path for given parameters

    Args:
        filename (str): name of file
        io_folder (str): input or output folder
        subfolder (str): name of subfolder
    Returns:
        path (os.path)
    """
    path = os.path.join(get_data_path(), io_folder, subfolder, filename)
    return path

import os


# TODO: Add docstring
def get_data_path():

    if os.path.isdir(os.path.join(os.getcwd(), 'data')):
        return os.path.join(os.getcwd(), 'data')
    elif os.path.isdir(os.path.join(os.getcwd(), "../data")):
        return os.path.join(os.getcwd(), "../data")
    elif os.path.isdir(os.path.join(os.getcwd(), "nextbike/data")):
        return os.path.join(os.getcwd(), "nextbike/data")
    else:
        raise FileNotFoundError


def get_path(p_filename, p_io_folder, p_sub_folder1="", p_sub_folder2=""):
    """ Return the path for given parameters

    Args:
        p_filename (str): name of file
        p_io_folder (str): input or output folder
        p_sub_folder1 (str): name of subfolder
        p_sub_folder2 (str): name of second subfolder
    Returns:
        path (os.path): Path for given parameters
    """
    path = os.path.join(get_data_path(), p_io_folder, p_sub_folder1, p_sub_folder2, p_filename)
    return path

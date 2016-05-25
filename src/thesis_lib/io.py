import os
import pickle as pk

PATH_TO_SAVE = "../data/"

def save_object(object_to_save, name_file, overwrite=False):
    '''
    Saves a specific object with a given name.
    If the file exist, it will be overwritten if overwrite is True.
    
    Parameter
    ---------
    object_to_save: object to save
    name_file: name of the file. Automatically add the .pk extension, if not already present.
    overwrite:
    
    References
    ----------
    https://github.com/WillahScott/facial-keypoint-detection/blob/master/scripts/tools/save4later.py
    '''
    # add extension if necessary
    if name_file[-2:] != ".pk":
        name_file += ".pk"
    
    file_path = PATH_TO_SAVE + name_file
    
    if os.path.isfile(file_path) and not overwrite:
        print("WARNING - file % exists. For overwriting specify overwrite=True.".format(file_path))
        return
    
    print(file_path)
    
    # Save object
    with open(file_path, 'wb') as f:
        pk.dump(object_to_save, f, protocol=pk.HIGHEST_PROTOCOL)

def load_object(name_file):
    """
    Load an object previously saved.
    
    Parameter
    ---------
    name_file: name of the file. Automatically add the .pk extension, if not already present.
    
    Return
    ------
    the loaded object
    """
    # add extension if necessary
    if name_file[-2:] != ".pk":
        name_file += ".pk"
    
    file_path = PATH_TO_SAVE + name_file
    
    print(file_path)
    
    try:
        with open(file_path, 'rb') as f:
            model = pk.load(f)
    
    except Exception as e:
        print(e)

    else:
        print("Object loaded from {}".format(file_path))
        return model

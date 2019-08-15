import json
from joblib import dump, load

def dict_to_json(dict, path):
    '''Serializes a dictionary as json and writes it to the specified file path.

    Args:
        dict (dict): The dictionary to serialize.
        path (string): The file path to write to.
    '''
    dict_json = json.dumps(dict)
    with open(path, 'w') as json_file:
        json_file.write(dict_json)

def save_model(value, filename):
    '''Saves a model using the specified filename.

    Args:
        value (object): The model to save.
        filename (string): The filename to save the model to.
    '''
    print('Saving {filename}.'.format(filename=filename))
    dump(value, filename)

def load_model(filename):
    '''Loads a model from the specified filename.

    Args:
        filename (string): The filename to load the model from.
    Returns:
        Object: The model.
    '''
    print('Loading {filename}'.format(filename=filename))
    return load(filename)
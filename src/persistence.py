from joblib import dump, load

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
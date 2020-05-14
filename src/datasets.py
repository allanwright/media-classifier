'''Collection of methods used for aquiring data for training and testing.

'''

import os

import numpy as np
import pandas as pd

def get_local_files(base_path):
    '''Gets a list of files in the specified directory and all sub-directories.

    Args:
        base_path (string): The base directory to search for files.

    Returns:
        list: The list of files.
    '''
    files = []
    for file in os.listdir(base_path):
        full_path = base_path + '//' + file
        if os.path.isdir(full_path):
            files.extend(get_local_files(full_path))
        else:
            files.append(file)
    return files

def write_list_to_file(items, path):
    '''Writes the contents of a list to the specified file path.

    Args:
        items (list): The list to write.
        path (string): The file to write to.
    '''
    with open(path, 'w', encoding='utf-8') as f:
        _ = [f.write('%s\n' % item) for item in items]

def get_train_test_data():
    '''Reads the processed and split train/test data.

    Returns:
        x_train (numpy array): The training features.
        y_train (nunmpy array): The training labels.
        x_eval (numpy array): The evaluation features.
        y_eval (numpy array): The evaluation labels.
        x_test (numpy array): The evaluation features.
        y_test (numpy array): The evaluation labels.
    '''
    return (
        get_processed_data('x_train.csv'),
        get_processed_data('y_train.csv'),
        get_processed_data('x_eval.csv'),
        get_processed_data('y_eval.csv'),
        get_processed_data('x_test.csv'),
        get_processed_data('y_test.csv')
    )


def get_processed_data(name):
    '''Reads the specified file and returns the contents as a flattened array.

    Args:
        name (string): The name (and path) of the file to read.
    '''
    df = pd.read_csv('data/processed/' + name, header=None)
    return np.ravel(df[0].to_numpy())

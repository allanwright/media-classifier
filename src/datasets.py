import http
import os
import numpy as np
import pandas as pd
import requests
import urllib.request
import time
from bs4 import BeautifulSoup

def get_kraken_data(path):
    '''Gets filenames from kraken and writes them to the raw data directory.

    Args:
        path (string): The base directory to search for files.
    '''
    paths = [
        ['movie', 'data/raw/movies/kraken.txt'],
        ['music', 'data/raw/music/kraken.txt'],
        ['tv', 'data/raw/tv/kraken.txt']
    ]

    for i in paths:
        write_list_to_file(get_local_files(path % i[0]), i[1])

def get_pig_data(path):
    '''Gets filenames from pig and writes them to the raw data directory.

    Args:
        path (string): The base directory to search for files.
    '''
    paths = [
        ['Movies', 'data/raw/movies/pig.txt'],
        ['Music', 'data/raw/music/pig.txt'],
        ['TV Shows', 'data/raw/tv/pig.txt']
    ]

    for i in paths:
        write_list_to_file(get_local_files(path % i[0]), i[1])

def get_xerus_data(url):
    '''Gets filenames from xerus and writes them to the raw data directory.

    Args:
        url (string): The base url to search for files.
    '''
    paths = [
        ['%s/Documentaries/%s/', 'data/raw/docos/xerus%s.txt'],
        ['%s/Music/%s/', 'data/raw/music/xerus%s.txt'],
        ['%s/TV/%s/', 'data/raw/tv/xerus%s.txt'],
        ['%s/Movies/%s/', 'data/raw/movies/xerus%s.txt'],
        ['%s/Apps/%s/', 'data/raw/apps/xerus%s.txt'],
        ['%s/Games/%s/', 'data/raw/games/xerus%s.txt']
    ]

    for x in paths:
        for y in range(1, 151):
            write_list_to_file(
                get_xerus_files(
                    url, x[0] % ('/cat', y)), x[1] % y)

def get_yak_data(url):
    '''Gets filenames from yak and writes them to the raw data directory.

    Args:
        url (string): The base url to search for files.
    '''
    site_y = [
        ['%s/Movies/date/%s/', 'data/raw/movies/yak%s.txt'],
        ['%s/Applications/date/%s/', 'data/raw/apps/yak%s.txt'],
        ['%s/Games/date/%s/', 'data/raw/games/yak%s.txt']
    ]

    for x in site_y:
        for y in range(1, 501):
            write_list_to_file(
                get_yak_files(
                    url, x[0] % ('/browse-torrents', y)), x[1] % y)

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

def get_xerus_files(base_url, search_path):
    '''Gets a list of files from a website codenamed xerus.

    Args:
        base_url (string): The base url of the website.
        search_path (string): The relative path to search for files.
    
    Returns:
        list: The list of files.
    '''
    files = []
    response = requests.get(base_url + search_path)
    soup = BeautifulSoup(response.text, 'html.parser')
    anchors = soup.select('td.name a:nth-of-type(2)')
    for a in anchors:
        print('Scraping: %s' % a['href'])
        item_response = requests.get('%s%s' % (base_url, a['href']))
        item_soup = BeautifulSoup(item_response.text, 'html.parser')
        list_items = item_soup.select('#files li')
        for li in list_items:
            files.append(li.text)
    return files

def get_yak_files(base_url, search_path):
    '''Gets a list of files from the website codenamed yak.

    Args:
        base_url (string): The base url of the website.
        search_path (string): The relative path to search for files.
    
    Returns:
        list: The list of files.
    '''
    files = []
    response = requests.get(base_url + search_path)
    soup = BeautifulSoup(response.text, 'html.parser')
    anchors = soup.select('.tt-name a:nth-of-type(2)')
    for a in anchors:
        while True:
            try:
                print('Scraping: {href}'.format(href=a['href']))
                item_response = requests.get('{url}{href}'
                    .format(url=base_url, href=a['href']))
                item_soup = BeautifulSoup(item_response.text, 'html.parser')
                list_items = item_soup.select('.fileline')
                for li in list_items:
                    for s in li.text.split('\xa0'):
                        if s:
                            files.append(s)
                break
            except http.client.RemoteDisconnected:
                print('Retrying: Remote host disconnected')
            except UnicodeEncodeError:
                print('Skipping: Url contains non ascii chars')
                break
    return files

def write_list_to_file(list, path):
    '''Writes the contents of a list to the specified file path.

    Args:
        list (list): The list to write.
        path (string): The file to write to.
    '''
    with open(path, 'w', encoding='utf-8') as f:
        [f.write('%s\n' % item) for item in list]

def get_train_test_data():
    '''Reads the processed and split train/test data.

    Returns:
        x_train (numpy array): The training features.
        y_train (nunmpy array): The training labels.
        x_test (numpy array): The test features.
        y_test (numpy array): The test labels.
    '''
    return (
        get_processed_data('x_train.csv'),
        get_processed_data('y_train.csv'),
        get_processed_data('x_test.csv'),
        get_processed_data('y_test.csv')
    )


def get_processed_data(name):
    df = pd.read_csv('data/processed/' + name, header=None)
    return np.ravel(df.to_numpy())
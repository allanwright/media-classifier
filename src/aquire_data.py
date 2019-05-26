import os
import pandas as pd
import requests
import urllib.request
import time
from bs4 import BeautifulSoup

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

def get_site_x_files(base_url, search_path):
    '''Gets a list of files from website x.

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
        print(a)
        item_response = requests.get('%s%s' % (base_url, a['href']))
        item_soup = BeautifulSoup(item_response.text, 'html.parser')
        list_items = item_soup.select('#files li')
        for li in list_items:
            files.append(li.text)
    return files

def write_list_to_file(list, path):
    '''Writes the contents of a list to the specified file path.

    Args:
        list (list): The list to write.
        path (string): The file to write to.
    '''
    with open(path, 'w', encoding='utf-8') as f:
        for item in list:
            f.write('%s\n' % item)

def get_raw_data():
    local = [
        ['\\\\kraken//movie', 'data/raw/movies/kraken.txt'],
        ['\\\\kraken//music', 'data/raw/music/kraken.txt'],
        ['\\\\kraken//tv', 'data/raw/tv/kraken.txt'],
        ['\\\\pig//Media//Movies', 'data/raw/movies/pigflix.txt'],
        ['\\\\pig//Media//Music', 'data/raw/music/pigflix.txt'],
        ['\\\\pig//Media//TV Shows', 'data/raw/tv/pigflix.txt']
    ]

    for i in local:
        write_list_to_file(get_local_files(i[0]), i[1])
    
    site_x = [        
        ['%s/Documentaries/seeders/desc/%s/', 'data/raw/docos/sitex%s.txt'],
        ['%s/Movies/seeders/desc/%s/', 'data/raw/movies/sitex%s.txt'],
        ['%s/Music/seeders/desc/%s/', 'data/raw/music/sitex%s.txt'],
        ['%s/TV/seeders/desc/%s/', 'data/raw/tv/sitex%s.txt'],
        ['%s/Apps/seeders/desc/%s/', 'data/raw/apps/sitex%s.txt'],
        ['%s/Games/seeders/desc/%s/', 'data/raw/games/sitex%s.txt']
    ]

    for x in site_x:
        for y in range(1, 51):
            write_list_to_file(
                get_site_x_files(
                    'https://site.x', x[0] % ('/sort-cat', y)), x[1] % y)

get_raw_data()
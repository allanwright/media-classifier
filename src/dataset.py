import http
import os
import pandas as pd
import requests
import urllib.request
import time
from bs4 import BeautifulSoup
from sklearn.utils import resample

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
        for item in list:
            f.write('%s\n' % item)

def get_consolidated_raw_data(path):
    '''Gets a pandas dataframe containing the contents of all raw data files.

    The name of the folder used to store each file is used for the category
    column in the resulting dataframe.

    Args:
        path (string): The path of the raw data.
    
    Returns:
        DataFrame: The contents of all raw data files.
    '''
    consolidated = pd.DataFrame()
    for x in os.listdir(path):
        x_path = '%s/%s' % (path, x)
        if os.path.isdir(x_path):
            for y in os.listdir(x_path):
                y_path = '%s/%s/%s' % (path, x, y)
                if os.path.isfile(y_path):
                    print('Processing %s' % y_path)
                    series = pd.read_csv(y_path, sep='\t', squeeze=True)
                    df = pd.DataFrame()
                    df['name'] = series
                    df['category'] = x
                    consolidated = consolidated.append(df, ignore_index=True)
    return consolidated

def process_data():
    '''Processes the raw data files.
    '''
    df = get_consolidated_raw_data('data/raw')

    # Remove commas from the name column
    df['name'] = df['name'].str.replace(',', '')

    # Remove file sizes from the end of filenames
    df['name'] = df['name'].str.replace(r'\s{1}\(.+\)$', '')
    df['name'] = df['name'].str.replace(r' - \S+\s{1}\S+$', '')

    #Merge categories
    df.loc[df['category'] == 'games', 'category'] = 'apps'

    # Create file extension column
    ext = df['name'].str.extract(r'\.(\w{3})$')
    ext.columns = ['ext']
    df['ext'] = ext['ext']

    # Remove file extension from filenames
    df['name'] = df['name'].str.replace(r'\.(\w{3})$', '')
    
    # Remove paths from filenames
    df['name'] = df['name'].str.split('/').str[-1]
    
    # Remove junk filenames
    music_ext = [
        'mp3',
        'm4a',
        'ogg',
        'flac',
        'wav'
    ]

    movie_ext = [
        'mp4',
        'mkv',
        'avi',
        'wmv',
        'mpg',
        'm4v'
    ]

    tv_ext = [
        'mp4',
        'mkv',
        'avi',
        'wmv',
        'mpg',
        'm4v'
    ]

    app_ext = [
        'exe',
        'bin',
        'zip',
        'rar',
        'iso',
        'cab',
        'dll',
        'msi',
        'dmg',
        'dat'
    ]

    df = df[((df['category'] == 'music') & (df['ext'].isin(music_ext))) |
            ((df['category'] == 'movies') & (df['ext'].isin(movie_ext))) |
            ((df['category'] == 'tv') & (df['ext'].isin(tv_ext))) |
            ((df['category'] == 'apps') & (df['ext'].isin(app_ext)))]
    
    # Remove duplicates by filename and category
    df.drop_duplicates(subset=['name', 'category'], inplace=True)

    # Normalize word separators
    df['name'] = df['name'].str.replace('.', ' ')
    df['name'] = df['name'].str.replace('_', ' ')
    df['name'] = df['name'].str.replace('-', ' ')
    df['name'] = df['name'].str.replace('[', ' ')
    df['name'] = df['name'].str.replace(']', ' ')
    df['name'] = df['name'].str.replace('+', ' ')
    df['name'] = df['name'].str.split().str.join(' ')

    # Remove rubbish characters
    df['name'] = df['name'].str.strip('`~!@#$%^&*()-_+=[]|;:<>,./?')

    # Save interim output before processing further
    df.to_csv('data/interim/combined.csv', index=False)

    # Downsample to fix class imbalance
    categories = [df[df.category == c] for c in df.category.unique()]
    sample_size = min([len(c) for c in categories])
    downsampled = [resample(c,
                            replace=False,
                            n_samples=sample_size,
                            random_state=123) for c in categories]
    df = pd.concat(downsampled)

    # Save interim output before processing further
    df.to_csv('data/interim/balanced.csv', index=False)
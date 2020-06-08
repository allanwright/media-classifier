'''Defines a pipeline step which aquires data from the yak data source.

'''

import datetime
import http
import os
import requests

from bs4 import BeautifulSoup

from src import datasets
from src.step import Step

class GetYakData(Step):
    '''Defines a pipeline step which aquires data from the yak data source.

    '''

    def __init__(self):
        '''Initializes a new instance of the GetYakData object.

        '''
        super(GetYakData, self).__init__()
        self.input = {
            'url': os.getenv('YAK_URL'),
            'start_page': 1,
            'end_page': 500,
        }
        self.output = {
            'movie': ['%s/Movies/date/%s/', 'data/raw/movie/yak%s.txt'],
            'app': ['%s/Applications/date/%s/', 'data/raw/app/yak%s.txt'],
            'game': ['%s/Games/date/%s/', 'data/raw/game/yak%s.txt'],
        }

    def run(self):
        '''Runs the pipeline step.

        '''
        url = self.input['url']
        start_page = self.input['start_page']
        end_page = self.input['end_page']

        for x in self.output.values():
            for y in range(start_page, end_page + 1):
                now = datetime.datetime.today().strftime('%Y%m%dT%H%M%S')
                name = x[1] % f'{now}_{y}'
                datasets.write_list_to_file(
                    self.__get_yak_files(
                        url, x[0] % ('/browse-torrents', y)), name)

    def __get_yak_files(self, base_url, search_path):
        files = []
        response = requests.get(base_url + search_path)
        soup = BeautifulSoup(response.text, 'html.parser')
        anchors = soup.select('.tt-name a:nth-of-type(2)')
        for anchor in anchors:
            while True:
                try:
                    print('Scraping: {href}'.format(href=anchor['href']))
                    item_response = requests.get(
                        '{url}{href}'.format(url=base_url, href=anchor['href']))
                    item_soup = BeautifulSoup(item_response.text, 'html.parser')
                    list_items = item_soup.select('.fileline')
                    for item in list_items:
                        for substring in item.text.split('\xa0'):
                            if substring:
                                files.append(substring)
                    break
                except http.client.RemoteDisconnected:
                    print('Retrying: Remote host disconnected')
                except UnicodeEncodeError:
                    print('Skipping: Url contains non ascii chars')
                    break
        return files

'''Defines a pipeline step which aquires data from the xerus data source.

'''

import os
import requests

from bs4 import BeautifulSoup

from src import datasets
from src.step import Step

class GetXerusData(Step):
    '''Defines a pipeline step which aquires data from the xerus data source.

    '''

    def __init__(self):
        '''Initializes a new instance of the GetXerusData object.

        '''
        super(GetXerusData, self).__init__()
        self.input = {
            'url': os.getenv('XERUS_URL'),
            'start_page': 1,
            'end_page': 150,
        }
        self.output = {
            'doco': ['%s/Documentaries/%s/', 'data/raw/doco/xerus%s.txt'],
            'music': ['%s/Music/%s/', 'data/raw/music/xerus%s.txt'],
            'tv': ['%s/TV/%s/', 'data/raw/tv/xerus%s.txt'],
            'movie': ['%s/Movies/%s/', 'data/raw/movie/xerus%s.txt'],
            'app': ['%s/Apps/%s/', 'data/raw/app/xerus%s.txt'],
            'game': ['%s/Games/%s/', 'data/raw/game/xerus%s.txt']
        }

    def run(self):
        '''Runs the pipeline step.

        '''
        url = self.input['url']
        start_page = self.input['start_page']
        end_page = self.input['end_page']
        for x in self.output.values():
            for y in range(start_page, end_page + 1):
                datasets.write_list_to_file(
                    self.__get_xerus_files(
                        url, x[0] % ('/cat', y)), x[1] % y)

    def __get_xerus_files(self, base_url, search_path):
        files = []
        response = requests.get(base_url + search_path)
        soup = BeautifulSoup(response.text, 'html.parser')
        anchors = soup.select('td.name a:nth-of-type(2)')
        for anchor in anchors:
            print('Scraping: %s' % anchor['href'])
            item_response = requests.get('%s%s' % (base_url, anchor['href']))
            item_soup = BeautifulSoup(item_response.text, 'html.parser')
            list_items = item_soup.select('#files li')
            for item in list_items:
                files.append(item.text)
        return files

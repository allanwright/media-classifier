'''Defines a pipeline step which aquires test data.

'''

import os

from azure.storage.queue import QueueClient

from src import classifier
from src import ner
from src.step import Step

class GetTestData(Step):
    '''Defines a pipeline step which aquires test data.

    '''

    def __init__(self):
        '''Initializes a new instance of the GetTestData object.

        '''
        super(GetTestData, self).__init__()
        self.input = {}
        self.output = {
            'classifier': 'data/test/classifier.csv',
            'ner': 'data/test/ner.csv',
        }

    def run(self):
        '''Runs the pipeline step.

        '''
        queue = QueueClient(
            account_url=os.getenv('AZ_QS_AC_URL'),
            queue_name=os.getenv('AZ_QS_QUEUE_NAME'),
            credential=os.getenv('AZ_QS_SAS_TOKEN'))
        response = queue.receive_messages(messages_per_page=5)
        for batch in response.by_page():
            for message in batch:
                filename = message.content
                label, _ = classifier.predict(filename)
                entities = ner.predict(filename)
                self.print(
                    '\'{filename}\' classified as \'{label}\'',
                    filename=filename,
                    label=label['name'])
                self.print(
                    '\'{filename}\' has entities \'{entities}\'',
                    filename=filename,
                    entities=entities)
                with open(self.output['classifier'], 'a') as f:
                    f.write(message.content + ',' + str(label['id']) + '\n')
                with open(self.output['ner'], 'a') as f:
                    f.write(message.content + ',' + str(entities) + '\n')
                queue.delete_message(message)

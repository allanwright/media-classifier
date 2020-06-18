'''Defines a pipeline step which aquires test data.

'''

import os

from azure.storage.queue import QueueClient

from src import classifier
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
            'test': 'data/test/test.csv',
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
                self.print(
                    '\'{filename}\' classified as \'{label}\'',
                    filename=filename,
                    label=label['name'])
                with open(self.output['test'], 'a') as f:
                    f.write(message.content + ',' + str(label['id']) + '\n')
                queue.delete_message(message)

'''Defines a pipeline step which aquires prediction data.

'''

import os

from azure.storage.queue import QueueClient

from src.step import Step

class GetPredictionData(Step):
    '''Defines a pipeline step which aquires prediction data.

    '''

    def __init__(self):
        '''Initializes a new instance of the GetPredictionData object.

        '''
        super(GetPredictionData, self).__init__()
        self.input = {}
        self.output = {
            'predictions': 'data/predictions/predictions.csv',
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
                print(message.content)
                with open(self.output['predictions'], 'a') as f:
                    f.write(message.content + '\n')
                queue.delete_message(message)

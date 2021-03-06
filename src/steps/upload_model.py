'''Defines a pipeline step which uploads a model to the production environment.

'''

import os

from azure.core.exceptions import ResourceNotFoundError
from azure.storage.blob import BlobServiceClient

from src.step import Step

class UploadModel(Step):
    '''Defines a pipeline step which uploads a model to the production environment.

    '''

    def run(self):
        '''Runs the pipeline step.

        '''
        model_type = self.input['--model']
        model_id = self.input['--id']
        path = './models/{model}/{id}'.format(model=model_type, id=model_id)

        if not os.path.exists(path):
            self.print('Path \'{path}\' could not be found.', path=path)
            return

        files = [f for f in os.listdir(path) if f.endswith('.json') or f.endswith('.pickle')]

        if not files:
            self.print('Path \'{path}\' is empty.', path=path)
            return

        connection_string = os.getenv(f'AZ_ST_{model_type.upper()}_CNN')
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_name = 'models'
        container_client = blob_service_client.get_container_client(container_name)

        try:
            container_client.get_container_properties()
        except ResourceNotFoundError:
            container_client.create_container()

        for file in files:
            blob_client = blob_service_client.get_blob_client(container_name, file)
            with open(path + '/' + file, 'rb') as data:
                blob_client.upload_blob(data, overwrite=True)
                self.print(
                    'Uploaded blob \'{file}\' to container \'{container_name}\'.',
                    file=file,
                    container_name=container_name)

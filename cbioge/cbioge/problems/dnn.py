import os
import numpy as np

from cbioge.utils.model import TimedStopping

from cbioge.utils import checkpoint as ckpt

class ModelRunner:
    ''' ModelRunner is focused on running keras models while keeping
        info from the model stored.

        It saves: loss, accuracy, #of params and history from the training.

        # Parameters
        model: keras model object
        path: path to save the model after training (optional)
        verbose: verbosity mode (optional)
    '''

    def __init__(self, model, path=None, verbose=False):
        self.model = model

        self.loss = 1
        self.accuracy = 0
        self.params = model.count_params() if model is not None else 0
        self.history = None
        
        self.verbose = verbose
        self.ckpt_path = ckpt.ckpt_folder if path is None else path

        if not os.path.exists(self.ckpt_path):
            os.makedirs(self.ckpt_path)

    def train_model(self, x_train, y_train, batch_size, epochs, **kwargs):
        ''' executes the training of a model.

            # Parameters
            x_train: training data
            y_train: training labels
            batch_size: size of the batches used during training
            epochs: number of epochs the training will be executed

            # Optional parameters (have to be passed with keyword)
            validation_data: Data on which to evaluate the loss and any model 
            metrics at the end of each epoch. Expects:
            - tuple (x_val, y_val)
            timelimit: max time (in seconds the model will be trained)
        '''

        callbacks = []

        if 'timelimit' in kwargs and kwargs['timelimit'] is not None:
            ts = TimedStopping(
                seconds=kwargs['timelimit'], 
                verbose=self.verbose)
            callbacks.append(ts)

        validation_data = None
        if ('validation_data' in kwargs 
            and kwargs['validation_data'] is tuple 
            and len(kwargs['validation_data']) == 2):
                validation_data = kwargs['validation_data']

        self.history = self.model.fit(x_train, y_train, 
            validation_data=validation_data, 
            batch_size=batch_size, 
            epochs=epochs, 
            verbose=self.verbose, 
            callbacks=callbacks)

        # TODO executar de acordo com a politica
        model_path = os.path.join(self.ckpt_path, f'weights.hdf5')
        self.model.save_weights(model_path)

    def test_model(self, x_test, y_test, batch_size, weights_path=None):

        if weights_path is not None:
            self.model.load_weights(weights_path)

        self.loss, self.accuracy = self.model.evaluate(
            x_test, y_test, 
            batch_size=batch_size, 
            verbose=self.verbose)

    def predict_model(self, x_test, batch_size):

        predictions = self.model.predict(x_test, 
            batch_size=batch_size, 
            verbose=self.verbose)

        pred_path = os.path.join(self.ckpt_path, 'predictions.npy')
        np.save(pred_path, predictions)

        return predictions        
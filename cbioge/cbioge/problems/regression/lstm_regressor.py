import sys
from typing import Union
import numpy as np
from keras import backend as K
from keras.layers import Embedding, Dense
import keras
import datetime as dt
from cbioge.algorithms.solution import Solution

from ..dnns import layers as clayers
from ...datasets import Dataset
from ...grammars import Grammar
from .. import DNNProblem
from keras.models import model_from_json

class LSTMRegressorProblem(DNNProblem):
    ''' Problem class for problems related to classification tasks for DNNs.
        This class includes methods focused on the design of CNNs.'''

    def __init__(self, parser: Grammar, dataset: Dataset,
        batch_size: int=32,
        epochs: int=1,
        opt: str='adam',
        loss: Union[str, callable]='mean_squared_error',
        metrics: list=['mse'],
        test_eval: bool=False,
        verbose: bool=False,
        train_args: dict={},
        test_args: dict={}
    ):

        super().__init__(parser, dataset, batch_size, epochs, opt, loss,
            metrics, test_eval, verbose, train_args, test_args)

    
    def _build_model(self, mapping: list):
        
        reshaped_mapping = self._reshape_mapping(mapping)
        layers = []

        # input layer
        layers.append(Embedding(input_dim=self.dataset.input_shape[0], output_dim=300))

        for block in reshaped_mapping:
            b_name, values = block[0], block[1:]
            l = clayers.get_layer(b_name, [clayers])
            config = {param: value for param, value in zip(values[::2], values[1::2])}
            layers.append(l.from_config(config))

        # regressor layers
        layers.append(Dense(1, activation='sigmoid'))

        try:
            # connecting the layers (functional API)
            model = keras.Sequential()

            for l in layers:
                model.add(l)    

            return model

        except ValueError:
            self.logger.exception('Invalid model')
            return None

    def evaluate(self, solution: Solution) -> bool:
        '''Evaluates a solution by executing the training and calculating the
        fitness on the validation or test

        The fitness is calculated on the validation set by default.
        Use test_eval to evaluate on the test set.

        Results are stored in the solution, including:
        - mse
        - loss
        - time spent
        - history training'''

        try:
            model = model_from_json(solution.phenotype)
            model.summary()
            model.compile(
                loss=self.loss,
                #TODO optimizer object must be instantiated every time
                # to create a new tf graph and avoid bugs
                optimizer=self._get_opt(),
                metrics=self.metrics)

            # defines the portions of data used for training and eval
            x_train, y_train = self.dataset.get_data('train')


            # there is validation data
            if self.dataset.x_valid is not None:
                self.train_args['validation_data'] = self.dataset.get_data('valid')

            # defines the folder for saving the model if requested
            # solution_path = f'solution_{solution.id}_weights.h5'

            # runs training
            start_time = dt.datetime.today()
            
            history = self.train_model(model, x_train, y_train,
                batch_size=self.batch_size,
                epochs=self.epochs,
                verbose=True,
                **self.train_args)
            
            if self.test_eval:
                x_eval, y_eval = self.dataset.get_data('test')
                # runs evaluations (on validation or test)
                loss, mse = self.test_model(model, x_eval, y_eval,
                    batch_size=self.batch_size,
                    verbose=self.verbose,
                    **self.test_args)
                fitness = mse
            else:
                # TODO custom metrics have to be named 'acc' and 'loss'
                # in order for this to work

                # temp solution that handles both naming styles
                if "val_mse" in history.history.keys():
                    mse_dict_key = "val_mse"
                else:
                    mse_dict_key = "val_mean_error_square"

                loss = history.history['val_loss'][-1]
                mse = history.history[mse_dict_key][-1]
                fitness = mse / np.mean(history.history[mse_dict_key])

            # updates the solution information
            # this is a regressor, so the smallest fitness is the best fitness
            if fitness == 0:
                fitness = sys.float_info.max
            else:
                fitness = 1/fitness

            solution.fitness = fitness
            solution.data['time'] = dt.datetime.today() - start_time
            solution.data['acc'] = 0
            solution.data['loss'] = loss
            solution.data['history'] = history.history

            return True

        except Exception: # pylint: disable=broad-except
            self.logger.exception('A problem was found during evaluation.')
            solution.fitness = -1
            return False

        finally:
            K.clear_session()



import os, logging, datetime as dt

import numpy as np

from keras import backend as K
from keras.models import Model, model_from_json

from cbioge.algorithms import GESolution
from cbioge.datasets import Dataset
from cbioge.grammars import Grammar
from cbioge.utils import checkpoint as ckpt
from .dnns.utils.callback import TimedStopping


class BaseProblem:
    ''' This BaseProblem class should be used as a reference of what a 
        problem class must have to be used with the evolutionary algorithms. 
        During the evolution the methods in this class are called.
    '''
    def __init__(self, parser: Grammar, verbose=False):
        if parser is None:
            raise AttributeError('Grammar parser cannot be None')

        self.parser = parser
        self.verbose = verbose
        self.logger = logging.getLogger('cbioge')

    def map_genotype_to_phenotype(self, solution):
        raise NotImplementedError('Not implemented yet.')

    def evaluate(self, solution):
        raise NotImplementedError('Not implemented yet.')


class DNNProblem(BaseProblem):
    ''' Base class used for Problems related to the automatic design of
        deep neural networks. Specific behavior must be implemented in child
        classes after calling the super() funcions.
    '''
    def __init__(self, parser: Grammar, dataset: Dataset, 
        batch_size=10, 
        epochs=1, 
        test_eval=False, 
        verbose=False, 
        **kwargs):

        super().__init__(parser, verbose)

        # if dataset is None:
        #     raise AttributeError('Dataset cannot be None')
        self.dataset = dataset

        self.batch_size = batch_size
        self.epochs = epochs
        self.test_eval = test_eval

        self.opt = 'adam'
        self.metrics = ['accuracy']

        self.kwargs = kwargs

    def map_genotype_to_phenotype(self, solution: GESolution) -> Model:
        raise NotImplementedError('Not implemented yet.')

    def evaluate(self, solution: GESolution, save_weights=False) -> bool:
        ''' Evaluates the phenotype

            phenotype: json structure containing the network architecture
            weights: network weights (optional)
        '''
        try:
            model = model_from_json(solution.phenotype)

            model.compile(loss=self.loss, 
                          optimizer=self.opt,
                          metrics=self.metrics)

            # defines the portions of data used for training and eval
            x_train, y_train = self.dataset.get_data('train')
            if self.test_eval: x_eval, y_eval = self.dataset.get_data('test')
            else: x_eval, y_eval = self.dataset.get_data('valid')

            # defines the folder for saving the model if requested
            solution_path = f'solution_{solution.id}_weights.h5'
            #solution_path = os.path.join(ckpt.ckpt_folder, f'solution_{solution.id}')

            # runs training
            start_time = dt.datetime.today()
            self.train_model(model, x_train, y_train, 
                batch_size=self.batch_size, 
                epochs=self.epochs, 
                save_weights=save_weights, 
                save_path=solution_path, 
                verbose=self.verbose, 
                **self.kwargs)

            # runs evaluations (on validation or test)
            _, accuracy = self.test_model(model, x_eval, y_eval, 
                batch_size=self.batch_size, 
                verbose=self.verbose, 
                **self.kwargs)

            # updates the solution information
            solution.fitness = accuracy
            solution.evaluated = True
            solution.data['time'] = dt.datetime.today() - start_time

            # clears keras session so memory wont stack up
            K.clear_session()

            return True

        except Exception as e:
            self.logger.exception('A problem was found during evaluation.')
            solution.fitness = -1
            solution.evaluated = True

            return False

    def train_model(self, model, x_train, y_train, save_weights=False, save_path=None, **kwargs):
        ''' executes the training of a model.

            # Parameters
            x_train: training data
            y_train: training labels
            batch_size: size of the batches used during training
            epochs: number of epochs the training will be executed

            # Optional parameters
            validation_data: Data on which to evaluate the loss and any model 
            metrics at the end of each epoch. Expects:
            - tuple (x_val, y_val)
            timelimit: max time (in seconds the model will be trained)
        '''
        callbacks = []

        if 'timelimit' in kwargs:
            callbacks.append(
                TimedStopping(seconds=kwargs['timelimit'], verbose=self.verbose))


        history = model.fit(x_train, y_train, callbacks=callbacks, **kwargs)

        if save_weights and save_path is not None:
            # only create the folders if we want to save the weights
            # if  not os.path.exists(save_path): os.makedirs(save_path)
            # model_path = os.path.join(save_path, f'weights.hdf5')
            model.save_weights(os.path.join(ckpt.ckpt_folder, save_path))
        
        return history

    def test_model(self, model, x_test, y_test, weights_path=None, **kwargs):

        if weights_path is not None: model.load_weights(weights_path)

        return model.evaluate(x_test, y_test, **kwargs)

    def predict_model(self, model, x_pred, save_pred=False, save_path=None, **kwargs):

        predictions = model.predict(x_pred, **kwargs)

        if save_pred and save_path is not None:
            if  not os.path.exists(save_path): os.makedirs(save_path)
            pred_path = os.path.join(save_path, 'predictions.npy')
            np.save(pred_path, predictions)

        return predictions
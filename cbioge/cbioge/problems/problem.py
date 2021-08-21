import typing
import os, logging, datetime as dt
from abc import ABC, abstractmethod

import numpy as np

from keras import backend as K
from keras.models import Model, model_from_json

from cbioge.algorithms import Solution
from cbioge.datasets import Dataset
from cbioge.grammars import Grammar
from cbioge.utils import checkpoint as ckpt


class BaseProblem(ABC):
    ''' This BaseProblem class should be used as a reference of what a 
        problem class must have to be used with the evolutionary algorithms. 
        During the evolution the methods in this class are called.
    '''
    def __init__(self, parser: Grammar, verbose: bool=False):
        if parser is None: raise AttributeError('Grammar parser cannot be None')

        self.parser = parser
        self.verbose = verbose
        self.logger = logging.getLogger('cbioge')

    @abstractmethod
    def map_genotype_to_phenotype(self, solution: Solution):
        raise NotImplementedError('Not implemented yet.')

    @abstractmethod
    def evaluate(self, solution: Solution):
        raise NotImplementedError('Not implemented yet.')


class DNNProblem(BaseProblem):
    ''' Base class used for Problems related to the automatic design of
        deep neural networks. Specific behavior must be implemented in child
        classes after calling the super() funcions.
    '''
    def __init__(self, parser: Grammar, dataset: Dataset, 
        batch_size=10, 
        epochs=1, 
        opt='adam', 
        loss='categorical_crossentropy', 
        metrics=['accuracy'], 
        test_eval=False, 
        verbose=False, 
        train_args={}, 
        test_args={}):

        super().__init__(parser, verbose)

        self.dataset = dataset

        self.batch_size = batch_size
        self.epochs = epochs
        self.test_eval = test_eval

        self.loss = loss
        self.opt = self._parse_opt(opt)
        self.metrics = metrics

        self.train_args = train_args
        self.test_args = test_args


    def _parse_opt(self, opt):
        if type(opt) == str: return opt
        return {'class': opt.__class__, 'config': opt.get_config()}

    def _get_opt(self):
        if type(self.opt) == str: return self.opt
        return self.opt['class'].from_config(self.opt['config'])

    def map_genotype_to_phenotype(self, solution: Solution) -> Model:
        raise NotImplementedError('Not implemented yet.')

    def evaluate(self, solution: Solution) -> bool:
        '''Evaluates a solution by executing the training and calculating the 
        fitness on the validation or test'''
        try:
            model = model_from_json(solution.phenotype)

            model.compile(
                loss=self.loss, 
                optimizer=self._get_opt(), #TODO GAMBI MASTER
                metrics=self.metrics)

            # defines the portions of data used for training and eval
            x_train, y_train = self.dataset.get_data('train')

            # defines the folder for saving the model if requested
            # solution_path = f'solution_{solution.id}_weights.h5'

            # runs training
            start_time = dt.datetime.today()
            history = self.train_model(model, x_train, y_train, 
                batch_size=self.batch_size, 
                epochs=self.epochs, 
                verbose=self.verbose, 
                **self.train_args)

            loss = history.history['val_loss'][-1]
            accuracy = history.history['val_acc'][-1]

            if self.test_eval:
                x_eval, y_eval = self.dataset.get_data('test')
                # runs evaluations (on validation or test)
                loss, accuracy = self.test_model(model, x_eval, y_eval, 
                    batch_size=self.batch_size, 
                    verbose=self.verbose, 
                    **self.test_args)

            # updates the solution information
            solution.fitness = accuracy
            solution.data['time'] = dt.datetime.today() - start_time
            solution.data['acc'] = accuracy
            solution.data['loss'] = loss
            solution.data['history'] = history.history

            return True

        except Exception:
            self.logger.exception('A problem was found during evaluation.')
            solution.fitness = -1
            solution.evaluated = True

            return False

        finally:
            K.clear_session()

    def train_model(self, model, x_train, y_train, save_path=None, **kwargs):
        ''' executes the training of a model.

            # Parameters
            x_train: training data
            y_train: training labels
            save_path: if value is different from None, the model weights will be saved

            # Optional parameters
            follows the same keras model.fit parameters
        '''

        #model.load_weights(os.path.join(ckpt.ckpt_folder, 'weights.h5'))

        history = model.fit(x_train, y_train, **kwargs)

        if save_path is not None:
            model.save_weights(os.path.join(ckpt.ckpt_folder, save_path))
        
        return history

    def test_model(self, model, x_test, y_test, **kwargs):

        if 'weights_path' in kwargs:
            model.load_weights(kwargs['weights_path'])
            kwargs.pop('weights_path')

        return model.evaluate(x_test, y_test, **kwargs)

    def predict_model(self, model, x_pred, save_path=None, **kwargs):

        predictions = model.predict(x_pred, **kwargs)

        if save_path is not None:
            if  not os.path.exists(save_path): os.makedirs(save_path)
            pred_path = os.path.join(save_path, 'predictions.npy')
            np.save(pred_path, predictions)

        return predictions
import os
import logging
import datetime as dt
from typing import Any, Union, List
from abc import ABC, abstractmethod

import numpy as np

from keras import backend as K
from keras.callbacks import History
from keras.models import Model, model_from_json

from ..algorithms import Solution
from ..datasets import Dataset
from ..grammars import Grammar
from ..utils import checkpoint as ckpt


class BaseProblem(ABC):
    '''BaseProblem class that works as a reference of what a problem class must
    have to be used with the evolutionary algorithms.

    Child classes must implement map_genotype_to_phenotype and evaluate.'''

    def __init__(self, parser: Grammar, verbose: bool=False):
        if parser is None:
            raise AttributeError('Grammar parser cannot be None')

        if not isinstance(parser, Grammar):
            raise AttributeError('parser must be of type Grammar')

        self.parser = parser
        self.verbose = verbose
        self.logger = logging.getLogger('cbioge')

    @abstractmethod
    def map_genotype_to_phenotype(self, solution: Solution) -> Any:
        raise NotImplementedError('Not implemented yet.')

    @abstractmethod
    def evaluate(self, solution: Solution):
        raise NotImplementedError('Not implemented yet.')


class DNNProblem(BaseProblem):
    '''Base class used for Problems related to the design of
    deep neural networks

    Specific behavior must be implemented in child classes.'''

    def __init__(self, parser: Grammar, dataset: Dataset,
        batch_size: int=10,
        epochs: int=1,
        opt: str='adam',
        loss: Union[str, callable]='categorical_crossentropy',
        metrics: list=['accuracy'],
        test_eval: bool=False,
        verbose: bool=False,
        train_args: dict={},
        test_args: dict={}
    ):

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

    def _parse_opt(self, opt: Union[str, callable]) -> Union[str, dict]:
        if isinstance(opt, str):
            return opt
        return {'class': opt.__class__, 'config': opt.get_config()}

    def _get_opt(self) -> Union[str, callable]:
        if isinstance(self.opt, str):
            return self.opt
        return self.opt['class'].from_config(self.opt['config'])

    def _reshape_mapping(self, mapping: List[Any]) -> List[List[Any]]:
        # groups layer name and parameters together

        cpy_mapping = mapping[:] # works for basic types
        new_mapping = []
        group = []
        while len(cpy_mapping) > 0:
            if cpy_mapping[0] != '#':
                group.append(cpy_mapping.pop(0))
            else:
                new_mapping.append(group)
                cpy_mapping.pop(0)
                group = []

        return new_mapping

    def _build_model(self, mapping: list) -> Model:
        raise NotImplementedError('_build_model must be implemented')

    def map_genotype_to_phenotype(self, solution: Solution) -> Model:

        mapping = self.parser.recursive_parse(solution.genotype)

        # creates the model
        model = self._build_model(mapping)

        if model is not None:
            solution.phenotype = model.to_json()
            solution.data['params'] = model.count_params()
        else:
            solution.phenotype = None
            solution.data['params'] = 0

        solution.data['mapping'] = mapping

        return model

    def evaluate(self, solution: Solution) -> bool:
        '''Evaluates a solution by executing the training and calculating the
        fitness on the validation or test

        The fitness is calculated on the validation set by default.
        Use test_eval to evaluate on the test set.

        Results are stored in the solution, including:
        - accuracy (on validation or test)
        - loss
        - time spent
        - history training'''

        try:
            model = model_from_json(solution.phenotype)

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
                verbose=self.verbose,
                **self.train_args)

            if self.test_eval:
                x_eval, y_eval = self.dataset.get_data('test')
                # runs evaluations (on validation or test)
                loss, accuracy = self.test_model(model, x_eval, y_eval,
                    batch_size=self.batch_size,
                    verbose=self.verbose,
                    **self.test_args)
                fitness = accuracy
            else:
                print(history.history.keys())
                # TODO custom metrics have to be named 'acc' and 'loss'
                # in order for this to work

                # temp solution that handles both naming styles
                if "val_acc" in history.history.keys():
                    acc_dict_key = "val_acc"
                else:
                    acc_dict_key = "val_accuracy"

                loss = history.history['val_loss'][-1]
                accuracy = history.history[acc_dict_key][-1]
                fitness = accuracy / np.mean(history.history[acc_dict_key])

            # updates the solution information
            solution.fitness = fitness
            solution.data['time'] = dt.datetime.today() - start_time
            solution.data['acc'] = accuracy
            solution.data['loss'] = loss
            solution.data['history'] = history.history

            return True

        except Exception: # pylint: disable=broad-except
            self.logger.exception('A problem was found during evaluation.')
            solution.fitness = -1

            return False

        finally:
            K.clear_session()

    def train_model(self, model: Model,
        x_train: list,
        y_train: list,
        save_path: str=None,
        **kwargs
    ) -> History:
        '''Executes the training of a model.

        # Parameters
        x_train: training data
        y_train: training labels
        save_path: if value is different from None, the model weights will be saved

        # Optional parameters
        kwargs: keras parameters, will be passed directly to model.fit'''

        #model.load_weights(os.path.join(ckpt.ckpt_folder, 'weights.h5'))

        history = model.fit(x_train, y_train, **kwargs)

        if save_path is not None:
            model.save_weights(os.path.join(ckpt.CKPT_FOLDER, save_path))

        return history

    def test_model(self, model: Model,
        x_test: list,
        y_test: list,
        weights_path: str=None,
        **kwargs
    ) -> Any:

        if weights_path is not None:
            model.load_weights(weights_path)

        return model.evaluate(x_test, y_test, **kwargs)

    def predict_model(self, model: Model,
        x_pred: list,
        save_path: str=None,
        **kwargs
    ) -> Any:

        predictions = model.predict(x_pred, **kwargs)

        if save_path is not None:
            if  not os.path.exists(save_path):
                os.makedirs(save_path)
            pred_path = os.path.join(save_path, 'predictions.npy')
            np.save(pred_path, predictions)

        return predictions

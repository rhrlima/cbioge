from cbioge.problems.problem import CoreProblem
import re, json, sys, os, gc, logging, keras
from tensorflow.keras.optimizers import Adam
from cbioge.utils.graphutils import *
from keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution1D, MaxPooling1D
from keras.utils import np_utils
from keras.layers.convolutional import Conv1D
from keras.layers import GRU, LSTM
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

logging.basicConfig(level=logging.INFO)

class MultipleProblem(CoreProblem):

    def __init__(self, 
        parser, 
        dataset=None, 
        verbose=False, build_method = None, 
        loss='categorical_crossentropy', 
        epochs = 1,
        metrics=['accuracy'], 
        batch_size=128, 
        workers = 1, multiprocessing = False, 
        training=True,learning_rate=1e-2,
        l2_reg=5e-4 / 2, opt=Adam, 
        es_patience=30, dropout=.5, timelimit=None):

        self.parser = parser
        self.loss = loss
        self.metrics = metrics
        self.epochs = epochs
        self.verbose = verbose
        self.batch_size = batch_size
        self.workers = workers
        self.multiprocessing = multiprocessing
        self.training = training
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        self.opt = opt(lr=self.learning_rate)
        self.es_patience = es_patience
        self.build_method = build_method
        self.dropout = dropout
        self.dataset = dataset
        self.timelimit = timelimit
        super().__init__(
            parser=self.parser, 
            batch_size=self.batch_size, 
            epochs=self.epochs, 
            timelimit=self.timelimit, 
            workers=self.workers, 
            multiprocessing=self.multiprocessing, 
            verbose=self.verbose, 
            metrics=self.metrics)

    def map_genotype_to_phenotype(self, genotype):
        self.mapping, genotype = self.parser.dsge_recursive_parse(genotype)
        self.mapping = self._reshape_mapping(self.mapping)
        self.mapping.insert(0, ['input', (None,)]) # input layer
        self.model = self._base_build(self.mapping)
        self._wrap_up_model(self.model)
        return json.dumps(self.model)


    def read_dataset_from_pickle(self, dataset, data_path):
        logging.info(f"::::: Reading {data_path}. Pickle file: {dataset}")
        A, X, y, train_mask, val_mask, test_mask = load_data(dataset_name=dataset, DATA_PATH=data_path)

        self.X = X
        self.y = y
        self.A = A
        self.train_mask = train_mask
        self.val_mask = val_mask,
        self.test_mask = test_mask
        self.n_classes = y.shape[-1]
        self.dropout = .5

    def evaluate(self, solution):
        ''' Evaluates the phenotype

            phenotype: json structure containing the network architecture
            weights: network weights (optional)

        '''
        F = self.X.shape[1]
        try:
            K.clear_session()
            #logging.info(f":: Mapping ({F}): {self.mapping}")
            model= keras.Sequential()
            for k, _map in enumerate(self.mapping):
                block, params = _map[0], _map[1:]
                if block == 'input':
                    model.add(keras.Input(shape=self.X.shape))
                if block == 'cnn':
                    model.add(Conv1D(params[0], kernel_size=params[1], padding='same', activation=params[2]))
                if block == 'dropout':
                    dropout = params[0]
                    model.add(Dropout(dropout))
                if block == 'lstm':
                    model.add(LSTM(params[0], activation=params[1], recurrent_activation=params[2]))
                if block == 'dense':
                    units = params[0]
                    if k == len(self.mapping)-1:
                        units = self.n_classes
                    model.add(Dense(units, activation=params[1]))
                if block == 'maxpool':
                    model.add(MaxPooling1D(pool_size=3))
                if block == 'flatten':
                    model.add(Flatten())



            model.compile(optimizer=self.opt,
                          loss='categorical_crossentropy',
                          metrics=self.metrics)

            model.summary()
            validation_data = (self.X, self.y)
            model.fit(self.X,
              self.y,
              epochs=self.epochs,
              batch_size=self.batch_size,
              validation_data=validation_data,
              verbose=self.verbose,
              callbacks=[EarlyStopping(patience=self.es_patience,  
                    restore_best_weights=True)])
            return True

        except Exception as e:
            logging.info(f'[evaluation] {e}')
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            logging.info(f'{exc_type}, {fname}, {exc_tb.tb_lineno}')
            solution.fitness = -1
            solution.params = 0
            solution.evaluated = True

            return False


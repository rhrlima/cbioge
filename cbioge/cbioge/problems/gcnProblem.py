from cbioge.problems.problem import CoreProblem
import re, json, sys, os, gc, logging
from tensorflow.keras.optimizers import Adam
from cbioge.utils.graphutils import *
from cbioge.utils.constants import *
from keras.layers import Input, Dropout
from spektral.layers import *
from keras.regularizers import l2
from spektral.utils.convolution import localpooling_filter
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


logging.basicConfig(level=logging.INFO)

class GCNProblem(CoreProblem):

    def __init__(self, 
        parser, dataset=None, 
        verbose=False, build_method = None, 
        loss='categorical_crossentropy', epochs = 1,
        metrics=['accuracy'], batch_size=-1, 
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


    def read_dataset_from_pickle(self, dataset, data_path):
        logging.info(f"::::: Reading {data_path}. Pickle file: {dataset}")
        A, X, y, train_mask, val_mask, test_mask = load_data(dataset_name=dataset, DATA_PATH=data_path)
        #A, X, y, train_mask, val_mask, test_mask = load_data_2(path=data_path, dataset=dataset)

        self.X = X
        self.y = y
        self.A = A
        self.train_mask = train_mask
        self.val_mask = val_mask,
        self.test_mask = test_mask
        self.n_classes = y.shape[-1]
        self.dropout = .5



    def map_genotype_to_phenotype(self, genotype):

        self.mapping, genotype = self.parser.dsge_recursive_parse(genotype)
        self.mapping = self._reshape_mapping(self.mapping)
        self.mapping.insert(0, ['input', (None,)]) # input layer
        self.model = self._base_build(self.mapping)
        self._wrap_up_model(self.model)
        return json.dumps(self.model)



    def evaluate(self, phenotype=None):
        try:
            K.clear_session()
            logging.info(f":: Mapping: {self.mapping}")

            model = None
            #build model from self.mapping
            X = self.X.toarray()
            A = self.A
            N = X.shape[0]          
            F = X.shape[1]          
            

            X_in = Input(shape=(F, ))
            fltr_in = Input((N, ), sparse=True)
            X_1 = X_in
            first_layer = None


            for _map in self.mapping:
                block, params = _map[0], _map[1:]
                if block == 'input':
                    continue
                if block == 'dropout':
                    dropout = params[0]
                    X_1 = Dropout(dropout)(X_1)
                elif block in GRAPH_CONVOLUTION_OPTIONS.keys():
                    layer = block
                    layer = GRAPH_CONVOLUTION_OPTIONS[layer]
                    if not first_layer:
                        first_layer = layer
                    units = params[0]
                    activation = params[1]
                    if activation=='softmax':
                        #last layer
                        units = self.n_classes
                    X_1 = layer(units,  
                        activation=activation,
                        kernel_regularizer=l2(self.l2_reg),
                        use_bias=False)([X_1, fltr_in])
                
            fltr = self.preprocess(first_layer, A)

            model = Model(inputs=[X_in, fltr_in], outputs=X_1)

            model.compile(optimizer=self.opt,
                          loss='categorical_crossentropy',
                          weighted_metrics=self.metrics)
            
            model.summary()

            validation_data = ([X, fltr], self.y, self.val_mask)
            model.fit([X, fltr],
              self.y,
              sample_weight=self.train_mask,
              epochs=self.epochs,
              batch_size=N,
              validation_data=validation_data,
              verbose=self.verbose,
              shuffle=False,  # Shuffling data means shuffling the whole graph
              callbacks=[
                  EarlyStopping(patience=self.es_patience,  
                    restore_best_weights=True)
              ])
            

            #scores = model.evaluate(x_valid, y_valid, verbose=self.verbose)
            scores = model.evaluate([X, fltr],
                  self.y,
                  sample_weight=self.test_mask,
                  batch_size=N)

            self.scores = scores

            count_params = model.count_params()

            gc.collect()
            
            del model

            return scores, count_params

        except Exception as e:
            logging.info(f'[evaluation] {e}')
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            logging.info(f'{exc_type}, {fname}, {exc_tb.tb_lineno}')
            return (-1, 0), 0
        

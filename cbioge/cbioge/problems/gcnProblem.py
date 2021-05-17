from cbioge.problems.problem import BaseProblem
import re, json, sys, os, gc
from tensorflow.keras.optimizers import Adam
from cbioge.utils.graphutils import *
from keras.layers import Input, Dropout
from spektral.layers import *
from keras.regularizers import l2
from spektral.utils.convolution import localpooling_filter
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

GRAPH_CONVOLUTION_OPTIONS = {
    "GraphConv": GraphConv,
    'ChebConv': ChebConv,
    'GraphSageConv':GraphSageConv,
    'ARMAConv':ARMAConv,
    'EdgeConditionedConv':EdgeConditionedConv,
    'GraphAttention':GraphAttention,
    'GraphConvSkip':GraphConvSkip,
    'APPNP':APPNP,
    'GINConv':GINConv,
    'DiffusionConv':DiffusionConv,
    'GatedGraphConv':GatedGraphConv,
    'AGNNConv':AGNNConv,
    'TAGConv':TAGConv,
    'CrystalConv':CrystalConv,
    'EdgeConv':EdgeConv
}

class GCNProblem(BaseProblem):

    def __init__(self, 
        parser, dataset=None, 
        verbose=False, build_method = None, 
        loss='categorical_crossentropy', epochs = 1,
        metrics=['accuracy'], batch_size=-1, 
        workers = 1, multiprocessing = False, 
        training=True,learning_rate=1e-2,
        l2_reg=5e-4 / 2, opt=Adam, 
        es_patience=30, dropout=.5):

        self.parser = parser
        self.loss = loss
        self.metrics = metrics
        self.epochs = epochs
        self.verbose = verbose
        self.batch_size = batch_size
        self.workers = workers
        self.multiprocessing = multiprocessing
        self.training = training
        self._initialize_blocks()
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        self.opt = opt(lr=self.learning_rate)
        self.es_patience = es_patience
        self.build_method = build_method
        self.dropout = dropout


    def _initialize_blocks(self):
        self.blocks = {
            'input': ['Input', 'shape'],
            'conv': ['conv_type', 'units', 'activation'],
            'dropout': ['Dropout', 'rate'],
            'filters': ['filter', 'max_degree', 'sym_norm']
        }

    

    def read_dataset_from_pickle(self, dataset, data_path):
        print(f"::::: Reading {data_path}. Pickle file: {dataset}")
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
        self.naming = {}
        self.mapping = None
        mapping, genotype = self.parser.dsge_recursive_parse(genotype)
        if self.verbose: print("dsge_recursive_parse", mapping)
        mapping = self._reshape_mapping(mapping)
        if self.verbose: print("_reshape_mapping", mapping)

        mapping = self._build_right_side(mapping)
        if self.verbose: print(mapping)
        model = {'class_name': 'Model', 
            'config': {'layers': [], 'input_layers': [], 'output_layers': []}}

        for i, layer in enumerate(mapping):
            block_name, params = layer[0], layer[1:]
            block = self._build_block(block_name, params)
            model['config']['layers'].append(block)

        self._wrap_up_model(model)
        self.mapping = mapping
        return json.dumps(model)



    def _build_block(self, block_name, params):

        base_block = {'class_name': None, 'name': None, 'config': {}, 'inbound_nodes': []}

        if block_name in self.naming:
            self.naming[block_name] += 1
        else:
            self.naming[block_name] = 0
        name = f'{block_name}_{self.naming[block_name]}'

        base_block['class_name'] = self.blocks[block_name][0]
        base_block['name'] = name
        for key, value in zip(self.blocks[block_name][1:], params):
            base_block['config'][key] = self._parse_value(value)
        return base_block

    def _parse_value(self, value):
        if type(value) is str:
            m = re.match('\\[(\\d+[.\\d+]*),\\s*(\\d+[.\\d+]*)\\]', value)
            if m:
                min_ = eval(m.group(1))
                max_ = eval(m.group(2))
                if type(min_) == int and type(max_) == int:
                    return np.random.randint(min_, max_)
                elif type(min_) == float and type(max_) == float:
                    return np.random.uniform(min_, max_)
                else:
                    raise TypeError('type mismatch')
            else:
                return value
        else:
            return value

    def _wrap_up_model(self, model):
        layers = model['config']['layers']
        stack = []
        for i, layer in enumerate(model['config']['layers']):
            if layer['class_name'] in ['push', 'bridge']: #CHECK
                stack.append(layers[i-1]) #layer before (conv)
                model['config']['layers'].remove(layers[i])

        for i, layer in enumerate(layers[1:]):

            last = model['config']['layers'][i]
            layer['inbound_nodes'].append([[last['name'], 0, 0]])

            if layer['class_name'] == 'Concatenate':
                other = stack.pop()
                # print('CONCATENATE', layer['name'], other['name'])
                layer['inbound_nodes'][0].insert(0, [other['name'], 0, 0])

        input_layer = model['config']['layers'][0]['name']
        output_layer = model['config']['layers'][-1]['name']
        model['config']['input_layers'].append([input_layer, 0, 0])
        model['config']['output_layers'].append([output_layer, 0, 0])

    def _build_right_side(self, mapping):

        blocks = None
        for block in reversed(mapping):
            name, params = block[0], block[1:]
            if name == 'maxpool':
                if blocks != None:
                    mapping.append(['upsamp', 2])
                    mapping.append(['conv', 0, 2, 1, 'same', 'relu'])
                    if ['bridge'] in blocks:
                        mapping.append(['concat', 3])
                        blocks.remove(['bridge'])
                    mapping.extend(blocks)
                blocks = []
            elif blocks != None:
                blocks.append(block)
        if blocks != None:
            if blocks != None:
                mapping.append(['upsamp', 2])
                mapping.append(['conv', 0, 2, 1, 'same', 'relu'])
                if ['bridge'] in blocks:
                    mapping.append(['concat', 3])
                    blocks.remove(['bridge'])
                mapping.extend(blocks)
        
        #mapping.insert(1, ['input', self.input_shape]) #input layer retirado, O input é controlado pelo Problem
        return mapping

    def _reshape_mapping(self, phenotype):
        new_mapping = []

        index = 0
        while index < len(phenotype):
            block = phenotype[index]
            if block == 'conv':
                end = index+4
            else:
                end = index+2
            new_mapping.append(phenotype[index:end])
            phenotype = phenotype[end:]

        return new_mapping

    def preprocess(self, first_layer, A):
        fltr = A.astype("f4")
        if first_layer in [GraphSageConv, GraphAttention, GINConv, GatedGraphConv, TAGConv]:
            print("no preprocessing")
            fltr = A.astype('f4')
        if first_layer in [GraphConv, ChebConv, ARMAConv, GraphConvSkip]:
            print("preprocessing like framework")
            fltr = first_layer.preprocess(A).astype('f4')

        if first_layer in [APPNP]:
            print("using localpooling_filter")
            fltr = localpooling_filter(A).astype('f4')
        return fltr


    def evaluate(self, phenotype=None):
        if self.build_method:
            return self.build_method(
                mapping=self.mapping, 
                optmizer=self.opt,
                metrics=self.metrics,
                epochs=self.epochs,
                es_patience=self.es_patience)
        try:
            
            K.clear_session()
            print(self.mapping)


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
                if block == 'dropout':
                    dropout = params[0]
                    X_1 = Dropout(dropout)(X_1)
                elif block == 'conv':
                    layer = params[0]
                    layer = GRAPH_CONVOLUTION_OPTIONS[layer]
                    if not first_layer:
                        first_layer = layer
                    units = params[1]
                    activation = params[2]
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
            print('[evaluation]', e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            return (-1, 0), 0
        

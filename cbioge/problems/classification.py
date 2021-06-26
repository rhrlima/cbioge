from cbioge.grammars.grammar import Grammar
import keras.layers
from keras.models import Model
from keras.utils import np_utils

import cbioge.layers as clayers
from cbioge.problems import DNNProblem
from cbioge.algorithms.solution import GESolution

class CNNProblem(DNNProblem):
    ''' Problem class for problems related to classification tasks for DNNs.
        This class includes methods focused on the design of CNNs.
    '''
    def __init__(self, parser: Grammar, dataset: dict,
        batch_size=10, 
        epochs=1, 
        timelimit=None, 
        test_eval=False, 
        verbose=False, 
        **kwargs):

        super().__init__(parser, dataset,
            batch_size, epochs, timelimit, test_eval, verbose, **kwargs)

        # classification specific
        self.loss = 'categorical_crossentropy'

    def _read_dataset(self, data_dict):
        ''' Reads a dataset stored in a dict. Check parent class for details.

            As a classification specific behavior, this method also reads
            the 'num_classes' key, and reshapes the labels to categorical.
        '''

        super()._read_dataset(data_dict)
        
        self.num_classes = data_dict['num_classes']
        self.y_train = np_utils.to_categorical(self.y_train, self.num_classes)
        self.y_valid = np_utils.to_categorical(self.y_valid, self.num_classes)
        self.y_test = np_utils.to_categorical(self.y_test, self.num_classes)

    def _sequential_build(self, mapping: list) -> Model:

        layers = []

        # input layer
        layers.append(keras.layers.Input(shape=self.input_shape))
        for block in mapping:
            b_name, values = block[0], block[1:]
            l = clayers._get_layer(self.parser.blocks[b_name][0],
                [keras.layers, clayers.layers])
            config = {param: value for param, value in zip(self.parser.blocks[b_name][1:], values)}
            layers.append(l.from_config(config))

        # classifier layers
        layers.append(keras.layers.Flatten())
        layers.append(keras.layers.Dense(self.num_classes, activation='softmax'))

        try:
            # connecting the layers (functional API)
            in_layer = layers[0]
            out_layer = layers[0]
            for l in (layers[1:]):
                out_layer = l(out_layer)

            return Model(inputs=in_layer, outputs=out_layer)
        except Exception as e:
            print('[problem.mapping] invalid model\n', e)
            return None

    def map_genotype_to_phenotype(self, solution: GESolution) -> Model:

        # classification problems apply a sequential build
        return self._sequential_build(
            self.parser.dsge_recursive_parse(solution.genotype))

        # mapping.insert(0, ['input', (None,)+self.input_shape]) # input layer
        # mapping.append(['dense', self.num_classes, 'softmax']) # output layer
        # model = self._base_build(mapping)
        # self._wrap_up_model(model)
        # return json.dumps(model)
        # return self.sequential_build(mapping)
        
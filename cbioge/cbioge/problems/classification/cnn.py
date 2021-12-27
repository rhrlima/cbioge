from typing import Union

from keras.layers import Input, Flatten, Dense
from keras.models import Model

from ..dnns import layers as clayers
from ...datasets import Dataset
from ...grammars import Grammar
from ...problems import DNNProblem


class CNNProblem(DNNProblem):
    ''' Problem class for problems related to classification tasks for DNNs.
        This class includes methods focused on the design of CNNs.'''

    def __init__(self, parser: Grammar, dataset: Dataset,
        batch_size: int=32,
        epochs: int=1,
        opt: str='adam',
        loss: Union[str, callable]='categorical_crossentropy',
        metrics: list=['accuracy'],
        test_eval: bool=False,
        verbose: bool=False,
        train_args=None,
        test_args=None
    ):

        super().__init__(parser, dataset, batch_size, epochs, opt, loss,
            metrics, test_eval, verbose, train_args, test_args)

    def _build_model(self, mapping: list) -> Model:

        reshaped_mapping = self._reshape_mapping(mapping)

        layers = []

        # input layer
        layers.append(Input(shape=self.dataset.input_shape))
        for block in reshaped_mapping:
            b_name, values = block[0], block[1:]
            l = clayers.get_layer(b_name, [clayers])
            config = {param: value for param, value in zip(values[::2], values[1::2])}
            layers.append(l.from_config(config))

        # classifier layers
        layers.append(Flatten())
        layers.append(Dense(self.dataset.num_classes, activation='softmax'))

        try:
            # connecting the layers (functional API)
            in_layer = layers[0]
            out_layer = layers[0]
            for l in layers[1:]:
                out_layer = l(out_layer)

            return Model(inputs=in_layer, outputs=out_layer)
        except ValueError:
            self.logger.exception('Invalid model')
            return None

import keras.layers as klayers


def get_layer(block_name: str, layers_database: list=None):

    if layers_database is None:
        layers_database = []

    # seaches first in custom layers, then keras layers
    layers_database.append(klayers)
    for ldb in layers_database:
        try:
            return getattr(ldb, block_name)
        except AttributeError:
            # exceptions should be handled only in the end
            continue

    raise AttributeError(f'{block_name} not found in the databases')


class ResBlock:
    ''' Custom layer that encapsulates and reproduce a Residual Block.

    A residual layer is composed of:\n
        Conv > Conv > BatchNorm > Add > ReLU'''

    def __init__(self, filters, kernel_size):
        self.conv1 = klayers.Conv2D(filters, kernel_size, padding='same')
        self.conv2 = klayers.Conv2D(filters, kernel_size, padding='same')
        self.batchnorm = klayers.BatchNormalization()
        self.conv3 = klayers.Conv2D(filters, 1, padding='same')
        self.add = klayers.Add()
        self.relu = klayers.ReLU()

    def __call__(self, inputs):
        output = self.conv1(inputs)
        output = self.conv2(output)
        output = self.batchnorm(output)
        # 1x1 conv to match shapes before adding
        aux = self.conv3(inputs)
        output = self.add([output, aux])
        output = self.relu(output)
        return output

    @classmethod
    def from_config(cls, config):
        return cls(**config)

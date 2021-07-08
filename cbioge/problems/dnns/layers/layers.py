from keras.layers import *

class ResBlock():
    ''' Custom layer that encapsulates and reproduce a Residual Block.'''

    def __init__(self, filters, kernel_size):
        self.conv1 = Conv2D(filters, kernel_size, padding='same')
        self.conv2 = Conv2D(filters, kernel_size, padding='same')
        self.batchnorm = BatchNormalization()
        self.conv3 = Conv2D(filters, 1, padding='same')
        self.add = Add()
        self.relu = ReLU()
        #self.conv1.name = 'res_' + self.conv1.name

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
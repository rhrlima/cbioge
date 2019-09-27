import os

from datasets.dataset import DataGenerator

from examples.unet_model import *


if __name__ == '__main__':

    path = 'datasets/membrane'
    input_shape = (256, 256, 1)

    train_ids = [f'{i}.npy' for i in range(30)]
    test_ids = train_ids[:30]

    train_gen = DataGenerator(os.path.join(path, 'train_npy'), train_ids, input_shape, batch_size=2, npy=True)
    test_gen = DataGenerator(os.path.join(path, 'test_npy'), test_ids, input_shape, batch_size=1, npy=True, shuffle=False)

    model = unet(input_shape)
    model.fit_generator(train_gen, steps_per_epoch=30, epochs=1, verbose=1)
    
    loss, acc = model.evaluate_generator(test_gen, 30, verbose=1)
    print('loss', loss, 'acc', acc)

from datasets.dataset import DataGenerator

from examples.unet_model import *


if __name__ == '__main__':

    input_shape = (64, 64, 1)
    batch_size = 8

    train_ids = [f'{i}.npy' for i in range(4000)]
    train_gen = DataGenerator('datasets/m2nist/train', train_ids, input_shape, batch_size=batch_size, npy=True)
    
    valid_ids = train_ids[:500]
    valid_gen = DataGenerator('datasets/m2nist/valid', valid_ids, input_shape, batch_size=batch_size, npy=True)

    test_ids = train_ids[:500]
    test_gen = DataGenerator('datasets/m2nist/test', test_ids, input_shape, batch_size=batch_size, npy=True, shuffle=False)

    model = unet(input_size=input_shape)
    model.fit_generator(
        train_gen, 
        validation_data=valid_gen, 
        steps_per_epoch=4000/batch_size, 
        epochs=8, 
        verbose=1,
        use_multiprocessing=True,
        workers=batch_size)

    loss, acc = model.evaluate_generator(train_gen, steps=500/batch_size, verbose=1)
    print('loss', loss, 'acc', acc)

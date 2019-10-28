from datasets.dataset import DataGenerator

from examples.unet_model import *

if __name__ == '__main__':

    input_shape = (64, 64, 1)

    train_ids = [f'{i}.npy' for i in range(4000)]
    train_gen = DataGenerator('datasets/m2nist/train', train_ids, input_shape, batch_size=32)
    
    valid_ids = train_ids[:500]
    valid_gen = DataGenerator('datasets/m2nist/valid', valid_ids, input_shape, batch_size=32)

    test_ids = train_ids[:500]
    test_gen = DataGenerator('datasets/m2nist/test', test_ids, input_shape, batch_size=32, shuffle=False)

    model = unet(input_size=input_shape)
    model.fit_generator(
        train_gen, 
        validation_data=valid_gen, 
        steps_per_epoch=4000, 
        epochs=1, 
        verbose=1)
    
    results = model.predict_generator(test_gen, 500, verbose=1)
    model.fit_generator(train_gen, steps_per_epoch=500, epochs=1, verbose=1)
    print('loss', loss, 'acc', acc)

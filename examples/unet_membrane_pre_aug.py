import glob
import os

from datasets.dataset import DataGenerator

from examples.unet_model import *

#TEMP
import skimage.io as io

if __name__ == '__main__':

    print('UNET ISOLDATED EVALUATION')

    path = 'datasets/membrane'
    input_shape = (256, 256, 1)

    train_gen = DataGenerator(os.path.join(path, 'train_posproc'), input_shape, batch_size=2, shuffle=False)
    test_gen = DataGenerator(os.path.join(path, 'test_posproc'), input_shape, batch_size=1, shuffle=False)

    model = unet(input_shape)
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.fit_generator(train_gen, steps_per_epoch=300, epochs=1, verbose=1)

    loss, acc = model.evaluate_generator(test_gen, steps=30, verbose=1)
    print('loss', loss, 'acc', acc)

    # results = model.predict_generator(test_gen, steps=30, verbose=1)
    # for i, img in enumerate(results):
    #     io.imsave(os.path.join(path, f'test/pred/{img}.png'), img)
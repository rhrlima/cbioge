import os

import numpy as np

from keras import callbacks
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.preprocessing.image import ImageDataGenerator

from datasets.dataset import DataGenerator

import skimage.io as io
import skimage.transform as trans

def unet(input_size):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(input = inputs, output = conv10)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])

    return model

def trainGenerator(batch_size, train_path, aug_dict, target_size = (256, 256), seed = 1):
  
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = ['image'],
        class_mode = None,
        color_mode = "grayscale",
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = None,
        save_prefix  = 'image',
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = ['label'],
        class_mode = None,
        color_mode = "grayscale",
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = None,
        save_prefix  = 'mask',
        seed = seed)

    for img, mask in zip(image_generator, mask_generator):
        if np.max(img) > 1:
            img = img / 255
            mask = mask / 255
            mask[mask > 0.5 ] = 1
            mask[mask <= 0.5] = 0
        yield img, mask

def testGenerator(test_path, num_image = 30, target_size = (256,256)):
    for i in range(num_image):
        img = io.imread(os.path.join(test_path, f'{i}.png'), as_gray = True)
        img = img / 255
        img = trans.resize(img, target_size)
        img = np.reshape(img, img.shape + (1,))
        img = np.reshape(img,(1,)+img.shape) #???
        yield img

def adjust_image(img, threshold=0.5):
    img = (img - img.min()) / (img.max() - img.min())
    img[img > threshold ] = 1.
    img[img <= threshold] = 0.
    return img

def iou_accuracy(true, pred):
    intersection = true * pred
    union = true + ((1. - true) * pred)
    return np.sum(intersection) / np.sum(union)

if __name__ == '__main__':

    input_shape = (256, 256, 1)

    #TEST
    data_gen_args = dict(rotation_range=0.2,
                width_shift_range=0.05,
                height_shift_range=0.05,
                shear_range=0.05,
                zoom_range=0.05,
                horizontal_flip=True,
                fill_mode='nearest')

    train_gen = trainGenerator(2, 'datasets/membrane/train', data_gen_args)
    test_gen = testGenerator('datasets/membrane/test/image', 30)

    model_checkpoint = callbacks.ModelCheckpoint('unet_membrane2.hdf5', monitor='loss', verbose=1, save_best_only=True)
    model = unet(input_size=input_shape)
    model.fit_generator(
        train_gen, 
        steps_per_epoch=300, 
        epochs=1, 
        callbacks=[model_checkpoint], 
        verbose=1)
    
    results = model.predict_generator(test_gen, 30, verbose=1)

    acc = 0.0
    for i, pred in enumerate(results):
        io.imsave(f'datasets/membrane/npy/test/pred/{i}.png', pred)
        true = io.imread(f'datasets/membrane/test/label/{i}.png')
        pred = adjust_image(pred)
        true = adjust_image(true)
        acc += iou_accuracy(true, pred)

    print('acc', acc/len(results))

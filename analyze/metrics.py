import glob
import os
import pickle
import re

import numpy as np

from utils.image import *


def load_dataset(file_name):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data


def get_natural_key(string):
    matches = re.findall('(\\d+)', string)
    if len(matches) > 0:
        return int(matches[-1])
    else:
        return 0


def load_predictions(folder):
    files = glob.glob(os.path.join(folder, '*.png'))
    files.sort(key=lambda x: get_natural_key(x))
    print(files)
    images = [load_image(f, as_gray=True) for f in files]
    return np.array(images)


def apply_measures(labels, preds, name='plot.png'):

    wmetric = WeightedMetric(w_spe=.1, w_dic=.4, w_sen=.4, w_jac=.1)

    measures = {
        #'iou': [],
        'jac': [],
        'spc': [],
        'sen': [],
        'dic': [],
        'all': []
    }

    for l, p in zip(labels, preds):

        #p = 1 - p

        p = normalize(p)
        l = normalize(l)

        p = binarize(p)
        l = binarize(l)

        #measures['iou'].append(iou_accuracy(l, p))
        measures['jac'].append(1.0-jaccard_distance(l, p))
        measures['spc'].append(specificity(l, p))
        measures['sen'].append(sensitivity(l, p))
        measures['dic'].append(dice_coef(l, p))
        #measures['all'].append(weighted_measures(l, p, *[.45, .05, .25, .25]))
        #measures['all'].append(weighted_measures(l, p, *[0.2, .05, .375, .375]))
        measures['all'].append(wmetric.get_metric()(l, p))

    markers = ['o-', '*-', 'v-', 'x-', '+-']

    for i, key in enumerate(measures):
        #print(key, measures[key])
        print(key, np.mean(measures[key]))
        plt.plot(measures[key], markers[i], label=key)
        #plt.scatter(range(len(labels)), measures[key], s=14, edgecolors='none', c='black',)
    print('---')
    plt.title(name)
    plt.xlabel('Image')
    plt.ylabel('Measure')
    plt.legend(loc="lower right")
    plt.savefig(f'{name}.png')
    plt.show()
    plt.clf()


if __name__ == '__main__':

    dataset_name = 'textures1'
    preds_names = ['unet']#, 'acc1', 'jac1', 'dic1', 'spe1', 'sen1']
    #preds_names = ['unet2', 'white', 'black']
    #preds_names = ['rand3', 'tex3']

    #dataset_name = 'textures_regular'
    #preds_names = ['bestR', 'bestTR', 'unetR']

    #dataset_name = 'textures_moderate'
    #preds_names = ['bestM', 'bestTM', 'unetM']

    #dataset_name = 'textures_full'
    #preds_names = ['bestF', 'bestTF', 'unetF']

    dataset = load_dataset(f'datasets/{dataset_name}.pickle')
    
    images = dataset['x_test']
    labels = dataset['y_test']

    preds = load_predictions('results/'+preds_names[0])
    #predsT = load_predictions('analyze/'+preds_names[1])
    #predsU = load_predictions('analyze/'+preds_names[2])

    i = np.reshape(images[0], (256, 256))
    l = np.reshape(labels[0], (256, 256))
    plot([i, l, preds[0]])#, predsT[0], predsU[0]])

    print(labels.shape)
    
    # apply_measures(labels, preds)
    # apply_measures(labels, labels)

    # preds = []
    for i, pname in enumerate(preds_names):
        preds = load_predictions(f'results/{pname}')
        print(preds.shape)
        apply_measures(labels, preds, pname)

    #apply_measures(labels, labels, 'labellabel')

    # labels = []
    # preds = []
    # for i in range(5):
    #     labels.append(load_image('analyze/test/true1.png'))
    #     preds.append(load_image(f'analyze/test/seg{i+1}.png'))

    # #plot([true, seg1])

    # apply_measures(labels, preds)
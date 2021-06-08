import copy
import glob
import os
import pickle
import re

import numpy as np

from utils.image import *

from skimage.util import compare_images
from sklearn.metrics import precision_recall_curve, f1_score, average_precision_score

import matplotlib.pyplot as plt


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
    images = [load_image(f, as_gray=True) for f in files]
    return np.array(images)


def apply_measures(dataset, folder):

    # le o dataset  
    dataset = load_dataset(f'datasets/{dataset}.pickle')
    images = dataset['x_test']
    labels = dataset['y_test']#[:10]
    labels = labels.astype('uint8')

    # le as predicoes
    preds = load_predictions(os.path.join('results', folder))
    #preds = preds[:10]

    if preds.max() > 1:
        preds = preds / 255

    wmetric = WeightedMetric(w_spe=.1, w_dic=.4, w_sen=.4, w_jac=.1)

    measures = {
        'jac': [],
        'spc': [],
        'sen': [],
        'dic': [],
        'all': []
    }

    #pre, rec, thresholds = precision_recall_curve(labels.flatten(), preds.flatten())
    thresholds = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9]

    cont = 0
    for l, p in zip(labels, preds):

        print('image', cont)
        cont += 1

        best_jac = 0
        best_spe = 0
        best_sen = 0
        best_dic = 0
        best_all = 0
        for t in thresholds:
            p_bin = binarize(p, t)

            #plot([l[:,:,0], p, p_bin])

            best_jac = max(best_jac, 1.0-jaccard_distance(l, p_bin))
            best_spe = max(best_spe, specificity(l, p_bin))
            best_sen = max(best_sen, sensitivity(l, p_bin))
            best_dic = max(best_dic, dice_coef(l, p_bin))
            best_all = max(best_all, wmetric.get_metric()(l, p_bin))

        measures['jac'].append(best_jac)
        measures['spc'].append(best_spe)
        measures['sen'].append(best_sen)
        measures['dic'].append(best_dic)
        measures['all'].append(best_all)

    markers = ['o-', '*-', 'v-', 'x-', '+-']

    for i, key in enumerate(measures):
        print(key, np.mean(measures[key]))
        plt.plot(measures[key], markers[i], label=key)

    print('---')
    plt.title(folder)
    plt.xlabel('Image')
    plt.ylabel('Measure')
    plt.legend(loc="lower right")
    plt.savefig(f'{folder}.png')
    plt.show()
    plt.clf()


def OIS_f1(labels, preds):

    f1_mean = []

    index = 0
    for l, p in zip(labels, preds):

        pre, rec, thresholds = precision_recall_curve(l.flatten(), p.flatten())

        best_f1 = 0
        best_t = 0
        for t in thresholds:
            p_bin = binarize(p, t)
            f1 = f1_score(l.flatten(), p_bin.flatten())

            if f1 > best_f1:
                best_f1 = f1
                best_t = t

        f1_mean.append((index, best_f1, best_t))
        #print(index, best_f1, best_t)
        index += 1

    return f1_mean


def ODS_f1(labels, preds):

    pre, rec, thresholds = precision_recall_curve(labels.flatten(), preds.flatten())

    best_f1 = 0
    best_t = 0
    for t in thresholds:

        preds_bin = copy.deepcopy(preds)
        preds_bin[preds_bin > t] = 1
        preds_bin[preds_bin <= t] = 0
        f1 = f1_score(labels.flatten(), preds_bin.flatten())

        if f1 > best_f1:
            #print('from', best_f1, 'to', f1, 'with', t)
            best_f1 = f1
            best_t = t

    #print(best_f1, best_t)
    return (best_f1, best_t)


def run_metrics(dataset, folder):

    print(folder)

    # le o dataset  
    dataset = load_dataset(f'datasets/{dataset}.pickle')
    images = dataset['x_test']
    labels = dataset['y_test']
    labels = labels.astype('uint8')

    # le as predicoes
    preds = load_predictions(os.path.join('results', folder))

    if preds.max() > 1:
        preds = preds / 255

    print('labels', labels.min(), labels.max())
    print('preds', preds.min(), preds.max())

    # amostra uma imagem
    index = 0
    i = images[index,:,:,0]
    l = labels[index,:,:,0]
    p = preds[index]

    #plot([i, l, p])

    # plot_precision_recall(labels.flatten(), preds.flatten())

    ois_mean = OIS_f1(labels, preds)
    ods_f1, ods_t = ODS_f1(labels, preds)
    avg_pre = average_precision_score(labels.flatten(), preds.flatten())

    print('OIS', np.mean(ois_mean, axis=0)[1])
    print('ODS', ods_f1, ods_t)
    print('AP', avg_pre)
   

def plot_precision_recall_group(dataset, folders):

    dataset = load_dataset(f'datasets/{dataset}.pickle')
    images = dataset['x_test']
    labels = dataset['y_test']
    labels = labels.astype('uint8')

    plt.figure()

    for folder in folders:
        preds = load_predictions(os.path.join('results', folder))

        if preds.max() > 1:
            preds = preds / 255

        pre, rec, t = precision_recall_curve(labels.flatten(), preds.flatten())
        plt.step(rec, pre, where='post', label=folder)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.legend(loc='upper right')
    plt.show()


def plot_precision_recall(y_true, y_pred):

    pre, rec, t = precision_recall_curve(y_true, y_pred)

    plt.figure()
    plt.step(rec, pre, where='post')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.legend(loc='upper right')
    plt.show()



if __name__ == '__main__':

    # run_metrics('bsds500', 'unet_bsds')
    # run_metrics('bsds500', 'unet_bsds2')
    # run_metrics('bsds500', 'bsds5002')
    # run_metrics('bsds500', 'canny')

    #folders = ['unet_bsds', 'bsds5002', 'canny']
    #plot_precision_recall_group('bsds500', folders)


    apply_measures('bsds500', 'unet_bsds')
    # apply_measures('bsds500', 'bsds5001')
    # apply_measures('bsds500', 'bsds5002')
    # apply_measures('bsds500', 'bsds5003')
    # apply_measures('bsds500', 'bsds5004')
    # apply_measures('bsds500', 'bsds5005')
    #apply_measures('bsds500', 'canny')

    #apply_measures('membrane', 'memb1')
    #apply_measures('membrane', 'memb2')
    #apply_measures('membrane', 'memb3')
    #apply_measures('membrane', 'memb4')
    apply_measures('membrane', 'unet_membrane')
    #apply_measures('membrane', 'membrane-test')
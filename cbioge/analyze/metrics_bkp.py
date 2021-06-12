import glob
import os
import pickle
import re

import numpy as np

from utils.image import *

from skimage.util import compare_images
from sklearn.metrics import precision_recall_curve, f1_score, average_precision_score

import matplotlib.pyplot as plt

import cv2


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
    #print(files)
    images = [load_image(f, as_gray=True) for f in files]
    return np.array(images)


def apply_measures(labels, preds, name='plot.png'):

    wmetric = WeightedMetric(w_spe=.1, w_dic=.4, w_sen=.4, w_jac=.1)

    measures = {
        'jac': [],
        'spc': [],
        'sen': [],
        'dic': [],
        'all': []
    }

    for l, p in zip(labels, preds):

        #p = 1 - p

        p = normalize(p)
        #l = normalize(l)

        p = binarize(p)    
        #l = binarize(l)

        measures['jac'].append(1.0-jaccard_distance(l, p))
        measures['spc'].append(specificity(l, p))
        measures['sen'].append(sensitivity(l, p))
        measures['dic'].append(dice_coef(l, p))
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


def OIS_f1(labels, preds):

    precision = []
    recall = []
    f1_mean = []

    for l, p in zip(labels, preds):

        pre, rec, thresholds = precision_recall_curve(l.flatten(), p.flatten())
        precision.append(pre)
        recall.append(rec)

        best_f1 = 0
        best_t = 0
        for t in thresholds:
            p_bin = binarize(p, t)
            f1 = f1_score(l.flatten(), p_bin.flatten())

            if f1 > best_f1:
                print('from', best_f1, 'to', f1, 'with', t)
                best_f1 = f1
                best_t = t
        f1_mean.append(best_f1)

    return f1_mean, np.array(precision), np.array(recall)


def ODS_f1(labels, preds):

    best_f1 = 0
    best_t = 0
    thresholds = []
    precision = []
    recall = []

    for l, p in zip(labels, preds):
        pre, rec, t = precision_recall_curve(l.flatten(), p.flatten())
        precision.append(pre)
        recall.append(rec)
        thresholds = np.concatenate((thresholds, t))

    unique_t = np.unique(thresholds)
    print(len(thresholds))
    print(len(unique_t))

    for t in unique_t:
        f1_mean = 0
        for l, p in zip(labels, preds):
            p_bin = binarize(p, t)
            f1_mean += f1_score(l.flatten(), p_bin.flatten())
        f1_mean /= len(labels)

        if f1_mean > best_f1:
            print('from', best_f1, 'to', f1_mean, 'with', t)
            best_f1 = f1_mean
            best_t = t

    print(best_f1, best_t)
    return best_f1, np.array(precision), np.array(recall)


def get_best_f1_threshold(l, p):
    scores = []
    for t in range(255):
        temp = binarize(p, t)
        f1 = f1_score(l.flatten(), temp.flatten())
        scores.append((f1, t))
    return max(scores)


def average_precision(labels, preds):

    avg_pre = []
    for l, p in zip(labels, preds/255):
        avg_pre.append(average_precision_score(l.flatten(), p.flatten()))
    return avg_pre


def get_precision_recall(labels, preds):

    max_len = 155

    all_pre = []
    all_rec = []
    avg_pre = []
    for l, p in zip(labels, preds/255):
        
        avg_pre.append(average_precision_score(l.flatten(), p.flatten()))

        pre, rec, t = precision_recall_curve(l.flatten(), p.flatten())
        
        if len(pre) < max_len:
            diff = max_len - len(pre)
            pre = np.concatenate((pre, [0] * diff))
            rec = np.concatenate((rec, [1] * diff))

        all_pre.append(pre)
        all_rec.append(rec)

    plt.figure()
    plt.step(np.mean(all_rec, axis=0), np.mean(all_pre, axis=0))

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.show()

    return all_pre, all_rec, np.mean(avg_pre)


def OIS(labels, preds):

    cont = 0
    all_f1 = []
    all_p = []
    for l, p in zip(labels, preds):

        f1, t = get_best_f1_threshold(l, p)
        all_f1.append((f1, t))
        all_p.extend(binarize(p, t).flatten())

        print(cont, f1, t)
        cont += 1

    print('all f1', np.mean(all_f1, axis=0))
    print('fla f1', f1_score(labels.flatten(), all_p))

    return all_f1


def OIS2(labels, preds):

    cont = 0
    all_l = []
    all_p = []
    all_t = []
    all_f1 = []
    for l, p in zip(labels, preds):

        f1, t = get_best_f1_threshold(l, p)
        all_t.append((cont, t, f1))
        all_f1.append(f1)
        print(cont, f1, t)

        all_l.extend(l.flatten())
        all_p.extend(p.flatten())
        cont += 1

    return all_t


def ODS(labels, preds):

    best_f1 = 0
    best_t = 0
    for t in range(255):

        f1 = f1_score(labels.flatten(), binarize(preds.flatten(), t))

        # mean_f1 = []
        # for l, p in zip(labels, preds):
        #     temp = binarize(p, t)
        #     f1 = f1_score(l.flatten(), temp.flatten())
        #     mean_f1.append(f1)
        #     #print(f1)
        # mean_f1 = np.mean(mean_f1)

        # if mean_f1 > best_f1:
        #     #print('improved from', best_f1, 'to', mean_f1, 'with', t)
        #     best_f1 = mean_f1
        #     best_t = t
        if f1 > best_f1:
            print('improved from', best_f1, 'to', f1, 'with', t)
            best_f1 = f1
            best_t = t

    #print('ODS AVG F1', best_f1, best_t)
    return best_f1, best_t


def run_metrics(dataset, folder):

    print(folder)

    # le o dataset  
    dataset = load_dataset(f'datasets/{dataset}.pickle')
    images = dataset['x_test']
    labels = dataset['y_test']
    labels = labels.astype('uint8')

    # le as predicoes
    preds = load_predictions(os.path.join('results', folder))

    print('labels', labels.min(), labels.max())
    print('preds', preds.min(), preds.max())

    #plot_precision_recall(labels.flatten(), preds.flatten())

    #ois_f1 = OIS(labels, preds)
    #ois2_f1 = OIS2(labels, preds)
    #ods_f1, ods_t = ODS(labels, preds)
    #avg_pre = average_precision_score(labels.flatten(), preds.flatten())

    #print('OIS', np.mean(ois_f1, axis=0))
    # #print('OIS2', np.mean(ois2_f1, axis=1))
    #print('ODS', ods_f1, ods_t)
    #print('AP', avg_pre)

    # amostra uma imagem
    index = 0
    i = images[index,:,:,0]
    l = labels[index,:,:,0]
    p = preds[index]

    # TEST
    # p_bin = binarize(p, 128)
    # F1a = get_best_f1_threshold(l, p)#f1_score(l.flatten(), p.flatten(), average=None)
    # F1b = f1_loss(l, p)

    #print(F1a, F1b)

    # TESTS
    # blank = binarize(p, 200)
    # F1c = f1_loss(l, blank)
    # F1d = f1_loss(l, l)
    # print(F1c, F1d)

    #plot([i, l, p, blank])

    #p2 = binarize(p, 114)

    #plot([i, l, p, p2])

    # metrics = [
    #     jaccard_distance,
    #     specificity,
    #     sensitivity,
    #     dice_coef
    # ]

    # for m in metrics:
    #     print(m(l, p), m(l, p2), m(l, l))

    apply_measures(labels, preds)
    pre, rec, t = precision_recall_curve(labels.flatten(), preds.flatten())

    return pre, rec


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

    plt.figure()

    pre, rec = run_metrics('bsds500', 'unet_bsds')
    plt.step(rec, pre, where='post', label='unet')
    # pre, rec = run_metrics('bsds500', 'bsds5001')
    # plt.step(rec, pre, where='post', label='bsds1')
    pre, rec = run_metrics('bsds500', 'bsds5002')
    plt.step(rec, pre, where='post', label='bsds2')
    # pre, rec = run_metrics('bsds500', 'bsds5003')
    # plt.step(rec, pre, where='post', label='bsds3')
    # pre, rec = run_metrics('bsds500', 'bsds5004')
    # plt.step(rec, pre, where='post', label='bsds4')
    # pre, rec = run_metrics('bsds500', 'bsds5005')
    # plt.step(rec, pre, where='post', label='bsds5')

    #pre, rec = run_metrics('bsds500gray', 'bsdsgray')
    #plt.step(rec, pre, where='post', label='unet gray')
    
    pre, rec = run_metrics('bsds500', 'canny')
    plt.step(rec, pre, where='post', label='bsds canny')

    #pre, rec = run_metrics('bsds500gray', 'canny')
    #plt.step(rec, pre, where='post', label='bsdsG canny')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.legend(loc='upper right')
    plt.show()


    # dataset_name = 'bsds500'
    # preds_folder = 'unet_bsds'

    # # le o dataset  
    # dataset = load_dataset(f'datasets/{dataset_name}.pickle')
    # images = dataset['x_test']
    # labels = dataset['y_test']
    # labels = labels.astype('uint8')

    # # le as predicoes
    # preds = load_predictions(os.path.join('results', preds_folder))
    # #preds = preds / 255

    # # amostra uma imagem
    # index = 0
    # i = images[index,:,:,0]#np.reshape(images[0], (256, 256))
    # l = labels[index,:,:,0]#np.reshape(labels[0], (256, 256))
    # p = preds[index]

    # # for index in range(10):
    # #     i = images[index,:,:,0]#np.reshape(images[0], (256, 256))
    # #     l = labels[index,:,:,0]#np.reshape(labels[0], (256, 256))
    # #     p = preds[index]
    # #     plot([i, l, p])

    # #f1, t = get_best_threshold(l, p)
    # #p2 = binarize(p, t)

    # #plot([i, l, p, p2])

    # #OIS(labels, preds)
    # ODS(labels, preds)




    # print('dataset', l.shape, l.min(), l.max())
    # print('preds', p.shape, p.min(), p.max())

    # #

    # # pre, rec, t = precision_recall_curve(l.flatten(), l.flatten())
    # # #print(t)
    # # plt.plot(pre, rec, label='label')
    # # #plt.show()

    # # pre, rec, t = precision_recall_curve(l.flatten(), p.flatten())
    # # #print(t)
    # # #print(rec)
    # # plt.plot(pre, rec, label='pred')
    # # # #plt.show()

    # # f1, cut, pre, rec = OIS_f1(l, p)
    # # p_bin = binarize(p, cut)
    # # plt.plot(pre, rec, label='bin pred')

    # # # plt.legend(loc="lower left")
    # # plt.show()

    # # plot([i, l, p, p_bin])

    # ois_f1, _, _ = OIS_f1(labels, preds)
    # ods_f1, pre, rec = ODS_f1(labels, preds)

    # avg_pre = 0
    # flat_pre = []
    # flat_rec = []
    # for p, r in zip(pre, rec):
    #     flat_pre = np.concatenate((flat_pre, p))
    #     flat_rec = np.concatenate((flat_rec, r))
    #     avg_pre += np.mean(p)
    # avg_pre /= len(pre)
    
    # print('OIS', np.mean(ois_f1))
    # print('ODS', ods_f1)
    # print('AP', avg_pre)

    # #plt.plot(flat_pre, flat_rec)
    # #plt.show()

    # apply_measures(labels, preds, 'membrane')
    


    # for name in preds_names:
    #     print(name)
    #     
    
    #     p = preds[index]
    #     p = normalize(p)
    #     p = binarize(p)

    #     print(p.shape, p.min(), p.max())

    #     # thresholds = [.4,.45,.5,.55,.6]
    #     # for t in thresholds:
    #     #     p = binarize(p, t)


    #     plt.hist(p.flatten(), bins=256, range=(0, 1))
    #     plt.show()
    #     #, predsT[0], predsU[0]])

    #     apply_measures(labels, preds, name)

        #pre, rec, t = precision_recall_curve(l.flatten(), p.flatten())
        #f1 = f1_score(l.flatten(), p.flatten())
        #print('f1', f1)

        #plt.plot(pre, rec)
        #plt.show()
    #print(labels.shape)
    
    # apply_measures(labels, preds)
    # apply_measures(labels, labels)

    # preds = []
    # for i, pname in enumerate(preds_names):
    #     preds = load_predictions(pname)
    #     print(preds.shape)
    #     apply_measures(labels, preds, pname)

    #apply_measures(labels, labels, 'labellabel')

    # labels = []
    # preds = []
    # for i in range(5):
    #     labels.append(load_image('analyze/test/true1.png'))
    #     preds.append(load_image(f'analyze/test/seg{i+1}.png'))

    # #plot([true, seg1])

    # apply_measures(labels, preds)
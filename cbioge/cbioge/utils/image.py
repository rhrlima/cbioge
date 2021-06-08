import numpy as np

import skimage.io as io
import skimage.transform as trans

import matplotlib.pyplot as plt


def load_image(path, npy=False, as_gray=False):

    return io.imread(path, as_gray=as_gray) if not npy else np.load(path)


def write_image(path, img, npy=False):

    io.imsave(path, img) if not npy else np.save(path, img)


def binarize(img, threshold=0.5):

    new_img = np.array(img)
    new_img[img > threshold ] = 1
    new_img[img <= threshold] = 0
    return new_img


def normalize(img):
    if img.min() == img.max():
        return img / 255
    return (img - img.min()) / (img.max() - img.min())


def resize(img, size):

    return trans.resize(img, size)


def calculate_output_size(img_shape, k, s, p):
    ''' width, height, kernel, padding, stride'''
    index = 1 if len(img_shape) == 4 else 0
    w = img_shape[index]
    h = img_shape[index+1]

    p = 0 if p == 'valid' else (k-1) / 2
    ow = ((w - k + 2 * p) // s) + 1
    oh = ((h - k + 2 * p) // s) + 1
    return (int(ow), int(oh))


def plot(imgs):
    items = len(imgs)
    for i in range(items):
        plt.subplot(1, items, i+1)
        plt.imshow(imgs[i])
    plt.show()


# distances
def iou_accuracy(y_true, y_pred):
    intersection = y_true * y_pred
    union = y_true + ((1. - y_true) * y_pred)
    return np.sum(intersection) / np.sum(union)


def jaccard_distance(y_true, y_pred, smooth=1):
    intersection = np.sum(np.abs(y_true * y_pred))
    sum_ = np.sum(np.abs(y_true) + np.abs(y_pred))
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth


def specificity(y_true,y_pred):
    specificity=0
    y_true_f = y_true.flatten()#np.flatten(y_true)
    y_pred_f = y_pred.flatten()#np.flatten(y_pred)
    TP=np.sum(y_true_f*y_pred_f)
    TN=np.sum((1-y_true_f)*(1-y_pred_f))
    FP=np.sum((1-y_true_f)*y_pred_f)
    FN=np.sum(y_true_f*(1-y_pred_f))
    specificity=(TN)/((TN+FP))
    return specificity


def sensitivity(y_true,y_pred):
    sensitivity=0
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    TP=np.sum(y_true_f*y_pred_f)
    TN=np.sum((1-y_true_f)*(1-y_pred_f))
    FP=np.sum((1-y_true_f)*y_pred_f)
    FN=np.sum(y_true_f*(1-y_pred_f))
    sensitivity=(TP)/((TP+FN))
    return sensitivity


def dice_coef(y_true, y_pred):
    smooth = 1
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection +smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) +smooth)


# losses
def iou_loss(y_true, y_pred):

    return 1 - iou_accuracy(y_true, y_pred)


def dice_coef_loss(y_true, y_pred):

    return 1 - dice_coef(y_true, y_pred)


# composed measure
def weighted_measures(y_true, y_pred, w1=.25, w2=.25, w3=.25, w4=.25):

    m1 = w1 * (1.0 - jaccard_distance(y_true, y_pred))
    m2 = w2 * specificity(y_true, y_pred)
    m3 = w3 * sensitivity(y_true, y_pred)
    m4 = w4 * dice_coef(y_true, y_pred)

    return m1 + m2 + m3 + m4


class WeightedMetric:

    def __init__(self, w_jac, w_dic, w_spe, w_sen):
        self.w_jac = w_jac
        self.w_dic = w_dic
        self.w_spe = w_spe
        self.w_sen = w_sen

    def execute_metric(self, y_true, y_pred):

        m1 = self.w_jac * (1 - jaccard_distance(y_true, y_pred))
        m4 = self.w_dic * dice_coef(y_true, y_pred)
        m2 = self.w_spe * specificity(y_true, y_pred)
        m3 = self.w_sen * sensitivity(y_true, y_pred)

        return m1 + m2 + m3 + m4

    def execute_loss(self, y_true, y_pred):

        return 1 - self.execute_metric(y_true, y_pred)

    def get_metric(self):

        return self.execute_metric

    def get_loss(self):

        return self.execute_loss


def f1_loss(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    TP = np.sum(y_true_f*y_pred_f)
    TN = np.sum((1-y_true_f)*(1-y_pred_f))
    FP = np.sum((1-y_true_f)*y_pred_f)
    FN = np.sum(y_true_f*(1-y_pred_f))

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = (2 * precision * recall) / (precision + recall)
    return F1
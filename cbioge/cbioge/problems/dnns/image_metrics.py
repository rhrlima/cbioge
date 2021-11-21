import keras.backend as K


# distances
def iou_accuracy(y_true, y_pred):
    intersection = y_true * y_pred
    union = y_true + ((1. - y_true) * y_pred)
    return K.sum(intersection) / K.sum(union)


def jaccard_distance(y_true, y_pred, smooth=100):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return 1-((1 - jac) * smooth)


def specificity(y_true,y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    tn_ = K.sum((1-y_true_f)*(1-y_pred_f))
    fp_ = K.sum((1-y_true_f)*y_pred_f)

    return (tn_)/(tn_+fp_)


def sensitivity(y_true,y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    tp_ = K.sum(y_true_f*y_pred_f)
    fn_ = K.sum(y_true_f*(1-y_pred_f))

    return (tp_)/((tp_+fn_))


def dice_coef(y_true, y_pred):
    smooth = 1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection +smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) +smooth)


#losses
def iou_loss(y_true, y_pred):

    return 1 - iou_accuracy(y_true, y_pred)


def dice_coef_loss(y_true, y_pred):

    return 1 - dice_coef(y_true, y_pred)


#composed measure
def weighted_measures(y_true, y_pred, w1=.3, w2=.05, w3=.35, w4=.3):

    return w1 * (1 - jaccard_distance(y_true, y_pred)) \
         + w2 * specificity(y_true, y_pred) \
         + w3 * sensitivity(y_true, y_pred) \
         + w4 * dice_coef(y_true, y_pred)


def weighted_measures_loss(y_true, y_pred, w1=.3, w2=.05, w3=.35, w4=.3):

    return 1 - weighted_measures(y_true, y_pred, w1, w2, w3, w4)


class WeightedMetric:

    def __init__(self, w_jac=.25, w_dic=.25, w_spe=.25, w_sen=.25):
        self.w_jac = w_jac
        self.w_dic = w_dic
        self.w_spe = w_spe
        self.w_sen = w_sen

    def __str__(self):
        return 'weighted_metric'

    def execute_metric(self, y_true, y_pred):

        return self.w_jac * (1 - jaccard_distance(y_true, y_pred)) \
             + self.w_dic * dice_coef(y_true, y_pred) \
             + self.w_spe * specificity(y_true, y_pred) \
             + self.w_sen * sensitivity(y_true, y_pred)

    def execute_loss(self, y_true, y_pred):

        return 1 - self.execute_metric(y_true, y_pred)

    def get_metric(self):

        return self.execute_metric

    def get_loss(self):

        return self.execute_loss

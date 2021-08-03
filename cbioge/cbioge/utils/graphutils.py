from __future__ import print_function
import networkx as nx
import scipy.sparse as sp
import numpy as np
from scipy.sparse.linalg.eigen.arpack import eigsh, ArpackNoConvergence
from spektral.utils.io import load_binary
import logging, os
import pandas as pd
import codecs
from spektral.utils.convolution import localpooling_filter

logging.basicConfig(level=logging.INFO)

def calculate_output_size(img_shape, k, s, p):
    ''' width, height, kernel, padding, stride'''
    index = 1 if len(img_shape) == 4 else 0
    w = img_shape[index]
    h = img_shape[index+1]

    p = 0 if p == 'valid' else (k-1) / 2
    ow = ((w - k + 2 * p) // s) + 1
    oh = ((h - k + 2 * p) // s) + 1
    return (int(ow), int(oh))


def preprocess(self, first_layer, A):
    fltr = A.astype("f4")
    if first_layer in [GraphSageConv, GraphAttention, GINConv, GatedGraphConv, TAGConv]:
        logging.info("no preprocessing")
        fltr = A.astype('f4')
    if first_layer in [GraphConv, ChebConv, ARMAConv, GraphConvSkip]:
        logging.info("preprocessing like framework")
        fltr = first_layer.preprocess(A).astype('f4')

    if first_layer in [APPNP]:
        logging.info("using localpooling_filter")
        fltr = localpooling_filter(A).astype('f4')
    return fltr

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

def _parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def _sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask)
    
def load_data(dataset_name='cora', DATA_PATH="", normalize_features=True, random_split=False):
   
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'adj', 'features', 'labels']
    objects = []
    data_path = DATA_PATH

    for n in names:
        filename = os.path.join(data_path, 'ind.{}.{}'.format(dataset_name, n))
        objects.append(load_binary(filename))

    x, y, tx, ty, allx, ally, graph, features, labels = tuple(objects)
    
    adj = graph
    
    _aux = len(x)

    train_idx = int(_aux*.9)
    test_size = len(tx)

    train_idxs = x[:train_idx]
    val_idxs = x[train_idx:]
    test_idxs = tx

    all_idxs = allx+tx
    all_data_size = len(all_idxs)

    label_size = len(labels[0])

    zero_labels = make_zero_labels(all_data_size-len(labels), label_size)

    labels = np.array(labels+zero_labels)
    
    logging.info(f":: Dataset size: {len(x)+len(tx)}")
    logging.info(f":: X size: {len(x)}")
    logging.info(f":: Train size: {len(train_idxs)}")
    logging.info(f":: AllX size: {len(allx)}")
    logging.info(f":: Validation size: {len(val_idxs)}")
    logging.info(f":: Test size: {len(test_idxs)}")
    logging.info(f":: Labels size: {labels.shape}")
    logging.info(f":: All data size: {all_data_size}")
    logging.info(f":: Feature size: {features.shape}")
    logging.info(f":: Adj size: {adj.shape}")
    

        
    # Row-normalize the features
    if normalize_features:
        logging.info(':: Pre-processing node features')
        features = preprocess_features(features)

    train_mask = _sample_mask(train_idxs, all_data_size)
    val_mask = _sample_mask(val_idxs, all_data_size)
    test_mask = _sample_mask(test_idxs, all_data_size)

    logging.info(f":: Train mask size: {train_mask.shape}")
    logging.info(f":: Val mask size: {val_mask.shape}")
    logging.info(f":: Test mask size: {test_mask.shape}")
    
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    
    return adj, features, labels, train_mask, val_mask, test_mask



def normalize_adj(adj, symmetric=True):
    if symmetric:
        d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
        a_norm = adj.dot(d).transpose().dot(d).tocsr()
    else:
        d = sp.diags(np.power(np.array(adj.sum(1)), -1).flatten(), 0)
        a_norm = d.dot(adj).tocsr()
    return a_norm


def preprocess_features(features):
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features

def preprocess_adj(adj, symmetric=True):
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize_adj(adj, symmetric)
    return adj


def sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def categorical_crossentropy(preds, labels):
    return np.mean(-np.log(np.extract(labels, preds)))


def accuracy(preds, labels):
    return np.mean(np.equal(np.argmax(labels, 1), np.argmax(preds, 1)))


def evaluate_preds(preds, labels, indices):

    split_loss = list()
    split_acc = list()

    for y_split, idx_split in zip(labels, indices):
        split_loss.append(categorical_crossentropy(preds[idx_split], y_split[idx_split]))
        split_acc.append(accuracy(preds[idx_split], y_split[idx_split]))

    return split_loss, split_acc


def normalized_laplacian(adj, symmetric=True):
    adj_normalized = normalize_adj(adj, symmetric)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    return laplacian


def rescale_laplacian(laplacian):
    try:
        #print('Calculating largest eigenvalue of normalized graph Laplacian...')
        largest_eigval = eigsh(laplacian, 1, which='LM', return_eigenvectors=False)[0]
    except ArpackNoConvergence:
        #print('Eigenvalue calculation did not converge! Using largest_eigval=2 instead.')
        largest_eigval = 2

    scaled_laplacian = (2. / largest_eigval) * laplacian - sp.eye(laplacian.shape[0])
    return scaled_laplacian


def chebyshev_polynomial(X, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices."""
    #print("Calculating Chebyshev polynomials up to order {}...".format(k))

    T_k = list()
    T_k.append(sp.eye(X.shape[0]).tocsr())
    T_k.append(X)

    def chebyshev_recurrence(T_k_minus_one, T_k_minus_two, X):
        X_ = sp.csr_matrix(X, copy=True)
        return 2 * X_.dot(T_k_minus_one) - T_k_minus_two

    for i in range(2, k+1):
        T_k.append(chebyshev_recurrence(T_k[-1], T_k[-2], X))

    return T_k


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


from keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def make_zero_labels(size, label_size):
    zero_labels = []
    for i in range(size):
        one_hot = [0 for l in range(label_size)]
        zero_labels.append(one_hot)
    return zero_labels
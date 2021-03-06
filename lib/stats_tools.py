import numpy as np
import tensorflow as tf
from keras import backend as K

def dice_coef(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def f1_score(y_true, y_pred, beta=1.0, smooth=1e-8):
    preds = tf.cast(tf.greater_equal(y_pred, 0.5), dtype=tf.float32)
    tp = tf.cast(tf.reduce_sum(preds * y_true), dtype=tf.float32)
    fp = tf.reduce_sum(tf.cast(tf.greater(preds, y_true), dtype=tf.float32))
    fn = tf.reduce_sum(tf.cast(tf.less(preds, y_true), dtype=tf.float32))
    b2 = beta * beta
    top = (1. + b2) * tp
    bottom = top + (b2 * fn) + fp
    return ((top + smooth)/(bottom + smooth))

def f2_score(y_true, y_pred, **kwargs):
    return f1_score(y_true, y_pred, beta=2.0, **kwargs)

def f05_score(y_true, y_pred, **kwargs):
    return f1_score(y_true, y_pred, beta=0.5, **kwargs)

def get_f_stats(predictions, labels, 
                true_positives=0, false_positives=0,
                false_negatives=0, true_negatives=0):
    preds = np.argmax(predictions, axis=3)
    not_preds = np.logical_not(preds)
    not_labels = np.logical_not(labels)
    tp = (preds * labels).sum()
    fp = (preds * not_labels).sum()
    fn = (not_preds * labels).sum()
    tn = (not_preds * not_labels).sum()
    assert labels.size == tp + fp + fn + tn, "Total did not sum up!"
    return (true_positives + tp, false_positives + fp, 
            false_negatives + fn, true_negatives + tn)

def f_beta_score(y_pred, y_true, beta=1.0, smooth=1e-8):
    """Returns F measure using tensorflow"""

    preds = tf.argmax(y_pred, 3)
    tp = tf.cast(tf.reduce_sum(preds * y_true), dtype=tf.float32)
    fp = tf.reduce_sum(tf.cast(tf.greater(preds, y_true), dtype=tf.float32))
    fn = tf.reduce_sum(tf.cast(tf.less(preds, y_true), dtype=tf.float32))
    b2 = beta * beta
    top = (1. + b2) * tp
    bottom = top + (b2 * fn) + fp
    return ((top + smooth)/(bottom + smooth))

def np_f_beta_score(true_positives=0, false_positives=0, 
                    false_negatives=0, beta=1.0, smooth=1e-8):
    """Returns F measure using numpy"""
    
    b2 = beta * beta
    top = (1. + b2) * true_positives
    bottom = top + (b2 * false_negatives) + false_positives
    return ((top + smooth)/(bottom + smooth))

def print_data_stats(set_name, x, y):
    print("\n{}:".format(set_name))
    print("    ... x: {}, dtype: {}".format(x.shape, x.dtype))
    print("           max: {}, min: {}".format(x.max(), x.min()))
    print("           mean: {}, std: {}".format(x.mean(), x.std()))
    print("\n    ... y: {}, dtype: {}".format(y.shape, y.dtype))
    print("           max: {}, min: {}".format(y.max(), y.min()))

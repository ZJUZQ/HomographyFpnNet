#coding=utf-8
from tensorflow.keras import backend as K

def mean_corner_error_4pts(y_true, y_pred):
    # mean pixel distance err for pt
    y_true = K.reshape(y_true, (-1, 4, 2))
    y_pred = K.reshape(y_pred, (-1, 4, 2))
    return K.mean(K.sqrt( K.sum(K.square(y_pred - y_true), axis=-1, keepdims=True)), axis=1)

def mean_corner_error_9pts(y_true, y_pred):
    # mean pixel distance err for pt
    y_true = K.reshape(y_true, (-1, 9, 2))
    y_pred = K.reshape(y_pred, (-1, 9, 2))
    return K.mean(K.sqrt( K.sum(K.square(y_pred - y_true), axis=-1, keepdims=True)), axis=1)
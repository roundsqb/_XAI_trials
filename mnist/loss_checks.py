import numpy as np

from keras import backend as K
from tensorflow.python.ops import math_ops
import tensorflow as tf

_EPSILON = K.epsilon()

def _loss_tensor(y_true, y_pred):
    y_pred = K.clip(y_pred, _EPSILON, 1.0-_EPSILON)
    out =K.sum(y_true * y_pred, axis=len(y_pred.get_shape())-1)

    return -K.log(1. - out)

def _loss_np(y_true, y_pred):

    y_pred = np.clip(y_pred, _EPSILON, 1.0-_EPSILON)
    
    out = np.sum(y_true * y_pred, axis=1)

    neg = 1.- out

    final = -np.log(neg)

    return final

def check_loss(_shape):
    if _shape == '2d':
        shape = (100, 10)

    y_a = np.zeros(shape)
    for i, j in enumerate(np.random.randint(shape[1], size=shape[0])):
        y_a[i,j] = 1.0
    y_b = np.random.random(shape)

    out1 = K.eval(_loss_tensor(K.variable(y_a), K.variable(y_b)))
    out2 = _loss_np(y_a, y_b)
    out3 = K.eval(categorical_inaccuracy(K.variable(y_a), K.variable(y_b)))
    out4 = K.eval(categorical_accuracy(K.variable(y_a), K.variable(y_b)))

    assert out1.shape == out2.shape
    assert out1.shape == shape[:-1]

    print(out3)
    print(out4)
    print(np.linalg.norm(out1))
    print(np.linalg.norm(out2))
    print(np.linalg.norm(out1-out2))

def categorical_accuracy(y_true, y_pred):
    return K.mean(K.cast(K.equal(K.argmax(y_true, axis=-1),
                          K.argmax(y_pred, axis=-1)),
                  K.floatx()))

def categorical_inaccuracy(y_true, y_pred):
    return K.mean(K.cast(K.not_equal(K.argmax(y_true, axis=-1),
                          K.argmax(y_pred, axis=-1)),
                  K.floatx()))

def test_loss():
    shape_list = ['2d']
    for _shape in shape_list:
        check_loss(_shape)
        print('======================')


if __name__ == '__main__':
    sess = tf.InteractiveSession()
    test_loss()
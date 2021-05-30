from tensorflow.keras import backend as K
import numpy as np
import tensorflow as tf

def weighted_log_loss(y_true, y_pred):
    """
    Can be used as the loss function in model.compile()
    ---------------------------------------------------
    """
    
    class_weights =  tf.constant([2., 1., 1., 1., 1., 1.])
    
    eps = tf.keras.backend.epsilon()
    
    
    y_pred = tf.clip_by_value(y_pred, eps, 1.0-eps)

    out = -(         y_true  * tf.math.log(      y_pred) * class_weights
            + (1.0 - y_true) * tf.math.log(1.0 - y_pred) * class_weights)
    
    return tf.reduce_mean(out, axis=-1)


####
def weighted_log_loss_V2(y_true, y_pred):
    """
    Can be used as the loss function in model.compile()
    ---------------------------------------------------
    """
    
    class_weights =  tf.constant([2., 1., 1., 1., 1., 1.])
    
    eps = tf.keras.backend.epsilon()
    
    
    y_pred = tf.clip_by_value(y_pred, eps, 1.0-eps)

    out = -(         y_true  * tf.math.log(      y_pred) * class_weights
            + (1.0 - y_true) * tf.math.log(1.0 - y_pred) * class_weights)
    
    return tf.reduce_mean(out, axis=-1)
####

def _normalized_weighted_average(arr, weights=None):
    """
    A simple Keras implementation that mimics that of 
    numpy.average(), specifically for this competition
    """
    
    if weights is not None:
        scl = K.sum(weights)
        weights = K.expand_dims(weights, axis=1)
        return K.sum(K.dot(arr, weights), axis=1) / scl
    return K.mean(arr, axis=1)


def weighted_loss(y_true, y_pred):
    """
    Will be used as the metric in model.compile()
    ---------------------------------------------
    
    Similar to the custom loss function 'weighted_log_loss()' above
    but with normalized weights, which should be very similar 
    to the official competition metric:
        https://www.kaggle.com/kambarakun/lb-probe-weights-n-of-positives-scoring
    and hence:
        sklearn.metrics.log_loss with sample weights
    """
    
    class_weights = tf.constant([2., 1., 1., 1., 1., 1.])
    
    eps = tf.keras.backend.epsilon()
    
    y_pred = tf.clip_by_value(y_pred, eps, 1.0-eps)

    loss = -(        y_true  * tf.math.log(      y_pred)
            + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
    
    loss_samples = _normalized_weighted_average(loss, class_weights)
    
    return tf.reduce_mean(loss_samples)


def weighted_log_loss_metric(trues, preds):
    """
    Will be used to calculate the log loss 
    of the validation set in PredictionCheckpoint()
    ------------------------------------------
    """
    class_weights = [2., 1., 1., 1., 1., 1.]
    
    epsilon = 1e-7
    
    preds = np.clip(preds, epsilon, 1-epsilon)
    loss = trues * np.log(preds) + (1 - trues) * np.log(1 - preds)
    loss_samples = np.average(loss, axis=1, weights=class_weights)

    return - loss_samples.mean()
'''
def log_loss(y_pred,y_true):
    #y_pred = np.clip(y_pred, 1e-7, 1.0-1e-7)
    ones = np.where(y_true)
    zeros = np.where(y_true == 0)
    likelihood = np.prod(y_pred[ones]) *  np.prod(y_pred[zeros])
    logloss = -1 * np.log(likelihood)
    return logloss
'''

def log_loss(y_pred,y_true):
    """
    Will be used to calculate the log loss 
    of the validation set in metrics
    ------------------------------------------
    """
    eps = 1e-15
    
    y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
    
    ones = tf.where(y_true)
    zeros = tf.where(y_true == 0)
    
    likelihood = tf.math.reduce_prod(tf.gather(y_pred,ones)) *  tf.math.reduce_prod(1 - tf.gather(y_pred,zeros))
    logloss = -1 * tf.math.log(likelihood)
    return logloss



import keras.backend as K
import numpy as np

def weighted_bincrossentropy(true, pred, weight_zero = 0.25, weight_one = 0.85):
    # this is the weighted binary cross entropy, the weights are determined from the distrobution in the test set
    binary_crossentropy = K.binary_crossentropy(true, pred)
    weights = true * weight_one + (1. - true) * weight_zero
    weighted_binary_crossentropy = weights * binary_crossentropy 
    return K.mean(weighted_binary_crossentropy)

def DiceLoss(y_true, y_pred):
    # this is the weighted dice + BCE method used for training
    smooth = 1.
    y_true_f = K.abs(K.flatten(y_true))
    y_pred_f = K.abs(K.flatten(y_pred))
    intersection = K.sum(y_true_f * y_pred_f)
    dice = ((2. * intersection + smooth) / ((K.sum(y_true_f) + K.sum(y_pred_f) + smooth)))
    return 0.1*weighted_bincrossentropy(y_true,y_pred) + 1.-dice

def pure_dice(y_true, y_pred):
    # pure dice loss
    smooth = 1.
    y_true_f = K.abs(K.flatten(y_true))
    y_pred_f = K.abs(K.flatten(y_pred))
    intersection = K.sum(y_true_f * y_pred_f)
    dice = ((2. * intersection + smooth) / ((K.sum(y_true_f) + K.sum(y_pred_f) + smooth)))
    return 1. - dice

def testMag(y_true, y_pred):
    # just used to debug issues with model converging to all zeors
    return K.mean(y_pred)

def DiceCoef(y_true, y_pred):
    # Generalized dice loss, is Dice coefficient if prediction is binary
    smooth = 1e-5
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)

def IOUCoef(y_true, y_pred):
    # IoU loss, is IOU coefficient if prediction is binary
    smooth = 1.
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    return ( intersection + smooth) / (union + smooth)
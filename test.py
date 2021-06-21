from imgaug.augmenters import color
from dataGenerator import * 
from model import *
from wavelet import *
from keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import Adam
import keras.backend as K
from keras.models import load_model

import tensorflow as tf
import datetime
import numpy as np
import matplotlib.pyplot as plt

config = {
"batchSize" : 1,
"lr" : 1e-3,
"wavelet":False,   
"deepSupervision" : False,
}

def DiceLoss(y_true, y_pred, smooth=1e-6):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    dice = (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)
    return 1 - dice

def JaccardLoss(y_true, y_pred, smooth=100):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth

dataGenTest = DataGeneratorSIIM(config["batchSize"], train_type=1, wavelet=config["wavelet"], deepSupervision=config["deepSupervision"])

if config["wavelet"]:
    model = UNetPlusPlus(256,256,color_type=12, deep_supervision=config["deepSupervision"])
else:
    model = UNetPlusPlus(256,256,color_type=3,  deep_supervision=config["deepSupervision"])
print(model.summary())

model = load_model("./Models/Loss_Dice_wavelet_True_lr_0.001_batch_12_DeSu_False_t_2021_06_20_11_12_epoch-046-0.856850-0.817176.h5", custom_objects={"DiceLoss":DiceLoss,"JaccardLoss":JaccardLoss})

out = model.predict_generator(dataGenTest, 10)

plt.imshow(out[0,:,:,0])

plt.show()
from imgaug.augmenters import color
from dataGenerator import * 
from model import *
from keras import *
import keras.backend as K

batchSize = 8
lr = 1e-4

def DiceLoss(y_true, y_pred, smooth=1e-6):
    
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    dice = (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)
    return 1 - dice

def JaccardLoss(y_true, y_pred, smooth=100):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth

dataGenTrain = DataGeneratorSIIM(batchSize,train_type=0)
dataGenTest  = DataGeneratorSIIM(batchSize,train_type=1)

model = UNetPlusPlus(256,256,color_type=3)

print(model.summary())

model.compile(optimizer=Adam(lr=lr), loss=DiceLoss, metrics=[JaccardLoss, DiceLoss])#, 'binary_crossentropy', 'mse'

model.fit_generator(dataGenTrain, validation_data=dataGenTest, epochs=25, callbacks=[], workers=8)
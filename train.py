from imgaug.augmenters import color
from dataGenerator import * 
from model import *
from keras.callbacks import TensorBoard, ReduceLROnPlateau
from keras.utils import multi_gpu_model
import keras.backend as K

batchSize = 16
lr = 1e-4
wavelet=True

def DiceLoss(y_true, y_pred, smooth=1e-6):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    dice = (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)
    return 1 - dice

def JaccardLoss(y_true, y_pred, smooth=100):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth

dataGenTrain = DataGeneratorSIIM(batchSize,train_type=0, wavelet=wavelet)
dataGenTest  = DataGeneratorSIIM(batchSize,train_type=1, wavelet=wavelet)

model = UNetPlusPlus(256,256,color_type=12)

print(model.summary())

tb = TensorBoard(log_dir='./Logs', batch_size=batchSize, write_graph=False)
rl = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)

multi_model = multi_gpu_model(model, gpus=2)

multi_model.compile(optimizer=Adam(lr=lr), loss=DiceLoss, metrics=[JaccardLoss, DiceLoss, 'binary_crossentropy', 'mse'])#

multi_model.fit_generator(dataGenTrain, validation_data=dataGenTest, epochs=100, callbacks=[tb,rl])
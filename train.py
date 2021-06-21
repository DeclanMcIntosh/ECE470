from dataGenerator import * 
from model import *
from keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from keras.losses import binary_crossentropy
import keras.backend as K
import datetime
import numpy as np
import tensorflow as tf

config = {
"batchSize" : 20,
"lr" : 3e-4,
"wavelet":True,   
"deepSupervision" : False,
"dataAugs" : True,
}
LogDescription = "Loss_BCE&Dice_wavelet_"+str(config["wavelet"])+ "_Augs_" + str(config["dataAugs"])+ "_DeepSu_" + str(config["deepSupervision"])+"_lr_" +str(config["lr"])+"_batch_" + str(config["batchSize"])+   "_t_" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M") 

def DiceLoss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 0.5*binary_crossentropy(y_true, y_pred) - (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def DiceCoef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

def testMag(y_true, y_pred):
    return K.mean(y_pred)

dataGenTrain = DataGeneratorSIIM(config["batchSize"], train_type=0, wavelet=config["wavelet"], deepSupervision=config["deepSupervision"], dataAugs=config["dataAugs"])
dataGenVal   = DataGeneratorSIIM(config["batchSize"], train_type=1, wavelet=config["wavelet"], deepSupervision=config["deepSupervision"], dataAugs=config["dataAugs"])

tb = TensorBoard(log_dir="./Logs/"+LogDescription, batch_size=config["batchSize"], write_graph=False)
mc = ModelCheckpoint("./Logs/"+LogDescription+"/_epoch-{epoch:03d}-{val_loss:03f}-{val_acc:03f}.h5", save_best_only=True, verbose=1)
es = EarlyStopping(patience=4, verbose=1)

if config["wavelet"]:
    model = UNetPlusPlus(256,256,color_type=12, deep_supervision=config["deepSupervision"])
else:
    model = UNetPlusPlus(256,256,color_type=3,  deep_supervision=config["deepSupervision"])
print(model.summary())

multi_model = multi_gpu_model(model, gpus=2)

multi_model.compile(optimizer=Adam(lr=config["lr"]), loss=DiceLoss, metrics=[testMag, 'acc'])

multi_model.fit_generator(dataGenTrain, validation_data=dataGenVal, epochs=25, callbacks=[tb, mc])
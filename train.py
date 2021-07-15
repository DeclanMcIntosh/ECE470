# author: Declan McIntosh
# contact: contact@declanmcintosh.com

from dataGenerator import * 
from model import *
from loss import *
from keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from keras.losses import binary_crossentropy
import keras.backend as K
import datetime

def train(config):
    LogDescription = "Res_Loss_NewWeightedBCE&Dice_wavelet_"+str(config["wavelet"])+ "_Augs_" + str(config["dataAugs"])+ "_DeepSu_" + str(config["deepSupervision"])+"_lr_" +str(config["lr"])+"_batch_" + str(config["batchSize"])+   "_t_" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M") 

    def weighted_bincrossentropy(true, pred, weight_zero = 0.25, weight_one = 0.85):
        binary_crossentropy = K.binary_crossentropy(true, pred)
        weights = true * weight_one + (1. - true) * weight_zero
        weighted_binary_crossentropy = weights * binary_crossentropy 
        return K.mean(weighted_binary_crossentropy)

    def DiceLoss(y_true, y_pred):
        smooth = 1.
        y_true_f = K.abs(K.flatten(y_true))
        y_pred_f = K.abs(K.flatten(y_pred))
        intersection = K.sum(y_true_f * y_pred_f)
        dice = ((2. * intersection + smooth) / ((K.sum(y_true_f) + K.sum(y_pred_f) + smooth)))
        #return 1.- dice
        #return 1. - dice
        return 0.1*weighted_bincrossentropy(y_true,y_pred) + 1.-dice


    def pure_dice(y_true, y_pred):
        smooth = 1.
        y_true_f = K.abs(K.flatten(y_true))
        y_pred_f = K.abs(K.flatten(y_pred))
        intersection = K.sum(y_true_f * y_pred_f)
        dice = ((2. * intersection + smooth) / ((K.sum(y_true_f) + K.sum(y_pred_f) + smooth)))
        #return 1.- dice
        return 1. - dice


    dataGenTrain = DataGeneratorSIIM(config["batchSize"], train_type=0, wavelet=config["wavelet"], deepSupervision=config["deepSupervision"], dataAugs=config["dataAugs"])
    dataGenVal   = DataGeneratorSIIM(config["batchSize"], train_type=1, wavelet=config["wavelet"], deepSupervision=config["deepSupervision"], dataAugs=config["dataAugs"])

    if config["wavelet"]:
        model = ResUnetPlusPlus(256,256,color_type=12, deep_supervision=config["deepSupervision"])
    else: 
        model = ResUnetPlusPlus(256,256,color_type=3,  deep_supervision=config["deepSupervision"])

    if  config["pretrained"] :
        model.load_weights(config["pretrained"], by_name=True)
        LogDescription = "PreTrained_"+LogDescription

    tb = TensorBoard(log_dir="./Logs/"+LogDescription, batch_size=config["batchSize"], write_graph=False)
    if config["deepSupervision"]:
        mc = ModelCheckpoint("./Logs/"+LogDescription+"/_epoch-{epoch:03d}-{val_output_4_loss:03f}-{val_output_4_acc:03f}.h5", save_best_only=True, verbose=1)
    else:
        mc = ModelCheckpoint("./Logs/"+LogDescription+"/_epoch-{epoch:03d}-{val_loss:03f}-{val_acc:03f}.h5", save_best_only=True, verbose=1)
    rl = ReduceLROnPlateau(patience=5, factor=0.1, verbose=1, min_lr=1e-6)
    es = EarlyStopping(patience=10, verbose=1)

    #print(model.summary())

    multi_model = multi_gpu_model(model, gpus=2)

    if config["deepSupervision"]:
        multi_model.compile(optimizer=Adam(lr=config["lr"]), loss=DiceLoss, metrics=[binary_crossentropy, pure_dice, 'acc'],  loss_weights={"output_1": 1.0, "output_1": 1.0, "output_1": 1.0, "output_1": 1.0})
    else:  
        multi_model.compile(optimizer=Adam(lr=config["lr"]), loss=DiceLoss, metrics=[binary_crossentropy, pure_dice, 'acc'])

    multi_model.fit_generator(dataGenTrain, validation_data=dataGenVal, epochs=100, callbacks=[tb, mc, es, rl])

config = {
"batchSize" : 16,
"lr" : 1e-4,
"wavelet":False,   
"deepSupervision" : False,
"dataAugs" : True,
"pretrained" : False
}

train(config=config)  
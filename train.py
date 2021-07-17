# author: Declan McIntosh
# contact: contact@declanmcintosh.com

from dataGenerator import * 
from model import *
from loss import *
from keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from keras.losses import binary_crossentropy
import datetime

def train(config):
    LogDescription = "Res_Loss_NewWeightedBCE&Dice_wavelet_"+str(config["wavelet"])+ "_Augs_" + str(config["dataAugs"])+"_lr_" +str(config["lr"])+"_batch_" + str(config["batchSize"])+   "_t_" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M") 
    # init data generator
    dataGenTrain = DataGeneratorSIIM(config["batchSize"], train_type=0, wavelet=config["wavelet"], dataAugs=config["dataAugs"])
    dataGenVal   = DataGeneratorSIIM(config["batchSize"], train_type=1, wavelet=config["wavelet"], dataAugs=config["dataAugs"])

    # configure and load model
    if config["wavelet"]:
        model = ResUnetPlusPlus(256,256,color_type=12)
    else: 
        model = ResUnetPlusPlus(256,256,color_type=3)


    # build our callbacks
    tb = TensorBoard(log_dir="./Logs/"+LogDescription, batch_size=config["batchSize"], write_graph=False)
    mc = ModelCheckpoint("./Logs/"+LogDescription+"/_epoch-{epoch:03d}-{val_loss:03f}-{val_acc:03f}.h5", save_best_only=True, verbose=1)
    rl = ReduceLROnPlateau(patience=5, factor=0.1, verbose=1, min_lr=1e-6)
    es = EarlyStopping(patience=10, verbose=1)

    print(model.summary())

    #multi_model = multi_gpu_model(model, gpus=2) # This is removed as it is not assumed most people have multiple gpus to reporduce results

    # compile our model with metrics, loss, and optimizer
    model.compile(optimizer=Adam(lr=config["lr"]), loss=DiceLoss, metrics=[binary_crossentropy, pure_dice, 'acc'])

    # fit the model to the data generator
    model.fit_generator(dataGenTrain, validation_data=dataGenVal, epochs=100, callbacks=[tb, mc, es, rl])


for x in range(10):
    config = {
    "batchSize" : 16,
    "lr" : 1e-4,
    "wavelet":False,
    "dataAugs" : False
    }

    train(config=config)  

    config = {
    "batchSize" : 16,
    "lr" : 1e-4,
    "wavelet":False,
    "dataAugs" : True
    }

    train(config=config)  

    config = {
    "batchSize" : 16,
    "lr" : 1e-4,
    "wavelet":True,
    "dataAugs" : True
    }

    train(config=config)  
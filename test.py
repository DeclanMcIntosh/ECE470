# author: Declan McIntosh
# contact: contact@declanmcintosh.com

from dataGenerator import * 
from model import *
from wavelet import *
from loss import *
import keras.backend as K
from keras.models import load_model
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error

import tensorflow as tf
import numpy as np

import time

config = {
"batchSize" : 1,
"lr" : 1e-3,
"wavelet":True,   
"deepSupervision" : False,
}


def test(model):
    # build structures for reporting values over entire test set
    meanIOU = []
    IOU = []
    meanDICE = []
    DICE = []
    ACC = []
    MSE = []
    Infer = []

    # make test generator with train_type=2
    dataGenTest = DataGeneratorSIIM(config["batchSize"], train_type=2, wavelet=config["wavelet"], deepSupervision=config["deepSupervision"])
    
    for x in range(dataGenTest.__len__()):
        # for each item in the test set predict and determine metrics
        start = time.time()

        inputVal, trueVal =  dataGenTest.__getitem__(x)

        predVal = model.predict(inputVal)

        Infer.append(1000*abs(start-time.time()))

        predValCts = predVal.copy()
        predVal2 = predVal.copy()
        trueVal2 = trueVal.copy()

        predVal[predVal2>0.5] = 1 
        predVal[predVal2<=0.5] = 0 

        predVal2[predVal2<0.5] = 1 
        predVal2[predVal2>=0.5] = 0 

        trueVal2[trueVal2<0.5] = 1 
        trueVal2[trueVal2>=0.5] = 0 

        DICE.append(DiceCoef(trueVal,predVal))
        meanDICE.append((DiceCoef(trueVal,predVal)+DiceCoef(trueVal2,predVal2))/2)
        ACC.append(accuracy_score(trueVal.flatten(),predVal.flatten()))
        meanIOU.append((IOUCoef(trueVal,predVal)+IOUCoef(trueVal2,predVal2))/2)
        IOU.append(IOUCoef(trueVal,predVal))
        MSE.append(mean_squared_error(trueVal.flatten(),predValCts.flatten()))





    print("Mean Dice: ", np.mean(np.array(meanDICE)))
    print("Dice: ", np.mean(np.array(DICE)))
    print("Mean IOU: ",np.mean(np.array(meanIOU)))
    print("IOU: ",np.mean(np.array(IOU)))
    print("MSE: ", np.mean(np.array(MSE)))
    print("Accuracy: ",np.mean(np.array(ACC)))
    print("Inferance speed: ",np.mean(np.array(Infer[5:])))
    
    K.clear_session()

def get_flops():
    run_meta = tf.RunMetadata()
    opts = tf.profiler.ProfileOptionBuilder.float_operation()

    # We use the Keras session graph in the call to the profiler.
    flops = tf.profiler.profile(graph=K.get_session().graph,
                                run_meta=run_meta, cmd='op', options=opts)

    return flops.total_float_ops  # Prints the "flops" of the model.

# With wavelet preprocessing and with Data Augmentation
model = load_model("./Logs/Res_Loss_NewWeightedBCE&Dice_wavelet_True_Augs_True_lr_0.0001_batch_16_t_2021_07_16_20_02/_epoch-055-0.256651-0.935096.h5", custom_objects={"DiceLoss":DiceLoss,"testMag":testMag, "pure_dice":pure_dice}) # pretty good
print("With wavelet preprocessing and with Data Augmentation")
print(model.summary())
get_flops()
test(model)

# Without wavelet preprocessing and with Data Augmentation
model = load_model("./Logs/Res_Loss_NewWeightedBCE&Dice_wavelet_False_Augs_True_lr_0.0001_batch_16_t_2021_07_16_19_12/_epoch-033-0.305945-0.923208.h5", custom_objects={"DiceLoss":DiceLoss,"testMag":testMag, "pure_dice":pure_dice}) # pretty good
print("Without wavelet preprocessing and with Data Augmentation")
print(model.summary())
get_flops()
test(model)

# Without wavelet preprocessing and without Data Augmentation
model = load_model("./Logs/Res_Loss_NewWeightedBCE&Dice_wavelet_False_Augs_False_lr_0.0001_batch_16_t_2021_07_16_18_35/_epoch-022-0.363203-0.914916.h5", custom_objects={"DiceLoss":DiceLoss,"testMag":testMag, "pure_dice":pure_dice}) # pretty good
print("Without wavelet preprocessing and without Data Augmentation")
print(model.summary())
get_flops()
test(model)

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
import matplotlib.pyplot as plt

import time

config = {
"batchSize" : 1,
"lr" : 1e-3,
"wavelet":True,   
"deepSupervision" : False,
}


def test(model, Visualize = True):
    meanIOU = []
    IOU = []
    meanDICE = []
    DICE = []
    ACC = []
    MSE = []

    dataGenTest = DataGeneratorSIIM(config["batchSize"], train_type=2, wavelet=config["wavelet"], deepSupervision=config["deepSupervision"])

    for x in range(dataGenTest.__len__()):

        inputVal, trueVal =  dataGenTest.__getitem__(x)

        predVal = model.predict(inputVal)

        predValCts = predVal.copy()
        predVal2 = predVal.copy()
        trueVal2 = trueVal.copy()

        predVal[predVal2>0.5] = 1 
        predVal[predVal2<=0.5] = 0 

        predVal2[predVal2<0.5] = 1 
        predVal2[predVal2>=0.5] = 0 

        trueVal2[trueVal2<0.5] = 1 
        trueVal2[trueVal2>=0.5] = 0 

        if Visualize:
            fig = plt.figure()
            ax1 = fig.add_subplot(2,2,1)
            ax1.imshow(inputVal[0,:,:,0])
            ax2 = fig.add_subplot(2,2,2)
            ax2.imshow(np.concatenate([trueVal[0],trueVal[0],trueVal[0]],axis=2))
            ax3 = fig.add_subplot(2,2,3)
            ax3.imshow(np.concatenate([predValCts[0],predValCts[0],predValCts[0]],axis=2))
            ax4 = fig.add_subplot(2,2,4)
            ax4.imshow(np.concatenate([predVal[0],predVal[0],predVal[0]],axis=2))
            plt.show()


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
    
def get_flops():
    run_meta = tf.RunMetadata()
    opts = tf.profiler.ProfileOptionBuilder.float_operation()

    # We use the Keras session graph in the call to the profiler.
    flops = tf.profiler.profile(graph=K.get_session().graph,
                                run_meta=run_meta, cmd='op', options=opts)

    return flops.total_float_ops  # Prints the "flops" of the model.

model = load_model("./Logs/Res_Loss_NewWeightedBCE&Dice_wavelet_True_Augs_True_DeepSu_False_lr_0.0001_batch_16_t_2021_07_08_20_25/_epoch-055-0.276358-0.929860.h5", custom_objects={"DiceLoss":DiceLoss,"testMag":testMag, "pure_dice":pure_dice}) # pretty good

print(model.summary())

get_flops()

test(model)

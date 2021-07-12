# author: Declan McIntosh
# contact: contact@declanmcintosh.com

from dataGenerator import * 
from model import *
from wavelet import *
from loss import *
from keras.models import load_model

import numpy as np
import matplotlib.pyplot as plt

config = {
"batchSize" : 1,
"lr" : 1e-3,
"wavelet":True,   
"deepSupervision" : False,
}


def demo(model):

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




        RGB = ((np.concatenate([np.expand_dims(inputVal[0,:,:,2],axis=2),np.expand_dims(inputVal[0,:,:,1],axis=2),np.expand_dims(inputVal[0,:,:,0],axis=2)], axis=2)/2.+0.5))

        fig = plt.figure()
        ax1 = fig.add_subplot(2,2,1)
        ax1.imshow(RGB)
        ax2 = fig.add_subplot(2,2,2)
        ax2.imshow(np.concatenate([trueVal[0],trueVal[0],trueVal[0]],axis=2))
        ax3 = fig.add_subplot(2,2,3)
        ax3.imshow(np.concatenate([predValCts[0],predValCts[0],predValCts[0]],axis=2))
        ax4 = fig.add_subplot(2,2,4)
        ax4.imshow(np.concatenate([predVal[0],predVal[0],predVal[0]],axis=2))
        plt.show()


model = load_model("./Logs/Res_Loss_NewWeightedBCE&Dice_wavelet_True_Augs_True_DeepSu_False_lr_0.0001_batch_16_t_2021_07_08_20_25/_epoch-055-0.276358-0.929860.h5", custom_objects={"DiceLoss":DiceLoss,"testMag":testMag, "pure_dice":pure_dice}) # pretty good

demo(model)

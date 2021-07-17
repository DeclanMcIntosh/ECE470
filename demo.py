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

    # test set generator
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
        
        # histogramEqualization for high frequency wavelet chanels to make them easier to visualize 
        def histEqual(img):
            img = 255*img
            img = img.astype(np.uint8)
            R, G, B = cv2.split(img)
            output1_R = cv2.equalizeHist(R)
            output1_G = cv2.equalizeHist(G)
            output1_B = cv2.equalizeHist(B)
            img = cv2.merge((output1_R, output1_G, output1_B))
            return img

        x =0
        RGB =   ((np.concatenate([np.expand_dims(inputVal[0,:,:,2+x],axis=2),np.expand_dims(inputVal[0,:,:,1+x],axis=2),np.expand_dims(inputVal[0,:,:,0+x],axis=2)], axis=2)/2.+0.5))
        x = 3
        RGB_HL = histEqual(((np.concatenate([np.expand_dims(inputVal[0,:,:,2+x],axis=2),np.expand_dims(inputVal[0,:,:,1+x],axis=2),np.expand_dims(inputVal[0,:,:,0+x],axis=2)], axis=2)/2.+0.5)))
        x = 6
        RGB_LH = histEqual(((np.concatenate([np.expand_dims(inputVal[0,:,:,2+x],axis=2),np.expand_dims(inputVal[0,:,:,1+x],axis=2),np.expand_dims(inputVal[0,:,:,0+x],axis=2)], axis=2)/2.+0.5)))
        RGB_LH = np.sqrt(((RGB_LH - np.min(RGB_LH))/(np.max(RGB_LH)-np.min(RGB_LH))))
        x = 9
        RGB_HH = histEqual(((np.concatenate([np.expand_dims(inputVal[0,:,:,2+x],axis=2),np.expand_dims(inputVal[0,:,:,1+x],axis=2),np.expand_dims(inputVal[0,:,:,0+x],axis=2)], axis=2)/2.+0.5)))
        RGB_HH = np.sqrt(((RGB_HH - np.min(RGB_HH))/(np.max(RGB_HH)-np.min(RGB_HH))))

        # plot wavelet
        fig = plt.figure()
        fig.suptitle("Wavelet Pre-processing")
        ax1 = fig.add_subplot(2,2,1)
        ax1.axis('off')
        ax1.title.set_text("RGB Image")
        ax1.imshow(RGB)
        ax2 = fig.add_subplot(2,2,2)
        ax2.axis('off')
        ax2.title.set_text("RGB Low Pass High Pass")
        ax2.imshow(RGB_HL)
        ax3 = fig.add_subplot(2,2,3)
        ax3.axis('off')
        ax3.title.set_text("RGB High Pass Low Pass")
        ax3.imshow(RGB_LH)
        ax4 = fig.add_subplot(2,2,4)
        ax4.axis('off')
        ax4.title.set_text("RGB High Pass High Pass")
        ax4.imshow(RGB_HH)
        plt.show()

        # plot prediction
        fig = plt.figure()
        fig.suptitle("Prediction")
        ax1 = fig.add_subplot(2,2,1)
        ax1.axis('off')
        ax1.title.set_text("RGB Image")
        ax1.imshow(RGB)
        ax2 = fig.add_subplot(2,2,2)
        ax2.axis('off')
        ax2.title.set_text("Mask Ground Truth")
        ax2.imshow(np.concatenate([trueVal[0],trueVal[0],trueVal[0]],axis=2))
        ax3 = fig.add_subplot(2,2,3)
        ax3.axis('off')
        ax3.title.set_text("Mask Confidences")
        ax3.imshow(np.concatenate([predValCts[0],predValCts[0],predValCts[0]],axis=2))
        ax4 = fig.add_subplot(2,2,4)
        ax4.axis('off')
        ax4.title.set_text("Final Mask Prediction")
        ax4.imshow(np.concatenate([predVal[0],predVal[0],predVal[0]],axis=2))
        plt.show()

# load the best model with wavelet pre-processing
model = load_model("./Logs/Res_Loss_NewWeightedBCE&Dice_wavelet_True_Augs_True_lr_0.0001_batch_16_t_2021_07_16_20_02/_epoch-055-0.256651-0.935096.h5", custom_objects={"DiceLoss":DiceLoss,"testMag":testMag, "pure_dice":pure_dice}) # pretty good

demo(model)

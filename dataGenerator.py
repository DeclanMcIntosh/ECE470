# name: LeGall 5-3 GPU 
# author: Declan McIntosh
# contact: contact@declanmcintosh.com
# paper: TBA

import numpy as np 
import cv2
import os
import keras
from random import shuffle


def generateDataReferancesKvasir(splits,train_type,dataDir="/Data/Kvasir-SEG/"):
    imgs   = list(os.listdir(dataDir+"images"))
    imgs   = [dataDir+"images/" + s for s in imgs] 
    masks  = list(os.listdir(dataDir+"masks"))
    masks  = [dataDir+"masks/" + s for s in masks] 

    start = 0 if train_type==0 else int(len(imgs)*splits[train_type-1])
    end =  int(len(imgs)*splits[train_type])

    return {"Raw":imgs[start:end], "Ann":masks[start:end]}

def getSampleKvasir(dataReferances, index, dim):
    img = cv2.resize(cv2.imread(dataReferances["Raw"][index]), dim).astype(np.float32)
    ann = cv2.resize(cv2.imread(dataReferances["Ann"][index]), dim).astype(np.float32)/255

    return img, ann


class DataGeneratorSIIM(keras.utils.Sequence):
    'Based loosely off of https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly'
    def __init__(self, data, batch_size, dim=(512,512), wavelet=False, train_type=0, splits=[0.7,1,1], channels = 3,\
        dataFinder=generateDataReferancesKvasir, dataPuller=getSampleKvasir): #
        'Initialization'
        self.train_type = train_type # 0 for train, 1 for validate
        
        self.channels = channels
        

        self.dataPuller = dataPuller
        self.dataReferances = dataFinder(splits,train_type)
        self.on_epoch_end()

    def __len__(self): 
        return len(self.dataReferances["Raw"])


    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X, Y = self.__data_generation(indexes)
        return X, Y

    def on_epoch_end(self): 
        if self.train_type == 0:
            c = list(zip(self.dataReferances["Raw"], self.dataReferances["Ann"]))
            shuffle(c)
            self.dataReferances["Raw"], self.dataReferances["Ann"] = zip(*c)


    def __data_generation(self, indexes): 
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        if self.wavelet:
            X = np.empty((self.batch_size, *self.dim, 4))
        else:
            X = np.empty((self.batch_size, *self.dim, 1))
        Y = np.empty((self.batch_size, *self.dim, 1), dtype=int)


        return X,Y


    



# name: LeGall 5-3 GPU 
# author: Declan McIntosh
# contact: contact@declanmcintosh.com
# paper: TBA

import numpy as np 
import csv
import os
import keras


def generateDataReferancesKvasir(dataDir="/Data/Kvasir-SEG/"):
    imgs   = list(os.listdir(dataDir+"images"))
    imgs   = [dataDir+"images/" + s for s in imgs] 
    masks  = list(os.listdir(dataDir+"masks"))
    masks  = [dataDir+"masks/" + s for s in masks] 

    dataReferances = {"Raw":imgs, "Ann":masks}

def getSampleKvasir(dataReferances, index):
    pass


class DataGeneratorSIIM(keras.utils.Sequence):
    'Based loosely off of https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly'
    def __init__(self, data, batch_size, dim=(512,512), wavelet=False, train_type=0): #
        'Initialization'
        self.train_type = train_type # 0 for train, 1 for validate, 2 for test
        self.on_epoch_end()

    def __len__(self): 
        return 0

    def __getitem__(self, index):
        pass


    def on_epoch_end(self): 
        pass


    def __data_generation(self, indexes): 
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        if self.wavelet:
            X = np.empty((self.batch_size, *self.dim, 4))
        else:
            X = np.empty((self.batch_size, *self.dim, 1))
        Y = np.empty((self.batch_size, *self.dim, 1), dtype=int)


        return X,Y


    



# name: LeGall 5-3 Segmenation DataGenerators
# author: Declan McIntosh
# contact: contact@declanmcintosh.com
# paper: TBA

from cv2 import data
from imgaug.augmenters.arithmetic import SaltAndPepper
import numpy as np 
import cv2
import os
import keras
from random import shuffle
from wavelet import *
import skimage.measure

from tensorflow.python.ops.gen_math_ops import segment_max
import imgaug.augmenters as iaa

import matplotlib.pyplot as plt


def generateDataReferancesKvasir(splits,train_type,dataDir="Data/Kvasir-SEG/"):
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
    ann = np.expand_dims(ann[:,:,0],2)
    return img, ann


class DataGeneratorSIIM(keras.utils.Sequence):
    'Based loosely off of https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly'
    def __init__(self, batch_size, dim=(256,256), wavelet=False, train_type=0, splits=[0.8,0.9,1], channels = 3,\
        dataFinder=generateDataReferancesKvasir, dataPuller=getSampleKvasir, deepSupervision=False): #
        'Initialization'
        self.train_type = train_type # 0 for train, 1 for validate
        self.batch_size = batch_size
        self.channels = channels
        self.wavelet = wavelet
        self.deepSupervision = deepSupervision
        if self.wavelet:
            self.dim = (dim[0]*2,dim[1]*2)
        else:
            self.dim = dim


        self.dataPuller = dataPuller
        self.dataReferances = dataFinder(splits,train_type)
        self.dataAugmentater = iaa.Sequential([
            iaa.Fliplr(0.25),
            iaa.Flipud(0.25),
            iaa.Sometimes(0.25, iaa.Affine(translate_px={"x": (-20, 20)})),
            iaa.Sometimes(0.25, iaa.Affine(rotate=(-45, 45))),
            iaa.Sometimes(0.25, iaa.SaltAndPepper()),
            iaa.Sometimes(0.35,iaa.OneOf([
                    iaa.GaussianBlur((0, 0.5)),
                    iaa.AverageBlur(k=(1, 3)),
                    iaa.MedianBlur(k=(1, 3)),])),
            ])
        self.on_epoch_end()


    def __len__(self): 
        
        return int(np.floor(len(self.dataReferances["Raw"]) / self.batch_size))

    def __getitem__(self, index):
        indexes = list(range(index*self.batch_size,(index+1)*self.batch_size))
        X, Y = self.__data_generation(indexes)
        if self.deepSupervision:
            return X, [Y, Y, Y, Y]
        else:
            return X, Y


    def on_epoch_end(self): 
        if self.train_type == 0:
            c = list(zip(self.dataReferances["Raw"], self.dataReferances["Ann"]))
            shuffle(c)
            self.dataReferances["Raw"], self.dataReferances["Ann"] = zip(*c)



    def __data_generation(self, indexes): 
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization

        X = np.empty((self.batch_size, *self.dim, self.channels), dtype=np.uint8)
        Y = np.empty((self.batch_size, *self.dim, 1), dtype=np.int32)

        for local, ind in enumerate(indexes): # output from here fine
            x, y = self.dataPuller(self.dataReferances, ind, self.dim)
            X[local,:,:,:] = x 
            Y[local,:,:,:] = y


        if self.train_type == 0: 
            X, Y = self.dataAugmentater(images=X, segmentation_maps=Y)

        X = X.astype(np.float32)/255.
        Y = Y.astype(np.float32)    

        if self.wavelet:
            X = dwt_5_3_CPU(X)
            Y = skimage.measure.block_reduce(Y, (1,2,2,1), np.max)


        return X,Y


    



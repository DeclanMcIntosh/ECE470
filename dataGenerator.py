# name: LeGall 5-3 Segmenation DataGenerators and ImageNet DataGenerators
# author: Declan McIntosh
# contact: contact@declanmcintosh.com

import numpy as np 
import cv2
import os
import keras
from random import shuffle
from wavelet import *
import skimage.measure

import imgaug.augmenters as iaa


def generateDataReferancesKvasir(splits,train_type,dataDir="Data/Kvasir-SEG/"):
    # find all referances to data in the Kvaiser-SEG folder
    imgs   = list(os.listdir(dataDir+"images"))
    imgs   = [s for s in imgs] 
    masks  = list(os.listdir(dataDir+"masks"))
    masks  = [s for s in masks] 

    start = 0 if train_type==0 else int(len(imgs)*splits[train_type-1])
    end =  int(len(imgs)*splits[train_type])

    return {"Raw":imgs[start:end], "Ann":masks[start:end]}

def getSampleKvasir(dataReferances, index, dim):
    # Take a index to our found data and load that image
    dataDir="Data/Kvasir-SEG/"
    img = cv2.imread(dataDir+"images/" + dataReferances["Raw"][index])
    ann = cv2.imread(dataDir+"masks/"  + dataReferances["Raw"][index])

    # Scale the input image without squishing it, padding with zeros
    scaling = max([img.shape[0] / dim[0], img.shape[1] / dim[1] ])
    new_x = int(img.shape[0] // scaling)
    new_y = int(img.shape[1] // scaling)
    img = cv2.resize(img, (new_y, new_x))
    ann = cv2.resize(ann, (new_y, new_x))

    img_out = np.zeros((dim[0],dim[1],3))
    ann_out = np.zeros((dim[0],dim[1],1))

    img_out[0:new_x,0:new_y] = img.astype(np.float32)
    ann_out[0:new_x,0:new_y] = np.expand_dims((ann.astype(np.float32)/255)[:,:,0],2)
    return img_out, ann_out

class DataGeneratorSIIM(keras.utils.Sequence):
    'Based loosely off of https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly'
    def __init__(self, batch_size, dim=(256,256), wavelet=False, train_type=0, splits=[0.85,0.925,1], channels = 3,\
        dataFinder=generateDataReferancesKvasir, dataPuller=getSampleKvasir, deepSupervision=False, dataAugs=False): #
        'Initialization'
        self.train_type = train_type # 0 for train, 1 for validate, 2 for test
        self.batch_size = batch_size
        self.channels = channels
        self.wavelet = wavelet
        self.deepSupervision = deepSupervision
        self.dataAugs = dataAugs
        if self.wavelet:
            self.dim = (dim[0]*2,dim[1]*2)
        else:
            self.dim = dim

        self.dataPuller = dataPuller
        self.dataReferances = dataFinder(splits,train_type)

        # define data augmenation
        self.dataAugmentater = iaa.Sequential([
            iaa.Fliplr(0.35),
            iaa.Flipud(0.35),
            iaa.Sometimes(0.35, iaa.Affine(rotate=(-45, 45))),
            iaa.Sometimes(0.35, iaa.Dropout(p=[0.05,0.2], per_channel=True)),
            iaa.Sometimes(0.35, iaa.LogContrast(gain=(0.6, 1.4))),
            iaa.Sometimes(0.35, iaa.Crop(percent=(0, 0.1))),
            ])
        self.on_epoch_end()


    def __len__(self): 
        return int(np.floor(len(self.dataReferances["Raw"]) / self.batch_size))

    def __getitem__(self, index):
        # get a batch of input tensors and target ground truths
        indexes = list(range(index*self.batch_size,(index+1)*self.batch_size))
        X, Y = self.__data_generation(indexes)
        if self.deepSupervision:
            return X, [Y, Y, Y, Y]
        else:
            return X, Y

    def on_epoch_end(self): 
        # if training shuffle things
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

        if self.train_type == 0 and self.dataAugs: 
            X, Y = self.dataAugmentater(images=X, segmentation_maps=Y)

        X = X.astype(np.float32)/128. - 1. 
        Y = Y.astype(np.float32)    


        if self.wavelet:
            X = dwt_5_3_CPU(X)
            Y = skimage.measure.block_reduce(Y, (1,2,2,1), np.max)

        return X,Y


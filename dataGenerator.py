# name: LeGall 5-3 Segmenation DataGenerators
# author: Declan McIntosh
# contact: contact@declanmcintosh.com
# paper: TBA

from cv2 import data
import numpy as np 
import cv2
import os
import keras
from random import shuffle

from tensorflow.python.ops.gen_math_ops import segment_max
import imgaug.augmenters as iaa


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
    def __init__(self, batch_size, dim=(256,256), wavelet=False, train_type=0, splits=[0.7,1,1], channels = 3,\
        dataFinder=generateDataReferancesKvasir, dataPuller=getSampleKvasir): #
        'Initialization'
        self.train_type = train_type # 0 for train, 1 for validate
        self.batch_size = batch_size
        self.channels = channels
        self.dim = dim


        self.dataPuller = dataPuller
        self.dataReferances = dataFinder(splits,train_type)
        self.dataAugmentater = iaa.Sequential([
            iaa.Crop(px=(0, 10)),
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.Sometimes(0.5, iaa.Affine(translate_px={"x": (-40, 40)})),
            iaa.Sometimes(0.25,iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)),
                    iaa.AverageBlur(k=(2, 7)),
                    iaa.MedianBlur(k=(3, 11)),])),
            ])
        self.on_epoch_end()


    def __len__(self): \
        
        return int(np.floor(len(self.dataReferances["Raw"]) / self.batch_size))

    def __getitem__(self, index):
        indexes = list(range(index*self.batch_size,(index+1)*self.batch_size))
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

        X = np.empty((self.batch_size, *self.dim, self.channels), dtype=np.uint8)
        Y = np.empty((self.batch_size, *self.dim, 1), dtype=np.int32)

        for local, ind in enumerate(indexes):
            if ind > 1000:
                print("dude why")
            x, y = self.dataPuller(self.dataReferances, ind, self.dim)
            X[local,:,:,:] = x 
            Y[local,:,:,:] = y

            X, Y = self.dataAugmentater(images=X, segmentation_maps=Y)

        return X,Y


    



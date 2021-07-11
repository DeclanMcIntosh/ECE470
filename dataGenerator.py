# name: LeGall 5-3 Segmenation DataGenerators and ImageNet DataGenerators
# author: Declan McIntosh
# contact: contact@declanmcintosh.com

import numpy as np 
import cv2
import random
import os
import keras
from random import shuffle
from wavelet import *
import skimage.measure
import json
from xml.dom import minidom

import imgaug.augmenters as iaa


def generateDataReferancesKvasir(splits,train_type,dataDir="Data/Kvasir-SEG/"):
    imgs   = list(os.listdir(dataDir+"images"))
    imgs   = [s for s in imgs] 
    masks  = list(os.listdir(dataDir+"masks"))
    masks  = [s for s in masks] 

    start = 0 if train_type==0 else int(len(imgs)*splits[train_type-1])
    end =  int(len(imgs)*splits[train_type])

    return {"Raw":imgs[start:end], "Ann":masks[start:end]}

def getSampleKvasir(dataReferances, index, dim):
    dataDir="Data/Kvasir-SEG/"
    img = cv2.imread(dataDir+"images/" + dataReferances["Raw"][index])
    ann = cv2.imread(dataDir+"masks/"  + dataReferances["Raw"][index])

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
        self.train_type = train_type # 0 for train, 1 for validate
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

        if self.train_type == 0 and self.dataAugs: 
            X, Y = self.dataAugmentater(images=X, segmentation_maps=Y)

        X = X.astype(np.float32)/128. - 1. 
        Y = Y.astype(np.float32)    


        if self.wavelet:
            X = dwt_5_3_CPU(X)
            Y = skimage.measure.block_reduce(Y, (1,2,2,1), np.max)

        return X,Y

class imageNetDataGenerator(keras.utils.Sequence):
    'Based loosely off of https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly'
    def __init__(self, imgDir, classesStructure, annotations, batch_size = 64, image_size=(256,256), wavelets=False, data_source= "Val"):

        self.wavelets=wavelets

        with open(classesStructure) as f:
            self.classes = json.load(f)
        self.imgDir = imgDir
        self.batch_size = batch_size
        self.image_size = image_size
        
        self.anns = annotations

        self.inputs = []

        self.data_source = data_source

        self.initalSetup()

    def __len__(self):
        '''Denotes the number of batches per epoch'''
        return int(np.floor(len(self.inputs) / self.batch_size))

    def __getitem__(self, index):
        if not self.wavelets:
            outX  =  np.zeros((self.batch_size, *self.image_size, 3))
        else:
            outX  =  np.zeros((self.batch_size, self.image_size[0]*2, self.image_size[1]*2, 3))
        outY  =  np.zeros((self.batch_size, 1000))
        

        images = self.inputs[index*self.batch_size:(index+1)*self.batch_size]


        batch_internal_index = 0
        for image in images:
            if self.data_source == "Val":
                sourceFile = image[:-5]
                className = minidom.parse("I:/ILSVRC/Annotations/CLS-LOC/val/" + sourceFile + ".xml").getElementsByTagName("object")[0].getElementsByTagName("name")[0].firstChild.nodeValue
                img = cv2.imread(self.imgDir + "/" + image, 0)
                if not self.wavelets:
                    scaling = max([img.shape[0] / self.image_size[0], img.shape[1] / self.image_size[1] ]) 
                else:
                    scaling = max([img.shape[0] / (self.image_size[0]*2), img.shape[1] / (self.image_size[1]*2) ]) 
                new_x = int(img.shape[0] // scaling)
                new_y = int(img.shape[1] // scaling)
                img = cv2.resize(img, (new_y, new_x))
                img = img[..., np.newaxis]

                outY[batch_internal_index, self.classes[className][0]-1] = 1.

                outX[batch_internal_index, 0:new_x, 0:new_y] = img.astype(np.float32)/128 - 1.

            else:
                source = image[0:len("n03255030")]

                img = cv2.imread(self.imgDir + "/" + source + "/" + image, 0)
                if not self.wavelets:
                    scaling = max([img.shape[0] / self.image_size[0], img.shape[1] / self.image_size[1] ]) 
                else:
                    scaling = max([img.shape[0] / (self.image_size[0]*2), img.shape[1] / (self.image_size[1]*2) ]) 
                    new_x = floor(img.shape[0] / scaling)
                    new_y = floor(img.shape[1] / scaling)
                    img = cv2.resize(img, (new_y, new_x))
                    img = img[..., np.newaxis]

                outY[batch_internal_index, self.classes[source][0]-1] = 1.
                
                outX[batch_internal_index, 0:new_x, 0:new_y] = img.astype(np.float32)/128 - 1.

            batch_internal_index += 1

        if self.wavelets:
            outX = dwt_5_3_CPU(outX)

        return outX, outY

    def on_epoch_end(self):
        ''' Shuffle the data if that is required'''
        if not self.data_source == "Val":
            random.shuffle(self.inputs)

    def initalSetup(self):
        '''We want only images that have corresponding right iamges and nearby right images'''
        if self.data_source == "Val":
            self.inputs.extend(os.listdir(self.imgDir))
        else:
            for key in self.classes.keys():
                self.inputs.extend(os.listdir(self.imgDir + "/" + key))

            self.on_epoch_end()

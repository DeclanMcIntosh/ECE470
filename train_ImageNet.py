# author: Declan McIntosh
# contact: contact@declanmcintosh.com

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.utils import multi_gpu_model

from model import * 
from dataGenerator import * 

imageSize = (256,256)
batchSize = 48

classesFile   = "I:/ILSVRC/Classes.json"
dataFileTrain = "I:/ILSVRC/Data/CLS-LOC/train"
dataFileVal   = "I:/ILSVRC/Data/CLS-LOC/val"
annDir        = "I:/ILSVRC/Annotations/CLS-LOC/val"
waveletsFlag = True

if waveletsFlag:
    channels = 12
    saveModMethod = "Wavelet"
else:
    channels = 3
    saveModMethod = "GreyScale"


DataGeneratorTrain = imageNetDataGenerator(imgDir=dataFileTrain,annotations=annDir, classesStructure=classesFile, batch_size=batchSize, image_size=imageSize, wavelets=waveletsFlag, data_source= "Train")
DataGeneratorVal   = imageNetDataGenerator(imgDir=dataFileVal,  annotations=annDir, classesStructure=classesFile, batch_size=batchSize, image_size=imageSize, wavelets=waveletsFlag)        


model = Unet_plus_plus_backbone(img_rows=imageSize[0], img_cols=imageSize[1], color_type=channels)

multi_model = multi_gpu_model(model, gpus=2)

multi_model.compile(optimizer=Adam(lr=1e-5), loss='categorical_crossentropy', metrics=["acc"])
mc = ModelCheckpoint("./Logs/ImageNetBackBoneWavelet_epoch-{epoch:03d}-{val_loss:03f}-{val_acc:03f}.h5", save_best_only=True)
rl = ReduceLROnPlateau(monitor='loss', patience=4)

multi_model.fit_generator(DataGeneratorTrain, validation_data=DataGeneratorVal, epochs=100, callbacks=[mc,rl], workers=8)

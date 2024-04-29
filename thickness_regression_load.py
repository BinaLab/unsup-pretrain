import keras,os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
import numpy as np

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.nasnet import NASNetLarge
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint

import numpy as np
from os import *
from os.path import *
import cv2
import tensorflow as tf
import datetime
import data_loader

from keras import backend as K

epochs = 100
max_labels = 34 ### number of classes = number of layers
load_timestamp = '10-28-21_23-23' ### load pretrained model
img_ext = '.png'
txt_ext = '.txt'
np_ext = '.npy'
h5_ext = '.h5'
models_str = 'models'
plots_str = 'plots'
evaluation_str = 'evaluation'
history_str = 'history'
out = 'output'
timestamp = datetime.datetime.now()
dt_format = "%m-%d-%y_%H-%M"
models = join(out, timestamp.strftime(dt_format),models_str)
plots = join(out, timestamp.strftime(dt_format),plots_str)
evaluation = join(out, timestamp.strftime(dt_format),evaluation_str)
history = join(out, timestamp.strftime(dt_format),history_str)

if not isdir(models):
    makedirs(models)
if not isdir(plots):
    makedirs(plots)
if not isdir(evaluation):
    makedirs(evaluation)
if not isdir(history):
    makedirs(history)



def train_model(model, model_name, traindata, thickness_estims):
    
    model_chkpnt = ModelCheckpoint(
    filepath=join(models,model_name+'.h5'),
    save_weights_only=False,
    monitor='loss',
    mode='min',
    save_best_only=True)
    
    # model.summary()
    hist = model.fit(traindata, thick_estims, verbose=1, epochs=epochs, callbacks=[model_chkpnt])

    print("-- " + model_name + ' trained -- ')
    np.save(join(history,model_name+np_ext),hist.history)
    np.savetxt(join(evaluation,model_name+txt_ext), model.evaluate(traindata, thick_estims, verbose=1))
    return hist

def plot_model(hist, model_name, folder_out=plots):
    print(hist.history.keys())
    plt.plot(hist.history["loss"])
    plt.title(model_name)
    plt.ylabel("MAE Loss")
    plt.xlabel("Epoch")
    plt.yscale("linear")
    plt.savefig(join(folder_out,model_name+img_ext))
    plt.close()


traindata = data_loader.traindata


thick_estims = data_loader.train_thick

model_in = join(out, load_timestamp,models_str)
### model definition ###

model_name1 = 'InceptionV3'
pretr_model = load_model(join(model_in, model_name1+h5_ext))
print('Pretrained' + model_name1+' loaded. Now training on new set...')
hist1 = train_model(pretr_model, model_name1, traindata, thick_estims)
plot_model(hist1, model_name1)


model_name2 = 'DenseNet121'
pretr_model = load_model(join(model_in, model_name2+h5_ext))
print('Pretrained' + model_name2+' loaded. Now training on new set...')
hist2 = train_model(pretr_model, model_name2, traindata, thick_estims)
plot_model(hist2, model_name2)


model_name3 = 'ResNet50'
pretr_model = load_model(join(model_in, model_name3+h5_ext))
print('Pretrained' + model_name3+' loaded. Now training on new set...')
hist3 = train_model(pretr_model, model_name3, traindata, thick_estims)
plot_model(hist3, model_name3)


model_name4 = 'Xception'
pretr_model = load_model(join(model_in, model_name4+h5_ext))
print('Pretrained' + model_name4+' loaded. Now training on new set...')
hist4 = train_model(pretr_model, model_name4, traindata, thick_estims)
plot_model(hist4, model_name4)


model_name5 = 'MobileNetV2'
pretr_model = load_model(join(model_in, model_name5+h5_ext))
print('Pretrained' + model_name5+' loaded. Now training on new set...')
hist5 = train_model(pretr_model, model_name5, traindata, thick_estims)
plot_model(hist5, model_name5)


## plot all
plt.plot(hist1.history["loss"])
plt.plot(hist2.history["loss"])
plt.plot(hist3.history["loss"])
plt.plot(hist4.history["loss"])
plt.plot(hist5.history["loss"])
plt.title("Model Loss Curves")
plt.ylabel("MAE Loss")
plt.xlabel("Epoch")
plt.yscale("linear")
plt.legend([model_name1,model_name2,model_name3,model_name4,model_name5])
plt.savefig(join(plots,'all_loss'+img_ext))
plt.close()

print('done')
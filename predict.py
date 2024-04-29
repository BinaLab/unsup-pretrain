import keras,os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.densenet import DenseNet121
# from tensorflow.keras.applications.resnet import ResNet50
# from tensorflow.keras.applications.xception import Xception
# from tensorflow.keras.applications.nasnet import NASNetLarge
# from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint

# from tensorflow.keras.preprocessing import image_dataset_from_directory
import numpy as np
from PIL import Image
import os
import cv2
import tensorflow as tf


img_ext = '.png'
txt_ext = '.txt'
max_labels = 27
inception_chkpnt = 'inception.h5'
epochs = 10
img_root = '../../Dataset/Snow Radar/2012_cropped/train/image/'
thick_root = '../../Dataset/Snow Radar/2012_cropped/train/thickness_estimates2/'

### load training images ###
traindata = []
img_files = [os.path.join(img_root,file) for file in os.listdir(img_root) if img_ext in file]
for file in img_files:
  img = cv2.imread(file)
  try:
      img_224 = cv2.resize(img, (224,224))
  except:
      print(file + ' error')
      continue
  traindata.append(img_224)

traindata = np.asarray(traindata)

### load thickness estimates ###
thick_estims = []
thick_files = [os.path.join(thick_root,file) for file in os.listdir(thick_root) if txt_ext in file]
for file in thick_files:
    thicks = np.loadtxt(file)
    thick_estims.append(thicks)
thick_estims = np.asarray(thick_estims)

print('data loaded')

model_mae = load_model('inception_mse.h5')
model_out = model_mae.evaluate(traindata, thick_estims, verbose=1, return_dict=True)
print(model_out)
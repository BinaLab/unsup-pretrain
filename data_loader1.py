from os.path import *
from os import *
import cv2
import numpy as np

img_ext = '.png'
txt_ext = '.txt'
layer_prfx = 'data' ## 'layer' or 'data'
img_prfx = 'image' ## 'data' or 'image'

train_img_root = '../../Dataset/Snow Radar/2012_main_dv/train/image/'
train_thick_root = '../../Dataset/Snow Radar/weatherModel_IBK/greenland_lora/2012/train/thick_ext'
test_img_root = '../../Dataset/Snow Radar/2012_main_dv/test/image/'
test_thick_root = '../../Dataset/Snow Radar/weatherModel_IBK/greenland_lora/2012/test/thick_ext'

# train_img_root = '../../Dataset/Snow Radar/2012_cropped/train/image/'
# train_thick_root = '../../Dataset/Snow Radar/2012_cropped/train/thickness_estimates3/'
# test_img_root = '../../Dataset/Snow Radar/2012_cropped/test/image/'
# test_thick_root = '../../Dataset/Snow Radar/2012_cropped/test/thickness_estimates3/'

### load training images ###
print('loading training images')
traindata = []
img_files = [join(train_img_root,file.replace(layer_prfx,img_prfx).replace(txt_ext,img_ext)) for file in listdir(train_thick_root) if txt_ext in file]
for file in sorted(img_files):
    img = cv2.imread(file)
    try:
        img_224 = cv2.resize(img, (224,224))
    except:
        print(file + ' error')
        continue
    traindata.append(img_224)

traindata = np.asarray(traindata)
train_mean = np.mean(traindata)
train_std = np.std(traindata)
traindata = (traindata - train_mean) / train_std

### load training thickness ###
print('loading training thickness')
train_thick = []
thick_files = [join(train_thick_root,file) for file in listdir(train_thick_root) if txt_ext in file]
for file in sorted(thick_files):
    thicks = np.loadtxt(file)
    train_thick.append(thicks)
train_thick = np.asarray(train_thick)

### --- test --- ###
### load test images ###
print('loading test images')
testdata = []
img_files = [join(test_img_root,file.replace(layer_prfx,img_prfx).replace(txt_ext,img_ext)) for file in listdir(test_thick_root) if txt_ext in file]
for file in sorted(img_files):
    img = cv2.imread(file)
    try:
        img_224 = cv2.resize(img, (224,224))
    except:
        print(file + ' error')
        continue
    testdata.append(img_224)

testdata = np.asarray(testdata)
test_mean = np.mean(testdata)
test_std = np.std(testdata)
testdata = (testdata - test_mean) / test_std
print(testdata.shape)
### load test thickness ###
print('loading test thickness')
test_thick = []
thick_files = [join(test_thick_root,file) for file in listdir(test_thick_root) if txt_ext in file]
for file in sorted(thick_files):
    thicks = np.loadtxt(file)
    test_thick.append(thicks)
test_thick = np.asarray(test_thick)
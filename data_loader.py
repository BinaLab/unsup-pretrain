from os.path import *
from os import *
import cv2
import numpy as np

img_ext = '.png'
txt_ext = '.txt'
layer_prfx = 'data' ## 'layer' or 'data'
img_prfx = 'image' ## 'data' or 'image'
# max_labels = 34


train_img_root = '../../Dataset/Snow Radar/2012_main_dv_dry2/train/image'
train_thick_man_root = '../../Dataset/Snow Radar/2012_main_dv_dry2/train/manual_thick_ext'
train_thick_phy_root = '../../Dataset/Snow Radar/2012_main_dv_dry2/train/mar_thick_ext'

test_img_root = '../../Dataset/Snow Radar/2012_main_dv_dry2/test/image'
test_thick_root = '../../Dataset/Snow Radar/2012_main_dv_dry2/test/manual_thick_ext'
# test_thick_root = '../../Dataset/Snow Radar/2012_main_dv_dry2/test/mar_thick_ext'

### load training images ###
print('loading training images')
traindata = []
img_files = [join(train_img_root,file) for file in listdir(train_img_root) if img_ext in file]
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
train_thick_man = []
thick_files = [join(train_thick_man_root,file) for file in listdir(train_thick_man_root) if txt_ext in file]
for file in sorted(thick_files):
    thicks = np.loadtxt(file)
    train_thick_man.append(thicks)
train_thick_man = np.asarray(train_thick_man)

train_thick_phy = []
thick_files = [join(train_thick_phy_root,file) for file in listdir(train_thick_phy_root) if txt_ext in file]
for file in sorted(thick_files):
    thicks = np.loadtxt(file)
    train_thick_phy.append(thicks)
train_thick_phy = np.asarray(train_thick_phy)

train_thick = train_thick_man
# train_thick = train_thick_phy
# train_thick = []

# for idx in range(len(train_thick_man)):
#     train_thick.append([train_thick_man[idx],train_thick_phy[idx]])

# train_thick = [train_thick_man, train_thick_phy]
# train_thick = train_thick_man


### shape = total files x 2 x 34
# for idx in range(len(train_thick_man)):
#     train_thick.append(np.stack((train_thick_man[idx],train_thick_phy[idx]), axis=0))
# train_thick = np.array(train_thick)


### shape = total files x 68 ( = both GTs one after the other)
# for idx in range(len(train_thick_man)):
#     train_thick.append(train_thick_man[idx].tolist() + train_thick_phy[idx].tolist())
# train_thick=np.asarray(train_thick)

### shape = total files x 34 x 2
# for idx in range(len(train_thick_man)):
#     file_thick=[]
#     for j in range(max_labels):
#         file_thick.append([train_thick_man[idx][j], train_thick_phy[idx][j]])
#     train_thick.append(file_thick)
# train_thick=np.asarray(train_thick)

### --- test --- ###
### load test images ###
print('loading test images')
testdata = []
img_files = [join(test_img_root,file) for file in listdir(test_img_root) if img_ext in file]
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
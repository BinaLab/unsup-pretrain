import cv2
from os import listdir, makedirs
import numpy as np
import data_loader
from os.path import *
from tensorflow.keras.models import load_model

timestamp = '10-31-21_00-16' ### insert timestamp here for which you want to predict output
out_root = '../../Outputs/Snow Radar/Thickness Regression'  ### output folder where you want to generate model predictions
model_ext = '.h5'
img_ext = '.png'
txt_ext = '.txt'
train = 'train'
test = 'test'
img_prfx = 'image'
layer_prfx = 'layer'
models_in = join('output',timestamp,'models')
models = [model for model in listdir(models_in) if model_ext in model]

    
# train_folder = '../../Dataset/Snow Radar/2012_main_dv/train/image/'
# train_thick_root = '../../Dataset/Snow Radar/weatherModel_IBK/greenland_lora/2012/train/thick_ext'
# test_folder = '../../Dataset/Snow Radar/2012_main_dv/test/image/'
# test_thick_root = '../../Dataset/Snow Radar/weatherModel_IBK/greenland_lora/2012/test/thick_ext'

# train_folder = '../../Dataset/Snow Radar/2012_cropped/train/image/'
# train_thick_root = '../../Dataset/Snow Radar/2012_cropped/train/thickness_estimates3/'
# test_folder = '../../Dataset/Snow Radar/2012_cropped/test/image/'
# test_thick_root = '../../Dataset/Snow Radar/2012_cropped/test/thickness_estimates3/'

train_folder = '../../Dataset/Snow Radar/2012_main_dv_dry2/train/image/'
train_thick_root = '../../Dataset/Snow Radar/2012_main_dv_dry2/train/manual_thick_ext/'
# train_thick_root = '../../Dataset/Snow Radar/2012_main_dv_dry2/train/mar_thick_ext/'
test_folder = '../../Dataset/Snow Radar/2012_main_dv_dry2/test/image/'
test_thick_root = '../../Dataset/Snow Radar/2012_main_dv_dry2/test/manual_thick_ext'
# test_thick_root = '../../Dataset/Snow Radar/2012_main_dv_dry2/test/mar_thick_ext'

out_train = join(out_root, timestamp, train)
out_test = join(out_root, timestamp, test)

if not isdir(out_train):
    makedirs(out_train)
if not isdir(out_test):
    makedirs(out_test)

# f_tr = open(join(out_train,'model_eval.txt'),'w+')
# f_te = open(join(out_test,'model_eval.txt'),'w+')  

def predict_train(model, model_name):
    
    out_folder = join(out_train, model_name[:model_name.index(model_ext)])
    if not isdir(out_folder):
        makedirs(out_folder)
        
    # f_tr.write(model_name[:model_name.index(model_ext)] + ": " 
    #            + str("{:.3f}".format(model.evaluate(data_loader.traindata, 
    #                                         data_loader.train_thick, verbose=1))) + "\n")
    np.savetxt(join(out_train,model_name.replace(model_ext,txt_ext)), 
                model.evaluate(data_loader.traindata, data_loader.train_thick, verbose=1), fmt="%.3f")
    print(model_name + ' training evaluated')
    mean = data_loader.train_mean
    std = data_loader.train_std
    
    files = [file.replace(layer_prfx,img_prfx).replace(txt_ext,img_ext) for file in listdir(train_thick_root) if txt_ext in file]
    
    print(model_name + " predicting ....")
    for file in files:
        img = cv2.imread(join(train_folder,file))
        try:
            img = cv2.resize(img, (224,224))
        except:
            print(file + ' error')
            continue
        
        img = (img - mean) / std
        
        img = np.expand_dims(img,axis=0)
        pred = model.predict(img)
        np.savetxt(join(out_folder,file.replace(img_ext,txt_ext)),pred,fmt="%.3f",delimiter="\n")
        
    print(model_name + ' train prediction complete')


def predict_test(model, model_name):
    
    out_folder = join(out_test, model_name[:model_name.index(model_ext)])
    if not isdir(out_folder):
        makedirs(out_folder)
    
    # f_te.write(model_name[:model_name.index(model_ext)] + ": " 
    #            + str("{:.3f}".format(model.evaluate(data_loader.testdata, 
    #                                         data_loader.test_thick, verbose=1))) + "\n")
    np.savetxt(join(out_test,model_name.replace(model_ext,txt_ext)), 
                model.evaluate(data_loader.testdata, data_loader.test_thick, verbose=1), fmt="%.3f")
    print(model_name + ' test evaluated')
    mean = data_loader.test_mean
    std = data_loader.test_std
    
    files = [file.replace(layer_prfx,img_prfx).replace(txt_ext,img_ext) for file in listdir(test_thick_root) if txt_ext in file]
    
    print(model_name + " predicting test set ....")
    for file in files:
        img = cv2.imread(join(test_folder,file))
        try:
            img = cv2.resize(img, (224,224))
        except:
            print(file + ' error')
            continue
        img = (img - mean) / std
        img = np.expand_dims(img,axis=0)
        pred = model.predict(img)
        np.savetxt(join(out_folder,file.replace(img_ext,txt_ext)),pred, fmt="%.3f",delimiter="\n") 
        
    print(model_name + ' test prediction complete')
        
    

    
for model_name in models:
    model = load_model(join(models_in,model_name))
    predict_train(model, model_name)
    predict_test(model,model_name)
    
# f_tr.close()
# f_te.close()   
print('done')

# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 15:48:22 2022

@author: User
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from models import DnCNN
from dataset import  Dataset_train, Dataset_val
from utils import *
from datetime import datetime

import os
import os.path
import numpy as np
import random
import h5py
import torch
import cv2
import glob
import torch.utils.data as udata


#%%
# parser = argparse.ArgumentParser(description="DnCNN")
# parser.add_argument("--prepare_data", action='store_true',  help='run prepare_data or not')
# parser.add_argument("--batchSize", type=int, default=128, help="Training batch size")
# parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
# parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
# parser.add_argument("--milestone", type=int, default=30, help="When to decay learning rate; should be less than epochs")
# parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
# parser.add_argument("--outf", type=str, default="logs", help='path of log files')
# parser.add_argument("--noiseL", type=float, default=25, help='noise level; ignored when mode=B')
# parser.add_argument("--alpha", type=float, default=0.5, help='alpha')

# parser.add_argument("--val_noiseL", type=float, default=25, help='noise level used on validation set')
# parser.add_argument("--gpu", type=int, default=0, help="gpu number")
# parser.add_argument("--training", type=str, default="R2R", help='trainnig type')



#%%
def normalize(data):
    return data/255.
def Im2Patch(img, win, stride=1):
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw-win+0+1:stride, 0:endh-win+0+1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win*win,TotalPatNum], np.float32)
    for i in range(win):
        for j in range(win):
            patch = img[:,i:endw-win+i+1:stride,j:endh-win+j+1:stride]
            Y[:,k,:] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])
def Patch2Im(Y, win, img_size,stride=1):
    
    endc = img_size[0]
    endw = img_size[1]
    endh = img_size[2]
    Y = Y.reshape([endc,win*win,-1])
    img = np.zeros([endc,endw,endh],np.float32)
    weight = np.zeros([endc,endw,endh],np.float32)
    tempw = (endw-win) //stride +1
    temph = (endh-win) // stride +1
    k = 0
    for i in range(win):
        for j in range(win):
            img[:,i:endw-win+i+1:stride,j:endh-win+j+1:stride] = img[:,i:endw-win+i+1:stride,j:endh-win+j+1:stride] + Y[:,k,:].reshape(endc,tempw,temph)
            weight[:,i:endw-win+i+1:stride,j:endh-win+j+1:stride] = weight[:,i:endw-win+i+1:stride,j:endh-win+j+1:stride] + 1.
            k = k+1
    img = img/(weight+1e-6)
    return img

def prepare_data(data_path,sigma, patch_size, stride, aug_times=1):
    # train
    print('process training data')
    scales = [1, 0.9, 0.8, 0.7]
    files = glob.glob(os.path.join(data_path, 'train', '*.png'))
    files.sort()
    h5f = h5py.File('train_sigma_%s.h5'%(sigma), 'w')

    train_num = 0
    for i in range(len(files)):
        img = cv2.imread(files[i])
        img = np.float32(normalize(img))
        
        h, w, c = img.shape
        for k in range(len(scales)):
            Img = cv2.resize(img, (int(h*scales[k]), int(w*scales[k])), interpolation=cv2.INTER_CUBIC)
            Img = np.expand_dims(Img[:,:,0].copy(), 0)
            Img_noisy = Img + np.random.normal(0,sigma/255.,Img.shape)

            patches = Im2Patch(Img, win=patch_size, stride=stride)
            patches_noisy = Im2Patch(Img_noisy, win=patch_size, stride=stride)
            # patches_noisy2 = Im2Patch(Img_noisy2, win=patch_size, stride=stride)
            print("file: %s scale %.1f # samples: %d" % (files[i], scales[k], patches.shape[3]*aug_times))
            for n in range(patches.shape[3]):
                data = patches[:,:,:,n].copy()
                data_noisy = patches_noisy[:,:,:,n].copy()
 
                h5f.create_dataset(str(train_num), data=np.stack((data,data_noisy),axis=0))

                train_num += 1

                for m in range(aug_times-1):
                    data_aug = data_augmentation(data, np.random.randint(1,8))

                    data_noisy_aug = data_augmentation(data_noisy, np.random.randint(1,8))
                    
                    h5f.create_dataset(str(train_num)+"_aug_%d" % (m+1), data=np.stack((data_aug,data_noisy_aug),axis=0))

                    train_num += 1

    h5f.close()
    print('training set, # samples %d\n' % train_num)
#    # val
    print('\n process validation data')
    files.clear()
    files = glob.glob(os.path.join(data_path,  'bsd68/test*.png'))
    files.sort()
    h5f = h5py.File('val_%s_Set68.h5'%(sigma), 'w')
    val_num = 0
    for i in range(len(files)):
        print("file: %s" % files[i])
        img = cv2.imread(files[i])
        img = np.expand_dims(img[:,:,0], 0)
        img = np.float32(normalize(img))
        img_noisy = img + np.random.normal(0,sigma/255.,img.shape)
        h5f.create_dataset(str(val_num), data=np.stack((img,img_noisy),axis=0))
        val_num += 1
    h5f.close()
    
    print('val set, # samples %d\n' % val_num)
    



#%%
# aug_choice = augmentations_new.augmentations_choice
prepare_data(data_path='./data/',sigma=25, patch_size=40, stride=10, aug_times=2)





    
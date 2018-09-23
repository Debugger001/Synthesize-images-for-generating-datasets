import os
from os.path import join as pjoin
import collections
import csv
import json
import torch
import numpy as np
import scipy.misc as m
import scipy.io as io
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
import signal
import argparse
from defaults import *
import skimage
import math
import random
import glob

def pure_black_Gaussian_noise(files):
    img_file = files[0]
    mask_files = files[1]
    print("Woring on %s"%img_file)
    already_syn_list = glob.glob(os.path.join(BLACK_DIR, '*.jpg'))
    print("Already synthesized %d images."%len(already_syn_list))
    img_name = img_file.split('.')[0]
    img = Image.open(TRAIN_IMG_DIR + img_file)
    img_np = np.array(img.getdata(), dtype=np.uint8).reshape(img.size[1], img.size[0], 3)
    img_fl = cv2.imread(TRAIN_IMG_DIR + img_file)
    for i in range(img_fl.shape[1]):
        for j in range(img_fl.shape[0]):
            r = random.randint(0, 255)
            b = random.randint(0, 255)
            g = random.randint(0, 255)
            img_fl[i][j] = [r, b, g]
    img_black_np = np.array(img.getdata(), dtype=np.uint8).reshape(img.size[1], img.size[0], 3)
    masks = []
    for mask_file in mask_files:
        mask = Image.open(TRAIN_MASK_DIR + mask_file)
        mask_np = np.array(mask.getdata(), dtype=np.uint8).reshape(mask.size[1], mask.size[0])
        masks.append(mask_np)
    for i in range(img_black_np.shape[1]):
        for j in range(img_black_np.shape[0]):
            isbg = True
            for mask in masks:
                if mask[i][j] > 0:
                    isbg = False
                    break
            if isbg:
                img_black_np[i][j] = [0,0,0]
            else:
                img_fl[i][j] = img_np[i][j]
    img_pure_blk = Image.fromarray(img_black_np)
    img_noise = Image.fromarray(img_fl)
    img_pure_blk.save(BLACK_DIR + img_name + 'black.jpg')
    img_noise.save(NOISE_DIR + img_name + 'noise.jpg')

def init_worker():
    '''
    Catch Ctrl+C signal to termiante workers
    '''
    signal.signal(signal.SIGINT, signal.SIG_IGN)

if not os.path.exists(BLACK_DIR):
    os.makedirs(BLACK_DIR)
if not os.path.exists(NOISE_DIR):
    os.makedirs(NOISE_DIR)

img_list = glob.glob(os.path.join(TRAIN_IMG_DIR, '*none.jpg'))
mask_list = glob.glob(os.path.join(TRAIN_MASK_DIR, '*none*.png'))

img_list = [i.split('/')[-1] for i in img_list]
mask_list = [i.split('/')[-1] for i in mask_list]

# print(img_list[0])
# print(mask_list[0])

files = []

for img in tqdm(img_list):
    file_pair = []
    mask_files = []
    img_num = img.split('none')[0]
    for mask in mask_list:
        mask_num = mask.split('none')[0]
        if mask_num == img_num:
            mask_files.append(mask)
    file_pair.append(img)
    file_pair.append(mask_files)
    files.append(file_pair)



partial_func = partial(pure_black_Gaussian_noise)
p = Pool(NUMBER_OF_WORKERS, init_worker)
try:
    p.map(partial_func, files)
except KeyboardInterrupt:
    print("....\nCaught KeyboardInterrupt, terminating workers")
    p.terminate()
else:
    print("Wrong...")
    p.close()
p.join()

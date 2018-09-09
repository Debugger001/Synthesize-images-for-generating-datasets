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
import argparse
import math

def check_pixel_value(args):
    img = cv2.imread(args.img_path, cv2.IMREAD_UNCHANGED)
    pixel_value = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            counter = 0
            isin = False
            for k in range(len(pixel_value)):
                if img[i][j] != pixel_value[k]:
                    counter += 1
                else:
                    break
            if counter == len(pixel_value):
                pixel_value.append(img[i][j])
    print(pixel_value)








if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create pdm from images.")
    parser.add_argument("--img_path", type = str, default = "/home/lc/syndata-generation/demo_data_dir/objects_dir/",
        help="The directory which contains the images.")
    args = parser.parse_args()
    check_pixel_value(args)

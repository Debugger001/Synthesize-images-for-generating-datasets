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

def create_white_image():
    img = np.zeros((512, 512), dtype=np.uint8)
    # for i in range(0, img.shape[0]):
    #     for j in range(0, img.shape[1]):
    #         for k in range(0, img.shape[2]):
    #             img[i][j][k] = 255
    cv2.imwrite('black_img_1C.jpg', img)

create_white_image()

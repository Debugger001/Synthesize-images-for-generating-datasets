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

def Read_pbm():
    data_path = "/Users/pro/Desktop/syndata-generation/data_dir/distractor_objects_dir/honey_bunches/NP1_123.pbm"
    img = cv2.imread(data_path, -1)
    print(img)
    print(img.shape)

Read_pbm()

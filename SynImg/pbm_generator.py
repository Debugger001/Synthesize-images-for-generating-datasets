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



# read image with alpha channel

def Read_image(args):
    dir_list = tuple(open(args.img_path + 'dir_list.txt', 'r'))
    dir_list = [id_.rstrip() for id_ in dir_list]
    print(dir_list)
    print("Reading diretories in dir lists...")
    label = 0
    for i in tqdm(dir_list):

        img_path = pjoin(args.img_path, i + '/')
        img_list = os.listdir(img_path)
        # img_list = tuple(open(args.img_path + i + '/img_list.txt', 'r'))
        img_list = [id_.rstrip() for id_ in img_list if id_.split('.')[-1] == 'png']
        # print("Reading images in images lists...")
        for ii in img_list:
            img_name = pjoin(img_path, ii)
            img = cv2.imread(img_name, cv2.IMREAD_UNCHANGED)
            image = cv2.imread(img_name, cv2.IMREAD_UNCHANGED)
            print(img.shape)

            # print(img.shape[0])



# alpha channel 0-1 quantization
    # alpha_channel = [...,3]
    # alpha_channel = alpha_channel.astype(np.bool)
            # alpha_channel = img[:, :, 3]
            # alpha_channel = alpha_channel.astype(np.bool)
            for row in range(0, img.shape[0]):
                for col in range(0, img.shape[1]):
                    if img[row, col, 3] == 0:
                        img[row][col][0] = 255
                        img[row][col][1] = 255
                        img[row][col][2] = 255
                    else:
                        img[row][col][0] = 0
                        img[row][col][1] = 0
                        img[row][col][2] = 0


            # img[:, :, 0] = math.floor(img[:, :, 0] / 255.0)
            # img[:, :, 1] = math.floor(img[:, :, 1] / 255.0)
            # img[:, :, 2] = math.floor(img[:, :, 2] / 255.0)


# save to pbm file
            # img[:, :, 3] = alpha_channel
            output_path = pjoin(args.export_path)
            if not os.path.exists(output_path): os.makedirs(output_path)
            # print(img.shape)
            # print(args.export_path + '/' + str(label) + ".png")
            cv2.imwrite(args.export_path + i + '/' + str(label) + ".png", img, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
            # image.save(output_path + str(label) + '.pbm')
            label += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create pdm from images.")
    parser.add_argument("--img_path", type = str, default = "/Users/pro/Desktop/Lab/syndata/syndata-generation/data_dir/objects_dir/",
      help="The directory which contains the images.")
    parser.add_argument("--export_path", type = str, default = "/Users/pro/Desktop/Lab/syndata/syndata-generation/data_dir/objects_mask_dir/",
      help="The directory where pbm lists will be created.")
    args = parser.parse_args()
    Read_image(args)

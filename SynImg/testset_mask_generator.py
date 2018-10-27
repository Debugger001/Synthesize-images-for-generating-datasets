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
import glob



# read image with alpha channel

def Read_image(args):
    dir_list = tuple(open(args.img_path + 'dir_list.txt', 'r'))
    dir_list = [id_.rstrip() for id_ in dir_list]
    print(dir_list)
    print("Reading diretories in dir lists...")
    label = 0
    for i in tqdm(dir_list):

        img_path = pjoin(args.img_path, i + '/')
        # img_list = tuple(open(args.img_path + i + '/img_list.txt', 'r'))
        img_list = glob.glob(os.path.join(img_path, '*.png'))
        # print(img_list)
        img_list = [id_.split('/')[-1] for id_ in img_list]
        # img_list = [id_.rstrip() for id_ in img_list]
        # print("Reading images in images lists...")
        for ii in img_list:
            img_name = pjoin(img_path, ii)
            img = cv2.imread(img_name, cv2.IMREAD_UNCHANGED)
            # image = cv2.imread(img_name, cv2.IMREAD_UNCHANGED)
            # print(img.shape)

            # print(img.shape[0])



# alpha channel 0-1 quantization
    # alpha_channel = [...,3]
    # alpha_channel = alpha_channel.astype(np.bool)
            # alpha_channel = img[:, :, 3]
            # alpha_channel = alpha_channel.astype(np.bool)
            # for row in range(0, img.shape[0]):
            #     for col in range(0, img.shape[1]):
            #         if img[row, col, 3] == 0:
            #             img[row][col][0] = 1
            #             img[row][col][1] = 1
            #             img[row][col][2] = 1
            #         else:
            #             img[row][col][0] = 0
            #             img[row][col][1] = 0
            #             img[row][col][2] = 0


            # img[:, :, 0] = math.floor(img[:, :, 0] / 255.0)
            # img[:, :, 1] = math.floor(img[:, :, 1] / 255.0)
            # img[:, :, 2] = math.floor(img[:, :, 2] / 255.0)


# save to pbm file
            # img[:, :, 3] = alpha_channel
            output_path = pjoin(args.export_path)
            if not os.path.exists(output_path): os.makedirs(output_path)
            # print(img.shape)
            # print(args.export_path + '/' + str(label) + ".png")
            img_label = ii.split('.')[-2]
            img_output_path = pjoin(args.export_path, i + '/')
            if not os.path.exists(img_output_path): os.makedirs(img_output_path)
            rows = img.shape[0]
            cols = img.shape[1]
            image = np.zeros((rows, cols), dtype = np.uint8)
            # print(image.shape)
            # print(img.shape)
            for iii in range(0, img.shape[0]):
                for jjj in range(0, img.shape[1]):
                    if img[iii, jjj, 3] == 0:
                        image[iii][jjj] = 1
                    else:
                        image[iii][jjj] = 0
            cv2.imwrite(img_output_path + img_label + '.pbm', image, [int(cv2.IMWRITE_PXM_BINARY), 1])
            # image.save(output_path + str(label) + '.pbm')
            label += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create pdm from images.")
    parser.add_argument("--img_path", type = str, default = "/home/lc/fyj/obj/",
      help="The directory which contains the images.")
    parser.add_argument("--export_path", type = str, default = "/home/lc/fyj/obj/",
      help="The directory where pbm lists will be created.")
    args = parser.parse_args()
    Read_image(args)

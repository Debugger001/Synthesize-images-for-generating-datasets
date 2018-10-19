import cv2
import numpy as np
import math
import datetime
import json
import os
import re
import fnmatch
import glob
from tqdm import tqdm
from PIL import Image

root_dir = "/home/lc/syndata-generation/output_dir/"
img_dir = "/home/lc/syndata-generation/output_dir/train/"
ins_mask_dir = "/home/lc/syndata-generation/output_dir/instance_segmentation_class/"
sem_mask_dir = "/home/lc/syndata-generation/output_dir/semantic_segmentation_class/"

INFO = {
    "description": "black_bg_generated_Dataset",
    "version": "0.1.0",
    "year": 2018,
    "contributor": "LC",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

CATEGORIES = [
    {
        'id': 1,
        'name': 'bottle_+c',
        'supercategory': 'bottle',
    },
    {
        'id': 2,
        'name': 'bottle_amecoffee',
        'supercategory': 'bottle',
    },
    {
        'id': 3,
        'name': 'bottle_blacktea',
        'supercategory': 'bottle',
    },
    {
        'id': 4,
        'name': 'bottle_coldtea',
        'supercategory': 'bottle',
    },
    {
        'id': 5,
        'name': 'bottle_graintea',
        'supercategory': 'bottle',
    },
    {
        'id': 6,
        'name': 'bottle_greentea',
        'supercategory': 'bottle',
    },
    {
        'id': 7,
        'name': 'bottle_jianlibao',
        'supercategory': 'bottle',
    },
    {
        'id': 8,
        'name': 'bottle_milktea',
        'supercategory': 'bottle',
    },
    {
        'id': 9,
        'name': 'bottle_mingtea',
        'supercategory': 'bottle',
    },

    {
        'id': 10,
        'name': 'bottle_newcoffee',
        'supercategory': 'bottle',
    },
    {
        'id': 11,
        'name': 'bottle_pepsi2',
        'supercategory': 'bottle',
    },
    {
        'id': 12,
        'name': 'bottle_redtea',
        'supercategory': 'bottle',
    },
    {
        'id': 13,
        'name': 'bottle_sour1',
        'supercategory': 'bottle',
    },
    {
        'id': 14,
        'name': 'bottle_sour2',
        'supercategory': 'bottle',
    },
    {
        'id': 15,
        'name': 'bottle_sprite',
        'supercategory': 'bottle',
    },
    {
        'id': 16,
        'name': 'bottle_yezhi',
        'supercategory': 'bottle',
    },
    {
        'id': 17,
        'name': 'cubic_green',
        'supercategory': 'bottle',
    },
    {
        'id': 18,
        'name': 'cup_orange',
        'supercategory': 'bottle',
    },
    {
        'id': 19,
        'name': 'drink_juice',
        'supercategory': 'bottle',
    },
    {
        'id': 20,
        'name': 'milk',
        'supercategory': 'bottle',
    },
    {
        'id': 21,
        'name': 'noodles_hongshao',
        'supercategory': 'bottle',
    },
    {
        'id': 22,
        'name': 'noodles_suancai',
        'supercategory': 'bottle',
    },
    {
        'id': 23,
        'name': 'tissue1',
        'supercategory': 'bottle',
    },
    {
        'id': 24,
        'name': 'tissue2',
        'supercategory': 'bottle',
    },
]

coord = []
for i in range(512):
    temp = []
    for j in range(512):
        temp.append([i,j])
    temp = np.array(temp)
    coord.append(temp)
coord = np.array(coord)

def rbbox_generator(mask_file, rbbox_id):
    # image_dir = "/Users/pro/Desktop/0_box.png"

    # ins_mask = cv2.imread(image_dir, cv2.IMREAD_GRAYSCALE)
    # colorimg = cv2.imread("/Users/pro/Desktop/0box.jpg")
    image_filename = mask_file.split('.')[-2]
    image_num = image_filename.split('_')[0]
    image_blur = image_filename.split('_')[1]
    img_file = image_num + image_blur + '.jpg'

    img = cv2.imread(img_dir + img_file)
    ins_mask = cv2.imread(ins_mask_dir + mask_file, cv2.IMREAD_GRAYSCALE)
    sem_mask = cv2.imread(sem_mask_dir + mask_file, cv2.IMREAD_GRAYSCALE)

    rbboxes = []
    gray_value = 1
    isexists = True

    while isexists:
        pixels = []
        isexists = False
        # for i in range(len(ins_mask)):
        #     temp = []
        #     for j in range(len(ins_mask)):
        #         temp.append([i,j])
        #         if ins_mask[i][j] == gray_value:
        #             pixels.append([i,j])
        #             isexists = True
        thisgray = np.zeros((512,512))
        thisgray.fill(gray_value)
        pixels = coord[(thisgray == ins_mask)]
        if len(pixels) == 0:
            isexists = False
        else:
            isexists = True

        if isexists:
            rbbox = cv2.minAreaRect(np.array(pixels))
            x,y = rbbox[0]
            w,h = rbbox[1]
            angle = rbbox[2] * math.pi / 180
            angle = -angle + math.pi / 2
            class_id = sem_mask[pixels[0][0]][pixels[0][1]]
            class_id = int(class_id)
            gray_value += 1
            rbbox_id += 1
            rbboxes.append([[x,y,w,h,angle], class_id, rbbox_id, image_num, image_blur])

        # ur = [y-h/2*math.sin(angle)+w/2*math.cos(angle), x+h/2*math.cos(angle)+w/2*math.sin(angle)]
        # dr = [y-h/2*math.sin(angle)-w/2*math.cos(angle), x+h/2*math.cos(angle)-w/2*math.sin(angle)]
        # ul = [y+h/2*math.sin(angle)+w/2*math.cos(angle), x-h/2*math.cos(angle)+w/2*math.sin(angle)]
        # dl = [y+h/2*math.sin(angle)-w/2*math.cos(angle), x-h/2*math.cos(angle)-w/2*math.sin(angle)]
        #
        # box = [dl,ul,ur,dr]
        # box = np.int0(box)

    return rbboxes

def json_file_generator():
    coco_output = {
        "info": INFO,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    image_id = 1
    rbbox_id = 1

    img_list = glob.glob(os.path.join(ins_mask_dir, '*.png'))
    for img_ in tqdm(img_list):
        img_name = ((img_.split('/')[-1]).split('.')[0]).split('_')[0] + ((img_.split('/')[-1]).split('.')[0]).split('_')[1] + '.jpg'
        img_file = img_.split('/')[-1]
        image_info = {"image_id": image_id, "image_name": img_name}
        coco_output["images"].append(image_info)
        rbboxes = rbbox_generator(img_file, rbbox_id)
        # print(rbboxes)
        for rbbox in rbboxes:
            coor, class_id, rbbox_id, image_num, image_blur = rbbox
            annotation = {"image_id": image_id, "box_id": rbbox_id, "rbbox": coor, "class_id": class_id}
            coco_output["annotations"].append(annotation)
        image_id += 1
    with open('{}black_bg_rbbox.json'.format(root_dir), 'w') as output_json_file:
        json.dump(coco_output, output_json_file)






















if __name__ == "__main__":
    json_file_generator()

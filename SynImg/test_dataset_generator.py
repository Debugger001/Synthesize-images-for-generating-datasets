import argparse
import glob
import sys
import os
from xml.etree.ElementTree import Element, SubElement, tostring
import xml.dom.minidom
import cv2
import numpy as np
import random
from PIL import Image
import scipy
from multiprocessing import Pool
from functools import partial
import signal
import time

from defaults import *
sys.path.insert(0, POISSON_BLENDING_DIR)
from pb import *
import math
from pyblur import *
from collections import namedtuple

def get_list_of_images(root_dir, N=1):
    '''Gets the list of images of objects in the root directory. The expected format
       is root_dir/<object>/<image>.jpg. Adds an image as many times you want it to
       appear in dataset.

    Args:
        root_dir(string): Directory where images of objects are present
        N(int): Number of times an image would appear in dataset. Each image should have
                different data augmentation
    Returns:
        list: List of images(with paths) that will be put in the dataset
    '''
    img_list = glob.glob(os.path.join(root_dir, '*/', '*.png'))
    # print(img_list)
    img_list_f = []
    for i in range(N):
        img_list_f = img_list_f + random.sample(img_list, len(img_list))
    return img_list_f

def get_labels(root_dir,imgs):
    '''Get list of labels/object names. Assumes the images in the root directory follow root_dir/<object>/<image>
       structure. Directory name would be object name.

    Args:
        imgs(list): List of images being used for synthesis
    Returns:
        list: List of labels/object names corresponding to each image
    '''

    labels = []
    for img_file in imgs:
        label = img_file.split('/')[-2]
        labels.append(label)
    print(len(labels))
    return labels

def write_labels_file(exp_dir, labels):
    '''Writes the labels file which has the name of an object on each line

    Args:
      exp_dir(string): Experiment directory where all the generated images, annotation and imageset
                         files will be stored
        labels(list): List of labels. This will be useful while training an object detector
    '''
    unique_labels = ['__background__'] + sorted(set(labels))
    with open(os.path.join(exp_dir,'labels.txt'),'w') as f:
        for i, label in enumerate(unique_labels):
            f.write('%s %s\n'%(i, label))

def gen_syn_data(img_files, labels, img_dir, anno_dir, scale_augment, rotation_augment, dontocclude, add_distractors):
    '''Creates list of objects and distrctor objects to be pasted on what images.
       Spawns worker processes and generates images according to given params

    Args:
        img_files(list): List of image files
        labels(list): List of labels for each image
        img_dir(str): Directory where synthesized images will be stored
        anno_dir(str): Directory where corresponding annotations will be stored
        scale_augment(bool): Add scale data augmentation
        rotation_augment(bool): Add rotation data augmentation
        dontocclude(bool): Generate images with occlusion
        add_distractors(bool): Add distractor objects whose annotations are not required
    '''
    w = WIDTH
    h = HEIGHT
    background_dir = BACKGROUND_DIR
    background_files = glob.glob(os.path.join(background_dir, BACKGROUND_GLOB_STRING))

    print("Number of background images : %s"%len(background_files) )
    img_labels = list(zip(img_files, labels))
    random.shuffle(img_labels)

    if add_distractors:
        with open(DISTRACTOR_LIST_FILE) as f:
            distractor_labels = [x.strip() for x in f.readlines()]

        distractor_list = []
        for distractor_label in distractor_labels:
            distractor_list += glob.glob(os.path.join(DISTRACTOR_DIR, distractor_label, DISTRACTOR_GLOB_STRING))

        distractor_files = list(zip(distractor_list, len(distractor_list)*[None]))
        print(distractor_files)
        random.shuffle(distractor_files)
    else:
        distractor_files = []
    print("List of distractor files collected: %s" % distractor_files)

    idx = 0
    img_files = []
    anno_files = []
    params_list = []
    while len(img_labels) > 0:
        # Get list of objects
        objects = []
        n = min(random.randint(MIN_NO_OF_OBJECTS, MAX_NO_OF_OBJECTS), len(img_labels))
        for _ in range(n):
            objects.append(img_labels.pop())
            print(objects)

        # Get list of distractor objects
        distractor_objects = []
        if add_distractors:
            n = min(random.randint(MIN_NO_OF_DISTRACTOR_OBJECTS, MAX_NO_OF_DISTRACTOR_OBJECTS), len(distractor_files))
            for _ in range(n):
                distractor_objects.append(random.choice(distractor_files))

        idx += 1
        bg_file = random.choice(background_files)
        for blur in BLENDING_LIST:
            img_file = os.path.join(img_dir, '%i_%s.jpg'%(idx,blur))
            anno_file = os.path.join(anno_dir, '%i.xml'%idx)
            params = (objects, distractor_objects, img_file, anno_file, bg_file)
            params_list.append(params)
            img_files.append(img_file)
            anno_files.append(anno_file)


    # partial_func = partial(create_image_anno_wrapper, w=w, h=h, scale_augment=scale_augment, rotation_augment=rotation_augment, blending_list=BLENDING_LIST, dontocclude=dontocclude)
    # p = Pool(NUMBER_OF_WORKERS, init_worker)
    # try:
    #     p.map(partial_func, params_list)
    # except KeyboardInterrupt:
    #     print("....\nCaught KeyboardInterrupt, terminating workers")
    #     p.terminate()
    # else:
    #     p.close()
    # p.join()
    return img_files, anno_files

root_dir = "/home/lc/syndata-generation/demo_data_dir/objects_dir/"
imgs = get_list_of_images(root_dir)
exp_dir = "/home/lc/syndata-generation/demo_data_dir/objects_dir/"
labels = get_labels(root_dir, imgs)

# print(get_list_of_images("/home/lc/syndata-generation/demo_data_dir/objects_dir/"))
# print(get_labels(root_dir, imgs))
# write_labels_file(exp_dir, labels)
gen_syn_data(imgs, labels, "/home/lc/syndata-generation/output_dir/images", "home/lc/syndata-generation/output_dir/annotations", False, False, False, True)

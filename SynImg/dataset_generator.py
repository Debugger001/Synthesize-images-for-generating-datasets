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

Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

gray_value_dict = {'__background__': 0}
gray_value = 1
dir_list = tuple(open('/home/lc/syndata-generation/demo_data_dir/objects_dir/dir_list.txt', 'r'))
for dir_list_itr in dir_list:
    gray_value_dict[dir_list_itr.rstrip()] = gray_value
    gray_value += 1

def randomAngle(kerneldim):
    """Returns a random angle used to produce motion blurring

    Args:
        kerneldim (int): size of the kernel used in motion blurring

    Returns:
        int: Random angle
    """
    kernelCenter = int(math.floor(kerneldim/2))
    numDistinctLines = kernelCenter * 4
    validLineAngles = np.linspace(0,180, numDistinctLines, endpoint = False)
    angleIdx = np.random.randint(0, len(validLineAngles))
    return int(validLineAngles[angleIdx])

def LinearMotionBlur3C(img):
    """Performs motion blur on an image with 3 channels. Used to simulate
       blurring caused due to motion of camera.

    Args:
        img(NumPy Array): Input image with 3 channels

    Returns:
        Image: Blurred image by applying a motion blur with random parameters
    """
    lineLengths = [3,5,7,9]
    lineTypes = ["right", "left", "full"]
    lineLengthIdx = np.random.randint(0, len(lineLengths))
    lineTypeIdx = np.random.randint(0, len(lineTypes))
    lineLength = lineLengths[lineLengthIdx]
    lineType = lineTypes[lineTypeIdx]
    lineAngle = randomAngle(lineLength)
    blurred_img = img
    for i in range(3):
        blurred_img[:,:,i] = PIL2array1C(LinearMotionBlur(img[:,:,i], lineLength, lineAngle, lineType))
    # sem_mask = LinearMotionBlur(sem_mask, lineLength, lineAngle, lineType)
    # ins_mask = LinearMotionBlur(ins_mask, lineLength, lineAngle, lineType)
    blurred_img = Image.fromarray(blurred_img, 'RGB')
    return blurred_img

def overlap(a, b):
    '''Find if two bounding boxes are overlapping or not. This is determined by maximum allowed
       IOU between bounding boxes. If IOU is less than the max allowed IOU then bounding boxes
       don't overlap

    Args:
        a(Rectangle): Bounding box 1
        b(Rectangle): Bounding box 2
    Returns:
        bool: True if boxes overlap else False
    '''
    dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)

    if (dx>=0) and (dy>=0) and float(dx*dy) > MAX_ALLOWED_IOU*(a.xmax-a.xmin)*(a.ymax-a.ymin):
        return True
    else:
        return False

def cover(a, b):
    '''
    Args:
        a(Rectangle): Bounding box 1
        b(Rectangle): Bounding box 2
    Returns:
        bool: True if boxes cover else False
    '''
    if (a.xmax > b.xmax) and (a.ymax > b.ymax) and (a.xmin < b.xmin) and (a.ymin < b.ymin):
        return True
    elif (b.xmax > a.xmax) and (b.ymax > a.ymax) and (b.xmin < a.xmin) and (b.ymin < a.ymin):
        return True
    else:
        return False


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
    img_list_f = []
    for i in range(N):
        img_list_f = img_list_f + random.sample(img_list, len(img_list))
    return img_list_f

def get_mask_file(img_file):
    '''Takes an image file name and returns the corresponding mask file. The mask represents
       pixels that belong to the object. Default implentation assumes mask file has same path
       as image file with different extension only. Write custom code for getting mask file here
       if this is not the case.

    Args:
        img_file(string): Image name
    Returns:
        string: Correpsonding mask file path
    '''
    mask_file = img_file.replace('.png','.pbm')
    return mask_file

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
    return labels

def get_annotation_from_mask_file(mask_file, scale=1.0):
    '''Given a mask file and scale, return the bounding box annotations

    Args:
        mask_file(string): Path of the mask file
    Returns:
        tuple: Bounding box annotation (xmin, xmax, ymin, ymax)
    '''
    if os.path.exists(mask_file):
        mask = cv2.imread(mask_file)
        if INVERTED_MASK:
            mask = 255 - mask
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if len(np.where(rows)[0]) > 0:
            ymin, ymax = np.where(rows)[0][[0, -1]]
            xmin, xmax = np.where(cols)[0][[0, -1]]
            return int(scale*xmin), int(scale*xmax), int(scale*ymin), int(scale*ymax)
        else:
            return -1, -1, -1, -1
    else:
        print("%s not found. Using empty mask instead."%mask_file)
        return -1, -1, -1, -1

def get_annotation_from_mask(mask):
    '''Given a mask, this returns the bounding box annotations

    Args:
        mask(NumPy Array): Array with the mask
    Returns:
        tuple: Bounding box annotation (xmin, xmax, ymin, ymax)
    '''
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if len(np.where(rows)[0]) > 0:
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        return xmin, xmax, ymin, ymax
    else:
        return -1, -1, -1, -1

def write_imageset_file(exp_dir, img_files, anno_files):
    '''Writes the imageset file which has the generated images and corresponding annotation files
       for a given experiment

    Args:
        exp_dir(string): Experiment directory where all the generated images, annotation and imageset
                         files will be stored
        img_files(list): List of image files that were generated
        anno_files(list): List of annotation files corresponding to each image file
    '''
    with open(os.path.join(exp_dir,'train.txt'),'w') as f:
        for i in range(len(img_files)):
            f.write('%s %s\n'%(img_files[i], anno_files[i]))

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

def keep_selected_labels(img_files, labels):
    '''Filters image files and labels to only retain those that are selected. Useful when one doesn't
       want all objects to be used for synthesis

    Args:
        img_files(list): List of images in the root directory
        labels(list): List of labels corresponding to each image
    Returns:
        new_image_files(list): Selected list of images
        new_labels(list): Selected list of labels corresponidng to each imahe in above list
    '''
    with open(SELECTED_LIST_FILE) as f:
        selected_labels = [x.strip() for x in f.readlines()]
    new_img_files = []
    new_labels = []
    for i in range(len(img_files)):
        if labels[i] in selected_labels:
            new_img_files.append(img_files[i])
            new_labels.append(labels[i])
    return new_img_files, new_labels

def PIL2array1C(img):
    '''Converts a PIL image to NumPy Array

    Args:
        img(PIL Image): Input PIL image
    Returns:
        NumPy Array: Converted image
    '''
    # print(img.size[0])
    # print(img.size[1])
    return np.array(img.getdata(),
                    np.uint8).reshape(img.size[1], img.size[0])

def PIL2array3C(img):
    '''Converts a PIL image to NumPy Array

    Args:
        img(PIL Image): Input PIL image
    Returns:
        NumPy Array: Converted image
    '''
    # print(img.size)
    return np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0], 3)

def count_pixel_value(img):
    img = np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0])
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
    return pixel_value

def count_pixel_num(img):
    img = np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0])
    counter = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] > 0:
                counter += 1
    return counter


def create_image_anno_wrapper(args, w=WIDTH, h=HEIGHT, scale_augment=False, rotation_augment=False, blending_list=['none'], dontocclude=False):
   ''' Wrapper used to pass params to workers
   '''
   # print("???")
   return create_image_anno(*args, w, h, scale_augment, rotation_augment, blending_list, dontocclude)
# def _addAlpha(self,image)
#    temp_image = []
#    for i in xrange(image.shape[0]):
#    	   for j in xrange(image.shape[1]):
#    	   	  temp = np.append(image[i][j],255)
#    	   	  temp_image.append(temp)

#    	return np.array(temp_image)

def create_image_anno(objects, distractor_objects, img_file, semantic_file, instance_file, anno_file, w=WIDTH, h=HEIGHT, scale_augment=False, rotation_augment=False, blending_list=['none'], dontocclude=True):
    '''Add data augmentation, synthesizes images and generates annotations according to given parameters

    Args:
        objects(list): List of objects whose annotations are also important
        distractor_objects(list): List of distractor objects that will be synthesized but whose annotations are not required
        img_file(str): Image file name
        anno_file(str): Annotation file name
        bg_file(str): Background image path
        w(int): Width of synthesized image
        h(int): Height of synthesized image
        scale_augment(bool): Add scale data augmentation
        rotation_augment(bool): Add rotation data augmentation
        blending_list(list): List of blending modes to synthesize for each image
        dontocclude(bool): Generate images with occlusion
    '''
    if 'none' not in img_file:
        return

    print("Working on %s" % img_file)

    if os.path.exists(img_file):
        return img_file

    generated_files = glob.glob(os.path.join('/home/lc/syndata-generation/output_dir/val/', '*.jpg'))
    print("Already synthesized images: %s"%(len(generated_files)) )

    all_objects = objects + distractor_objects

    gray_value_offset = 1

    while True:
        # top = Element('annotation')

        background_dir = BACKGROUND_DIR
        background_files = glob.glob(os.path.join(background_dir, BACKGROUND_GLOB_STRING))
        backgrounds = []
        sem_backgrounds = []
        ins_backgrounds = []


        for i in range(len(blending_list)):
            while True:
                bg_file = random.choice(background_files)
                background = Image.open(bg_file)
                if len(background.getbands()) == 3:
                    break
            background = background.resize((w, h), Image.ANTIALIAS)
            backgrounds.append(background.copy())
            sem_backgrounds_blur = []
            ins_backgrounds_blur = []
            for o in range(len(objects)):
                sem_backgrounds_blur.append(Image.fromarray(np.zeros((512, 512), dtype=np.uint8)))
                ins_backgrounds_blur.append(Image.fromarray(np.zeros((512, 512), dtype=np.uint8)))
            sem_backgrounds.append(sem_backgrounds_blur)
            ins_backgrounds.append(ins_backgrounds_blur)

        already_syn = []
        obj_num = 0
        obj_already_syn = {'__background__': 0}
        dir_list = tuple(open('/home/lc/syndata-generation/demo_data_dir/objects_dir/dir_list.txt', 'r'))
        for dir_list_itm in dir_list:
            obj_already_syn[dir_list_itm.rstrip()] = 0
        for idx, obj in enumerate(all_objects):
            foreground = Image.open(obj[0])
            if len(foreground.getpixel((0,0))) == 4:
                r,g,b,a = foreground.split()
            elif len(foreground.getpixel((0,0))) == 3:
                r,g,b = foreground.split()
            else:
                print("Neither .png nor .jpg...")
            foreground = Image.merge('RGB',(r,g,b))
            xmin, xmax, ymin, ymax = get_annotation_from_mask_file(get_mask_file(obj[0]))
            if xmin == -1 or ymin == -1 or xmax-xmin < MIN_WIDTH or ymax-ymin < MIN_HEIGHT :
                continue
            foreground = foreground.crop((xmin, ymin, xmax, ymax))
            orig_w, orig_h = foreground.size
            mask_file =  get_mask_file(obj[0])
            mask = Image.open(mask_file)
            mask = mask.crop((xmin, ymin, xmax, ymax))
            if INVERTED_MASK:
                mask = Image.fromarray(255-PIL2array1C(mask))

            o_w, o_h = orig_w, orig_h
            if scale_augment:
                while True:
                    scale = random.uniform(MIN_SCALE, MAX_SCALE)
                    o_w, o_h = int(scale*orig_w), int(scale*orig_h)
                    if  w-o_w > 0 and h-o_h > 0 and o_w > 0 and o_h > 0:
                        break
                foreground = foreground.resize((o_w, o_h), Image.ANTIALIAS)
                mask = mask.resize((o_w, o_h))
            if rotation_augment:
                max_degrees = MAX_DEGREES
                while True:
                    rot_degrees = random.randint(-max_degrees, max_degrees)
                    foreground_tmp = foreground.rotate(rot_degrees, expand=True)
                    mask_tmp = mask.rotate(rot_degrees, expand=True)
                    o_w, o_h = foreground_tmp.size
                    if  w-o_w > 0 and h-o_h > 0:
                        break
                mask = mask_tmp
                foreground = foreground_tmp
            xmin, xmax, ymin, ymax = get_annotation_from_mask(mask)
            attempt = 0
            while True:
                attempt +=1
                x = random.randint(int(-MAX_TRUNCATION_FRACTION*o_w), int(w-o_w+MAX_TRUNCATION_FRACTION*o_w))
                y = random.randint(int(-MAX_TRUNCATION_FRACTION*o_h), int(h-o_h+MAX_TRUNCATION_FRACTION*o_h))
                if dontocclude:
                    found = True
                    for prev in already_syn:
                        ra = Rectangle(prev[0], prev[2], prev[1], prev[3])
                        rb = Rectangle(x+xmin, y+ymin, x+xmax, y+ymax)
                        if overlap(ra, rb):
                            found = False
                            break

                qualified = False
                for prev in already_syn:
                    ra = Rectangle(prev[0], prev[2], prev[1], prev[3])
                    rb = Rectangle(x+xmin, y+ymin, x+xmax, y+ymax)
                    if cover(ra, rb):
                        break
                    else:
                        qualified = True
                        break
                if qualified:
                    break

                if attempt == MAX_ATTEMPTS_TO_SYNTHESIZE:
                    break

            already_syn.append([x+xmin, x+xmax, y+ymin, y+ymax])

            if obj[1] in gray_value_dict:
                sem_mask_np = PIL2array1C(mask)
                sem_mask = np.zeros((sem_mask_np.shape[0], sem_mask_np.shape[1]), dtype=np.uint8)
                for sem_mask_rows in range(0, sem_mask_np.shape[0]):
                    for sem_mask_cols in range(0, sem_mask_np.shape[1]):
                        if sem_mask_np[sem_mask_rows][sem_mask_cols] > 0:
                            sem_mask[sem_mask_rows][sem_mask_cols] = gray_value_dict[obj[1]]
                sem_mask_PIL = Image.fromarray(sem_mask)
                # print("%i pixel values in the mask"%len(count_pixel_value(sem_mask_PIL)))
                # print("The pixels are: ")
                # print(count_pixel_value(sem_mask_PIL))
                # print("%i objects in the image"%len(objects))

                # print(sem_mask_PIL)

                ins_mask_np = PIL2array1C(mask)
                ins_mask = np.zeros((ins_mask_np.shape[0], ins_mask_np.shape[1]), dtype=np.uint8)
                for ins_mask_rows in range(0, ins_mask_np.shape[0]):
                    for ins_mask_cols in range(0, ins_mask_np.shape[1]):
                        if ins_mask_np[ins_mask_rows][ins_mask_cols] > 0:
                                # ins_mask[ins_mask_rows][ins_mask_cols] = gray_value_offset
                                ins_mask[ins_mask_rows][ins_mask_cols] = 255
                # gray_value_offset += 1
                ins_mask_PIL = Image.fromarray(ins_mask)

                paste_mask = mask
                paste_mask_np = PIL2array1C(mask)



            # if obj[1] in gray_value_dict:
            #     sem_mask_np = PIL2array1C(mask)
            #     sem_mask = np.zeros((sem_mask_np.shape[0], sem_mask_np.shape[1], 3), dtype=np.uint8)
            #     for sem_mask_rows in range(0, sem_mask_np.shape[0]):
            #         for sem_mask_cols in range(0, sem_mask_np.shape[1]):
            #             if sem_mask_np[sem_mask_rows][sem_mask_cols] > 0:
            #                 sem_mask[sem_mask_rows][sem_mask_cols] = Color[obj[1]]
            #     sem_mask_PIL = Image.fromarray(sem_mask)

            # print("%i pixel values in the mask"%count_pixel_value(mask))
            # print("%i objects in the image"%len(objects))

            if obj[1] in gray_value_dict:
                obj_already_syn[obj[1]] += 1


            for i in range(len(blending_list)):
                if blending_list[i] == 'none' or blending_list[i] == 'motion':

                    backgrounds[i].paste(foreground, (x, y), mask)

                    if obj[1] in gray_value_dict:
                        # sem_backgrounds[i][obj_num].paste(sem_mask_PIL, (x, y), mask)
                        ins_backgrounds[i][obj_num].paste(ins_mask_PIL, (x, y), mask)

                elif blending_list[i] == 'poisson':
                    offset = (y, x)
                    img_mask = PIL2array1C(mask)
                    # sem_mask_mask = PIL2array1C(mask)
                    # ins_mask_mask = PIL2array1C(mask)
                    img_src = PIL2array3C(foreground).astype(np.float64)
                    # sem_mask_src = PIL2array1C(sem_mask_PIL).astype(np.float64)
                    # ins_mask_src = PIL2array1C(ins_mask_PIL).astype(np.float64)

                    img_target = PIL2array3C(backgrounds[i])
                    # sem_mask_target = PIL2array1C(sem_backgrounds[i])
                    # ins_mask_target = PIL2array1C(ins_backgrounds[i])

                    img_mask, img_src, offset_adj \
                        = create_mask(img_mask.astype(np.float64),
                          img_target, img_src, offset=offset)
                    # sem_mask_mask, sem_mask_src, offset_adj \
                    #     = create_mask(sem_mask_mask.astype(np.float64),
                    #       sem_mask_target, sem_mask_src, offset=offset)
                    # ins_mask_mask, ins_mask_src, offset_adj \
                    #     = create_mask(ins_mask_mask.astype(np.float64),
                    #       ins_mask_target, ins_mask_src, offset=offset)

                    background_array = poisson_blend(img_mask, img_src, img_target,
                                    method='normal', offset_adj=offset_adj)
                    # sem_background_array = poisson_blend_1C(sem_mask_mask, sem_mask_src, sem_mask_target,
                                    # method='normal', offset_adj=offset_adj)
                    # ins_background_array = poisson_blend_1C(ins_mask_mask, ins_mask_src, ins_mask_target,
                                    # method='normal', offset_adj=offset_adj)

                    backgrounds[i] = Image.fromarray(background_array, 'RGB')
                    if obj[1] in gray_value_dict:
                        # sem_backgrounds[i][obj_num].paste(sem_mask_PIL, (x, y), mask)
                        ins_backgrounds[i][obj_num].paste(ins_mask_PIL, (x, y), mask)
                    # sem_backgrounds[i] = Image.fromarray(sem_background_array)
                    # ins_backgrounds[i] = Image.fromarray(ins_background_array)

                elif blending_list[i] == 'gaussian':
                    Gaussian_mask = Image.fromarray(cv2.GaussianBlur(PIL2array1C(mask),(5,5),2))

                    backgrounds[i].paste(foreground, (x, y), Gaussian_mask)
                    Gaussian_mask_np = np.array(Gaussian_mask.getdata(), np.uint8).reshape(Gaussian_mask.size[1], Gaussian_mask.size[0])
                    for G_i in range(Gaussian_mask_np.shape[0]):
                        for G_j in range(Gaussian_mask_np.shape[1]):
                            if Gaussian_mask_np[G_i][G_j] >= 100:
                                Gaussian_mask_np[G_i][G_j] = 255
                            else:
                                Gaussian_mask_np[G_i][G_j] = 0
                    mask_Gaussian_mask = Image.fromarray(Gaussian_mask_np)
                    if obj[1] in gray_value_dict:
                        # sem_backgrounds[i][obj_num].paste(sem_mask_PIL, (x, y), mask_Gaussian_mask)
                        ins_backgrounds[i][obj_num].paste(ins_mask_PIL, (x, y), mask_Gaussian_mask)

                elif blending_list[i] == 'box':
                    Box_mask = Image.fromarray(cv2.blur(PIL2array1C(mask),(3,3)))

                    backgrounds[i].paste(foreground, (x, y), Box_mask)
                    Box_mask_np = np.array(Box_mask.getdata(), np.uint8).reshape(Box_mask.size[1], Box_mask.size[0])
                    for B_i in range(Box_mask_np.shape[0]):
                        for B_j in range(Box_mask_np.shape[1]):
                            if Box_mask_np[B_i][B_j] >= 100:
                                Box_mask_np[B_i][B_j] = 255
                            else:
                                Box_mask_np[B_i][B_j] = 0
                    mask_Box_mask = Image.fromarray(Box_mask_np)
                    if obj[1] in gray_value_dict:
                        # sem_backgrounds[i][obj_num].paste(sem_mask_PIL, (x, y), mask_Box_mask)
                        ins_backgrounds[i][obj_num].paste(ins_mask_PIL, (x, y), mask_Box_mask)
                if blending_list[i] == 'motion':
                    backgrounds[i] = LinearMotionBlur3C(PIL2array3C(backgrounds[i]))
                if obj[1] in gray_value_dict:
                    ins_backgrounds[i][obj_num].save((instance_file + '_' + obj[1] + '_' + str(obj_already_syn[obj[1]]) + '.png').replace('none', blending_list[i]))
            if obj[1] in gray_value_dict:
                obj_num += 1

        if attempt == MAX_ATTEMPTS_TO_SYNTHESIZE:
            continue
        else:
            break
    for i in range(len(blending_list)):
        # print("???")
        backgrounds[i].save(img_file.replace('none', blending_list[i]))
        # print(obj_num)
        # if idx >= len(objects):
        #     continue
        # object_root = SubElement(top, 'object')
        # object_type = obj[1]
        # object_type_entry = SubElement(object_root, 'name')
        # object_type_entry.text = str(object_type)
        # object_bndbox_entry = SubElement(object_root, 'bndbox')
        # x_min_entry = SubElement(object_bndbox_entry, 'xmin')
        # x_min_entry.text = '%d'%(max(1,x+xmin))
        # x_max_entry = SubElement(object_bndbox_entry, 'xmax')
        # x_max_entry.text = '%d'%(min(w,x+xmax))
        # y_min_entry = SubElement(object_bndbox_entry, 'ymin')
        # y_min_entry.text = '%d'%(max(1,y+ymin))
        # y_max_entry = SubElement(object_bndbox_entry, 'ymax')
        # y_max_entry.text = '%d'%(min(h,y+ymax))
        # difficult_entry = SubElement(object_root, 'difficult')
        # difficult_entry.text = '0' # Add heuristic to estimate difficulty later on
    print(img_file + ' done generating.')

    # for i in range(len(blending_list)):
    #     if blending_list[i] == 'motion':
    #         backgrounds[i] = LinearMotionBlur3C(PIL2array3C(backgrounds[i]), PIL2array1C(sem_backgrounds[i]), PIL2array1C(ins_backgrounds[i]))
    #
    #     # print("???")
    #     backgrounds[i].save(img_file.replace('none', blending_list[i]))
    #
    #     sem_backgrounds[i].save(semantic_file.replace('none', blending_list[i]))
    #     # print("%i pixel values in the image"%count_pixel_value(sem_backgrounds[i]))
    #     # print("%i objects in the image"%len(objects))
    #
    #     ins_backgrounds[i].save(instance_file.replace('none', blending_list[i]))

    # print("XML...")
    # xmlstr = xml.dom.minidom.parseString(tostring(top)).toprettyxml(indent="    ")
    # with open(anno_file, "w") as f:
        # f.write(xmlstr)

def gen_syn_data(image_files, labels, img_dir, anno_dir, sem_mask_dir, ins_mask_dir, scale_augment, rotation_augment, dontocclude, add_distractors):
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
    img_labels = list(zip(image_files, labels))
    random.shuffle(img_labels)

    if add_distractors:
        with open(DISTRACTOR_LIST_FILE) as f:
            distractor_labels = [x.strip() for x in f.readlines()]

        distractor_list = []
        for distractor_label in distractor_labels:
            distractor_list += glob.glob(os.path.join(DISTRACTOR_DIR, distractor_label, DISTRACTOR_GLOB_STRING))

        distractor_files = list(zip(distractor_list, len(distractor_list)*[None]))
        random.shuffle(distractor_files)
    else:
        distractor_files = []
    # print("List of distractor files collected: %s" % distractor_files)

    idx = 0
    img_files = []
    semantic_files = []
    instance_files = []
    anno_files = []
    params_list = []

    generation_num = 0

    while generation_num < 4000:
        # Get list of objects
        objects = []
        img_labels = list(zip(image_files, labels))
        random.shuffle(img_labels)
        n = min(random.randint(MIN_NO_OF_OBJECTS, MAX_NO_OF_OBJECTS), len(img_labels))
        for _ in range(n):
            objects.append(img_labels.pop())

        # Get list of distractor objects
        distractor_objects = []
        if add_distractors:
            n = min(random.randint(MIN_NO_OF_DISTRACTOR_OBJECTS, MAX_NO_OF_DISTRACTOR_OBJECTS), len(distractor_files))
            for _ in range(n):
                distractor_objects.append(random.choice(distractor_files))



        for blur in BLENDING_LIST:
            img_file = os.path.join(img_dir, '%i%s.jpg'%(idx,blur))
            semantic_file = os.path.join(sem_mask_dir, '%i_%s'%(idx,blur))
            instance_file = os.path.join(ins_mask_dir, '%i%s'%(idx,blur))
            anno_file = os.path.join(anno_dir, '%i.xml'%idx)
            params = (objects, distractor_objects, img_file, semantic_file, instance_file, anno_file)
            params_list.append(params)
            img_files.append(img_file)
            semantic_files.append(semantic_file)
            instance_files.append(instance_file)
            anno_files.append(anno_file)

        idx += 1
        generation_num += 1


    # print("Third...")
    partial_func = partial(create_image_anno_wrapper, w=w, h=h, scale_augment=scale_augment, rotation_augment=rotation_augment, blending_list=BLENDING_LIST, dontocclude=dontocclude)
    p = Pool(NUMBER_OF_WORKERS, init_worker)
    try:
        p.map(partial_func, params_list)
    except KeyboardInterrupt:
        print("....\nCaught KeyboardInterrupt, terminating workers")
        p.terminate()
    else:
        print("Wrong...")
        p.close()
    p.join()
    # for i in params_list:
    #     create_image_anno_wrapper(i, w, h, scale_augment, rotation_augment, BLENDING_LIST, dontocclude)
    return img_files, anno_files


def init_worker():
    '''
    Catch Ctrl+C signal to termiante workers
    '''
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def generate_synthetic_dataset(args):
    ''' Generate synthetic dataset according to given args
    '''
    img_files = get_list_of_images(args.root, args.num)
    labels = get_labels(args.root,img_files)

    if args.selected:
       img_files, labels = keep_selected_labels(img_files, labels)

    if not os.path.exists(args.exp):
        os.makedirs(args.exp)

    if args.auto_label:
        write_labels_file(args.exp, labels)

    anno_dir = os.path.join(args.exp, 'IamUseless')
    img_dir = os.path.join(args.exp, 'val')
    sem_mask_dir = os.path.join(args.exp, 'DeleteMe')
    # ins_mask_dir = os.path.join(args.exp, 'Instance_segmentation_class')
    ins_mask_dir = os.path.join(args.exp, 'annotations')
    if not os.path.exists(os.path.join(anno_dir)):
        os.makedirs(anno_dir)
    if not os.path.exists(os.path.join(img_dir)):
        os.makedirs(img_dir)
    if not os.path.exists(os.path.join(sem_mask_dir)):
        os.makedirs(sem_mask_dir)
    if not os.path.exists(os.path.join(ins_mask_dir)):
        os.makedirs(ins_mask_dir)
    # print("First...")
    gen_syn_data(img_files, labels, img_dir, anno_dir, sem_mask_dir, ins_mask_dir, args.scale, args.rotation, args.dontocclude, args.add_distractors)
    #syn_img_files, anno_files = gen_syn_data(img_files, labels, img_dir, anno_dir, args.scale, args.rotation, args.dontocclude, args.add_distractors)

    #write_imageset_file(args.exp, syn_img_files, anno_files)


def parse_args():
    '''Parse input arguments
    '''
    parser = argparse.ArgumentParser(description="Create dataset with different augmentations")
    parser.add_argument("--root", default = "/home/lc/syndata-generation/demo_data_dir/objects_dir/",
      help="The root directory which contains the images and annotations.")
    parser.add_argument("--exp", default = "/home/lc/syndata-generation/output_dir",
      help="The directory where images and annotation lists will be created.")
    parser.add_argument("--selected",
      help="Keep only selected instances in the test dataset. Default is to keep all instances in the root directory", action="store_true")
    parser.add_argument("--scale",
      help="Add scale augmentation.Default is to add scale augmentation.", action="store_false")
    parser.add_argument("--rotation",
      help="Add rotation augmentation.Default is to add rotation augmentation.", action="store_false")
    parser.add_argument("--num",
      help="Number of times each image will be in dataset", default=10, type=int)
    parser.add_argument("--dontocclude", default=True, type=bool,
      help="Add objects without occlusion. Default is to produce occlusions")
    parser.add_argument("--add_distractors",
      help="Add distractors objects. Default is to not use distractors", default=True, type=bool)
    parser.add_argument("--auto_label", help="if True it will  cat_id for each category, else use customized labels", default=True, type=bool)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print(args)
    generate_synthetic_dataset(args)

import os
import cv2
import glob
from tqdm import tqdm
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from skimage.measure import label, regionprops
root_dir = '/Users/pro/Desktop/Lab/Syndata/syndata-generation-10-11/mask/'
train_image_dir = os.path.join(root_dir, 'images')
test_image_dir = os.path.join(root_dir, 'images')
np.set_printoptions(threshold=np.inf)

from skimage.morphology import label

def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction

def masks_as_image(in_mask_list, all_masks=None):
    # Take the individual ship masks and create a single mask array for all ships
    if all_masks is None:
        all_masks = np.zeros((768, 768), dtype = np.int16)
    #if isinstance(in_mask_list, list):
    for mask in tqdm(in_mask_list):
        if isinstance(mask, str):
            all_masks += rle_decode(mask)
    return np.expand_dims(all_masks, -1)


# A = [1,1,1,1,1,1,0,0,1,0,0,0,0,1,1,0,1,0,0,0,0,0,0,1,0]
A = cv2.imread("/Users/pro/Desktop/Lab/Syndata/syndata-generation-10-11/images/0gaussian.jpg")
mask_list = rle_encode(A)
# print(mask_list)
mask_0 = masks_as_image(mask_list)
print(mask_0)

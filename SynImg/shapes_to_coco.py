#!/usr/bin/env python3

import datetime
import json
import os
import re
import fnmatch
from PIL import Image
import numpy as np
from pycococreatortools import pycococreatortools

ROOT_DIR = '/home/dynamite/transfer_to_coco'
#IMAGE_DIR = os.path.join(ROOT_DIR, "shapes_train2018")
#ANNOTATION_DIR = os.path.join(ROOT_DIR, "annotations")
IMAGE_DIR = '/home/lc/syndata-generation/output_dir/train'
ANNOTATION_DIR = '/home/lc/syndata-generation/output_dir/anno'

INFO = {
    "description": "Example Dataset",
    "url": "https://github.com/waspinator/pycococreator",
    "version": "0.1.0",
    "year": 2018,
    "contributor": "fyj",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]

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
        'name': 'bottle_pepsi1',
        'supercategory': 'bottle',
    },
    {
        'id': 12,
        'name': 'bottle_pepsi2',
        'supercategory': 'bottle',
    },
    {
        'id': 13,
        'name': 'bottle_redtea',
        'supercategory': 'bottle',
    },
    {
        'id': 14,
        'name': 'bottle_sour1',
        'supercategory': 'bottle',
    },
    {
        'id': 15,
        'name': 'bottle_sour2',
        'supercategory': 'bottle',
    },
    {
        'id': 16,
        'name': 'bottle_sprite',
        'supercategory': 'bottle',
    },
    {
        'id': 17,
        'name': 'bottle_yezhi',
        'supercategory': 'bottle',
    },
    {
        'id': 18,
        'name': 'cup_orange',
        'supercategory': 'bottle',
    },
    {
        'id': 19,
        'name': 'tissue1',
        'supercategory': 'bottle',
    },
    {
        'id': 20,
        'name': 'tissue2',
        'supercategory': 'bottle',
    },
    {
        'id': 21,
        'name': 'drink_juice',
        'supercategory': 'bottle',
    },
    {
        'id': 22,
        'name': 'noodles_hongshao',
        'supercategory': 'bottle',
    },
    {
        'id': 23,
        'name': 'noodles_suancai',
        'supercategory': 'bottle',
    },
    {
        'id': 24,
        'name': 'milk',
        'supercategory': 'bottle',
    },
    {
        'id': 25,
        'name': 'cubic_green',
        'supercategory': 'bottle',
    },
]

def filter_for_jpeg(root, files):
    file_types = ['*.jpeg', '*.jpg']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]

    return files

def filter_for_annotations(root, files, image_filename):
    file_types = ['*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    basename_no_extension = os.path.splitext(os.path.basename(image_filename))[0]
    file_name_prefix = basename_no_extension + '.*'
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    files = [f for f in files if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])]

    return files

def main():

    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    image_id = 1
    segmentation_id = 1

    # filter for jpeg images
    for root, _, files in os.walk(IMAGE_DIR):
        image_files = filter_for_jpeg(root, files)

        # go through each image
        for image_filename in image_files:
            image = Image.open(image_filename)
            image_info = pycococreatortools.create_image_info(
                image_id, os.path.basename(image_filename), image.size)
            coco_output["images"].append(image_info)

            # filter for associated png annotations
            for root, _, files in os.walk(ANNOTATION_DIR):
                annotation_files = filter_for_annotations(root, files, image_filename)

                # go through each associated annotation
                for annotation_filename in annotation_files:

                    print(annotation_filename)
                    class_id = [x['id'] for x in CATEGORIES if x['name'] in annotation_filename][0]

                    category_info = {'id': class_id, 'is_crowd': 'crowd' in image_filename}
                    binary_mask = np.asarray(Image.open(annotation_filename)
                        .convert('1')).astype(np.uint8)

                    annotation_info = pycococreatortools.create_annotation_info(
                        segmentation_id, image_id, category_info, binary_mask,
                        image.size, tolerance=2)

                    if annotation_info is not None:
                        coco_output["annotations"].append(annotation_info)

                    segmentation_id = segmentation_id + 1

            image_id = image_id + 1

    with open('{}/instances_bottle_train2018.json'.format(ROOT_DIR), 'w') as output_json_file:
        json.dump(coco_output, output_json_file)


if __name__ == "__main__":
    main()

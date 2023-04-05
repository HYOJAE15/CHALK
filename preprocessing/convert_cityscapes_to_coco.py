import cv2
import numpy as np 
import os
import json 

from glob import glob 
from tqdm import tqdm

from copy import deepcopy

from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries

from skimage.measure import label, regionprops

from pycococreatortools import create_image_info, create_annotation_info

import datetime
import shutil

# global arguments 

INFO = {
    "description": "Concrete Damage Dataset",
    "version": "1.0",
    "year": 2018,
    "contributor": "UOS-SSaS",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]

CATEGORIES = []

SOURCE_DIR = '//172.16.113.151/UOS-SSaS Dropbox/05. Data/02. Training&Test/013. General Concrete Damage/cityscapes/v0.1.1'

TARGET_DIR = '//172.16.113.151/UOS-SSaS Dropbox/05. Data/02. Training&Test/013. General Concrete Damage/coco_crack_only/v0.1.2'

SUBSET = 'val' # choose one of 'train', 'val', 'test'

CLASSES = ['crack'] #, 'effl', 'rebar', 'spall']

CONVERT_STYLE = {
    'crack' : 'overlap',
    'effl' : 'normal',
    'rebar' : 'overlap',
    'spall' : 'normal'
}

CATEGORIES = []

WINDOW_SIZE = 128

OVERLAP = 0.5

for i, name in enumerate(CLASSES):
    cat = {'id': i, 'name': name, 'supercategory': 'concrete_damage'}
    CATEGORIES.append(cat)

def create_normal_annotation(coco_output, gt, img_id, width, height, class_idx, segmentation_id):
    """
    Create normal annotation (without overalapping)
    Args:
        coco_output: coco output
        gt: ground truth
        img_id: image id
        width: width of the image
        height: height of the image
        class_idx: class index
        segmentation_id: segmentation id

    Returns:
        coco_output: coco output
        segmentation_id: segmentation id
    """

    gt_label = label(gt == class_idx)

    category_info = {'id': class_idx - 1, 'is_crowd': 0}

    for label_idx in range(1, np.max(gt_label)+1):
        binary_mask = gt_label == label_idx

        annotation_info = create_annotation_info(
            segmentation_id, img_id, category_info, binary_mask, (width, height), tolerance=2
        )

        if annotation_info is not None:
            coco_output["annotations"].append(annotation_info)

        segmentation_id += 1

    return coco_output, segmentation_id

def create_overlap_annotation(coco_output, gt, img_id, width, height, class_idx, segmentation_id, window_size = 256, overlap = 0.5):
    """
    Create overlap annotation
    Args:
        coco_output: coco output
        gt: ground truth
        img_id: image id
        width: width of the image
        height: height of the image
        class_idx: class index
        segmentation_id: segmentation id
        window_size: window size
        overlap: overlap ratio

    Returns:
        coco_output: coco output
        segmentation_id: segmentation id
    """
    gt_label = label(gt == class_idx)
    # get regionprops
    props = regionprops(gt_label)

    category_info = {'id': class_idx - 1, 'is_crowd': 0}

    for label_idx in range(1, np.max(gt_label)+1):
        binary_mask = gt_label == label_idx

        # extract bounding box
        minr, minc, maxr, maxc = props[label_idx-1].bbox

        # check bounding box size
        # if bounding box is smaller than 10, skip
        if (maxr - minr) < 10 and (maxc - minc) < 10:
            pass

        # else if bounding box is smaller than window size, create normal annotation
        elif (maxr - minr) < window_size and (maxc - minc) < window_size:
            annotation_info = create_annotation_info(
            segmentation_id, img_id, category_info, binary_mask, (width, height), tolerance=2
            )

            if annotation_info is not None:
                coco_output["annotations"].append(annotation_info)

            segmentation_id += 1

        # else if bounding box is bigger than window size, create overlap annotation
        else:

            _binary_mask = deepcopy(binary_mask)
            # coordinate of grid 
            x = np.arange(minc, maxc, int(window_size * overlap))
            y = np.arange(minr, maxr, int(window_size * overlap))

            # remove the pixels of binary_mask on the grid edge
            for i in range(1, len(x), 2):
                _binary_mask[:, x[i]] = 0

            for i in range(1, len(y), 2):
                _binary_mask[y[i], :] = 0

            # get label of binary_mask 
            label_mask = label(_binary_mask)
            
            for label_idx in range(1, np.max(label_mask)+1):
                label_obj = label_mask == label_idx

                if np.sum(label_obj) > 50:

                    annotation_info = create_annotation_info(
                        segmentation_id, img_id, category_info, label_obj, (width, height), tolerance=2
                        )

                    if annotation_info is not None:
                        coco_output["annotations"].append(annotation_info)

                    segmentation_id += 1

            _binary_mask = deepcopy(binary_mask)

            for i in range(2, len(x), 2):
                _binary_mask[:, x[i]] = 0

            for i in range(2, len(y), 2):
                _binary_mask[y[i], :] = 0

            # get label of binary_mask 
            label_mask = label(_binary_mask)
            
            for label_idx in range(1, np.max(label_mask)+1):
                label_obj = label_mask == label_idx

                if np.sum(label_obj) > 50:

                    annotation_info = create_annotation_info(
                        segmentation_id, img_id, category_info, label_obj, (width, height), tolerance=2
                        )

                    if annotation_info is not None:
                        coco_output["annotations"].append(annotation_info)

                    segmentation_id += 1


    return coco_output, segmentation_id


def main():
    
    coco_output = {
    "info": INFO,
    "licenses": LICENSES,
    "categories": CATEGORIES,
    "images": [],
    "annotations": []
    }

    segmentation_id = 1


    # source dataset style is Cityscapes
    img_dir = os.path.join(SOURCE_DIR, 'leftImg8bit', SUBSET)
    img_list = glob(os.path.join(img_dir, '*.png'))

    # target dataset style is COCO
    target_dir = TARGET_DIR
    target_img_dir = os.path.join(target_dir, f'{SUBSET}2018')
    target_json_path = os.path.join(target_dir, 'annotations', f'instances_{SUBSET}2018.json')

    os.makedirs(target_img_dir, exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'annotations'), exist_ok=True)

    new_img_id = 1

    # add tqdm and description
    for img_id, img_path in tqdm(enumerate(img_list), desc="Creating COCO dataset"):

        # read images 
        img = cv2.imread(img_path)
        
        # copy image to target dir
        target_img_path = os.path.join(target_img_dir, os.path.basename(img_path))

        # read gt
        gt_path = img_path.replace("_leftImg8bit.png", "_gtFine_labelIds.png")
        gt_path = gt_path.replace("leftImg8bit", "gtFine")
        
        gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)

        for class_idx, class_name in enumerate(CLASSES):

            class_idx += 1
            convert_style = CONVERT_STYLE[class_name]

            if convert_style == 'normal':
                coco_output, segmentation_id = create_normal_annotation(coco_output, gt, new_img_id, img.shape[1], img.shape[0], class_idx, segmentation_id)                

            elif convert_style == 'overlap':
                coco_output, segmentation_id = create_overlap_annotation(coco_output, gt, new_img_id, img.shape[1], img.shape[0], class_idx, segmentation_id, window_size=WINDOW_SIZE, overlap=OVERLAP)

            if np.sum(gt == class_idx) > 0:
                shutil.copy(img_path, target_img_path)
                # create image info
                
                image_info = create_image_info(
                    image_id=new_img_id, 
                    file_name=os.path.basename(img_path), 
                    image_size=(img.shape[1], img.shape[0])
                )
                coco_output["images"].append(image_info)
                new_img_id += 1
            


    with open(target_json_path, 'w') as output_json_file:
        json.dump(coco_output, output_json_file)
        



if __name__ == "__main__":
    main()
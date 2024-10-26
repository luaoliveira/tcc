import json
import os
from pathlib import Path
import numpy as np
from PIL import Image
import cv2

#####################################################################
# Code to generate the masks based on the json file exported from COCO annotator
#####################################################################

img_height= 400
img_width=400

folder = Path('./coco-annotator/datasets/drone_imagens/.exports/')
json_file = next(folder.glob('*.json'), None)
# mask_dir = 'masks/'

mask_dir = Path("./masks")
mask_dir.mkdir(parents=True, exist_ok=True)

def create_mask(annotations, height, width):

    mask = np.zeros((height, width), dtype=np.uint8)

    for annotation in annotations:
        segmentation = annotation['segmentation']
        for polygon in segmentation:
            polygon = np.array(polygon).reshape((-1,2)).astype(np.int32)
            cv2.fillPoly(mask, [polygon], color=1)
            
    return mask

if json_file:
    with open(json_file) as f:
        coco_data = json.load(f)

for image in coco_data['images']:
    image_id = image['id']
    file_name = image['file_name']
    annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_id]

    mask = create_mask(annotations, img_height, img_width)

    mask_image=Image.fromarray(mask*255)

    save_path = mask_dir / file_name

    mask_image.save(save_path)
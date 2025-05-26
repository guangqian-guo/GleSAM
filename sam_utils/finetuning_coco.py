from genericpath import exists
import numpy as np
import pycocotools.mask as mask_util
import cv2
from pycocotools.coco import COCO
import os
import random
from pycocotools import mask as mask_utils
from tqdm import tqdm

def coco_polygons_to_mask(segmentation, image_size):
    mask = np.zeros(image_size, dtype=np.uint8)
    rles = mask_utils.frPyObjects(segmentation, image_size[1], image_size[0])  # 注意此处宽度和高度的顺序
    rle = mask_utils.merge(rles)
    mask = mask_utils.decode(rle)
    # 在每个多边形区域上进行填充
    for polygon in segmentation:
        rr, cc = mask_utils.frPyObjects([polygon], mask.shape[0], mask.shape[1])
        mask[rr, cc] = 1
    return mask


def save_mask_image(mask, output_path):
    cv2.imwrite(output_path, mask)


ann_path = '/home/ps/Guo/Project/sam-hq-main/eval_coco_ours/data/COCO/annotations/instances_val2017.json'
out_dir = '/home/ps/Guo/Project/sam-hq-main/eval_coco_ours/data/COCO/mask_val/'
os.makedirs(out_dir, exist_ok=True)


coco = COCO(ann_path)
empty_img = 0
for img_id in tqdm(coco.getImgIds()):
    img_info = coco.loadImgs(img_id)[0]
    img_name = img_info['file_name']
    # print(img_name)
    # img_path = os.path.join(img_dir, img_name)
    # image = cv2.imread(img_path)
    ann_ids  = coco.getAnnIds(img_id)
    anns = coco.loadAnns(ann_ids)
    if len(anns) > 0:
        selected_ann = random.choice(anns)
        seg = selected_ann['segmentation']
        mask = coco.annToMask(selected_ann)
    else:
        mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
        print(img_name)
        empty_img += 1
    
    # size = (img_info['height'], img_info['width'])


    # Example usage
    # binary_mask = coco_polygons_to_mask(seg, size)
    # print(binary_mask.max())
    
    
    # # Example usage
    output_path = os.path.join(out_dir, img_name)
    
    save_mask_image(mask*255, output_path)


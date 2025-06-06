from typing import List, Dict
import random
import math

import numpy as np
from PIL import Image
import cv2
from glob import glob
import os

def load_img_gt_list(datasets, is_train=True):
    img_list = []
    gt_list = []
    for i in range(len(datasets)):
        print("--->>>", " dataset ", i,"/",len(datasets)," ",datasets[i]["name"],"<<<---")
        tmp_im_list = glob(datasets[i]["im_dir"]+os.sep+'*'+datasets[i]["im_ext"])
        if is_train:
            img_list.extend(tmp_im_list)
        else:
            img_list.append(tmp_im_list)
        # print(datasets[i]["im_dir"]+os.sep+'*'+datasets[i]["im_ext"])
        print('-im-',datasets[i]["name"],datasets[i]["im_dir"], ': ',len(tmp_im_list))
        tmp_gt_list = [datasets[i]["gt_dir"]+os.sep+x.split(os.sep)[-1].split(datasets[i]["im_ext"])[0]+datasets[i]["gt_ext"] for x in tmp_im_list]
        if is_train:
            gt_list.extend(tmp_gt_list)
        else:
            gt_list.append(tmp_gt_list)

        print('-gt-', datasets[i]["name"],datasets[i]["gt_dir"], ': ',len(tmp_gt_list))

    return img_list, gt_list


def load_file_list(file_list_path: str) -> List[Dict[str, str]]:
    files = []
    with open(file_list_path, "r") as fin:
        for line in fin:
            path = line.strip()
            if path:
                files.append({"image_path": path, "prompt": ""})
    return files


# https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/image_datasets.py
def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


# https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/image_datasets.py
def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]

import os
import shutil
import cv2
import numpy as np
from tqdm import tqdm 
# source_folder = "/home/ps/Guo/Project/sam-hq-main/train/data/DIS5K/DIS-TR-L/lowlight_im_1536"
# des_folder = "/home/ps/Guo/Project/sam-hq-main/train/data/DIS5K/DIS-TR-L/lowlight_im"

# source_folder = '/home/ps/Guo/Project/sam-hq-main/train/data/DIS5K/DIS-VD-1536/day2night_hr/test_latest/images'
# des_folder = '/home/ps/Guo/Project/sam-hq-main/train/data/DIS5K/DIS-VD-1536/lowlight_im'

source_folder = '/home/ps/Guo/Project/sam-hq-main/train/data/cascade_psp/DUTS-TE-low/images'
des_folder = '/home/ps/Guo/Project/sam-hq-main/train/data/cascade_psp/DUTS-TE-low/DUTS-TE-low'
gt_dir = '/home/ps/Guo/Project/sam-hq-main/train/data/cascade_psp/DUTS-TE-split/gt'
b_map_save_dir = '/home/ps/Guo/Project/sam-hq-main/train/data/cascade_psp/DUTS-TE-low/b_map'

source_folder = '/home/ps/Guo/Project/sam-hq-main/train/data/cascade_psp/DUTS-TR-low/images'
des_folder = '/home/ps/Guo/Project/sam-hq-main/train/data/cascade_psp/DUTS-TR-low/DUTS-TR-low'
gt_dir = '/home/ps/Guo/Project/sam-hq-main/train/data/cascade_psp/DUTS-TR-split/gt'
b_map_save_dir = '/home/ps/Guo/Project/sam-hq-main/train/data/cascade_psp/DUTS-TR-low/b_map'


source_folder = '/home/ps/Guo/Project/sam-hq-main/train/data/cascade_psp/ecssd-low/images'
des_folder = '/home/ps/Guo/Project/sam-hq-main/train/data/cascade_psp/ecssd-low/ecssd-low'
gt_dir = '/home/ps/Guo/Project/sam-hq-main/train/data/cascade_psp/ecssd-split/gt'
b_map_save_dir = '/home/ps/Guo/Project/sam-hq-main/train/data/cascade_psp/ecssd-low/b_map'

source_folder = '/home/ps/Guo/Project/sam-hq-main/train/data/cascade_psp/fss_all-low/images'
des_folder = '/home/ps/Guo/Project/sam-hq-main/train/data/cascade_psp/fss_all-low/fss_all-low'
gt_dir = '/home/ps/Guo/Project/sam-hq-main/train/data/cascade_psp/fss_all-split/gt'
b_map_save_dir = '/home/ps/Guo/Project/sam-hq-main/train/data/cascade_psp/fss_all-low/b_map'

source_folder = '/home/ps/Guo/Project/sam-hq-main/train/data/cascade_psp/MSRA_10K-low/images'
des_folder = '/home/ps/Guo/Project/sam-hq-main/train/data/cascade_psp/MSRA_10K-low/MSRA_10K-low'
gt_dir = '/home/ps/Guo/Project/sam-hq-main/train/data/cascade_psp/MSRA_10K-split/gt'
b_map_save_dir = '/home/ps/Guo/Project/sam-hq-main/train/data/cascade_psp/MSRA_10K-low/b_map'

source_folder = '/home/ps/Guo/Project/sam-hq-main/train/data/thin_object_detection/ThinObject5K_low/images'
des_folder = '/home/ps/Guo/Project/sam-hq-main/train/data/thin_object_detection/ThinObject5K_low/images_train/'
gt_dir = '/home/ps/Guo/Project/sam-hq-main/train/data/thin_object_detection/ThinObject5K/masks_train'
b_map_save_dir = '/home/ps/Guo/Project/sam-hq-main/train/data/thin_object_detection/ThinObject5K_low/b_map'

source_folder = '/home/ps/Guo/Project/sam-hq-main/train/data/thin_object_detection/ThinObject5K_low/day2night_hr/test_latest/images'
des_folder = '/home/ps/Guo/Project/sam-hq-main/train/data/thin_object_detection/ThinObject5K_low/images_test/'
gt_dir = '/home/ps/Guo/Project/sam-hq-main/train/data/thin_object_detection/ThinObject5K/masks_test'
b_map_save_dir = '/home/ps/Guo/Project/sam-hq-main/train/data/thin_object_detection/ThinObject5K_low/b_map'

source_folder = '/home/ps/Guo/Project/P2BNet-main/TOV_mmdetection/data/COCO/val2017-low/day2night_hr/test_latest/images'
des_folder = '/home/ps/Guo/Project/P2BNet-main/TOV_mmdetection/data/COCO/val2017-low/'
gt_dir = '/home/ps/Guo/Project/sam-hq-main/train/data/thin_object_detection/ThinObject5K/masks_test'
b_map_save_dir = '/home/ps/Guo/Project/sam-hq-main/train/data/thin_object_detection/ThinObject5K_low/b_map'

# save_dir = '/home/ps/Guo/Project/sam-hq-main/train/data/DIS5K/DIS-VD-1536/'

os.makedirs(des_folder, exist_ok=True)
os.makedirs(b_map_save_dir, exist_ok=True)


file_names = os.listdir(source_folder)

for file_name in tqdm(file_names):
    if "fake" in file_name:
        source_path = os.path.join(source_folder, file_name)
        new_name = file_name.replace("_fake", "")
        gt_path = os.path.join(gt_dir, new_name)
        try:
            img = cv2.imread(source_path)
            h, w, c = img.shape
        except:
            print(file_name)
            continue
        
        # # resize gt--------------------------------------------------
        # gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        # gt_resize = cv2.resize(gt, (w, h))
        
        # # save gt
        # save_path = os.path.join(des_folder, new_name)
        # cv2.imwrite(save_path, gt_resize)
        
        # # generate b map--------------------------------------
        # contours, hierarchy = cv2.findContours(gt_resize, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # # 创建一个新的图像，用于绘制边界
        # boundary_image = np.zeros_like(gt_resize, dtype=np.uint8)
        # cv2.drawContours(boundary_image, contours, -1, 255, 1)
        # write boundary 
        # b_save_path = os.path.join(b_map_save_dir, new_name)
        # cv2.imwrite(b_save_path, boundary_image)
        
        # remove img -------------------------------------------------------
        des_path = os.path.join(des_folder, new_name)
        cv2.imwrite(des_path, img)
    
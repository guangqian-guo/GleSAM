import os
import cv2
from tqdm import tqdm

img_dir = '/home/ps/Guo/Project/sam-hq-main/train/data/DIS5K/DIS-TR-1536/lowlight_im_1536'

gt_dir = '/home/ps/Guo/Project/sam-hq-main/train/data/DIS5K/DIS-TR/gt'
save_dir = '/home/ps/Guo/Project/sam-hq-main/train/data/DIS5K/DIS-TR-1536/'

img_dir = '/home/ps/Guo/Project/sam-hq-main/train/data/DIS5K/DIS-VD-1536/lowlight_im'

gt_dir = '/home/ps/Guo/Project/sam-hq-main/train/data/DIS5K/DIS-VD/gt'
save_dir = '/home/ps/Guo/Project/sam-hq-main/train/data/DIS5K/DIS-VD-1536/'


gt_save_dir = os.path.join(save_dir, 'gt')
os.makedirs(gt_save_dir, exist_ok=True)


img_list = os.listdir(img_dir)

for img_name in tqdm(img_list):
    img_path = os.path.join(img_dir, img_name)
    gt_path = os.path.join(gt_dir, img_name)
    img = cv2.imread(img_path)
    h, w, c = img.shape
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    gt_resize = cv2.resize(gt, (w, h))
    save_path = os.path.join(gt_save_dir, img_name)
    
    cv2.imwrite(save_path, gt_resize)
    
    # generate boundary
    
    
    
    
    # cv2.imshow("gt", gt)
    # # cv2.imshow("gt_resize", gt_resize)
    # cv2.waitKey(10000)
    
    # exit()

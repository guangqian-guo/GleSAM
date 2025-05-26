import cv2
import numpy as np
import os
from tqdm import tqdm
# 读取二值分割的 mask 图
mask_dir = "/home/ps/Guo/Project/sam-hq-main/train/data/thin_object_detection/ThinObject5K/masks_test"
save_dir = "/home/ps/Guo/Project/sam-hq-main/train/data/thin_object_detection/ThinObject5K/edges_test"

# mask_dir = ''
# save_dir = ''


os.makedirs(save_dir, exist_ok=True)
mask_list = os.listdir(mask_dir)

for mask_name in tqdm(mask_list):
    if mask_name.split('.')[-1] == 'png':
        mask_path = os.path.join(mask_dir, mask_name)
        save_path = os.path.join(save_dir, mask_name)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # 寻找前景区域的轮廓
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 创建一个新的图像，用于绘制边界
        boundary_image = np.zeros_like(mask, dtype=np.uint8)
        # for contour in contours:
        #     epsilon = 0.02 * cv2.arcLength(contour, True)
        #     approx = cv2.approxPolyDP(contour, epsilon, True)
        #     cv2.drawContours(boundary_image, [approx], -1, 255, 1)
        # 在新图像上绘制边界
       
        cv2.drawContours(boundary_image, contours, -1, 255, 5)
        cv2.imwrite(save_path, boundary_image)
        # # 显示原始 mask 和生成的边界图像
        # cv2.imshow("Original Mask", mask)
        # cv2.imshow("Boundary Image", boundary_image)
        # cv2.waitKey(100)

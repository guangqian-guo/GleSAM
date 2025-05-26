import os
import shutil


source_folder = '/home/ps/Guo/Project/sam-hq-main/train/data/cascade_psp/MSRA_10K'
des_folder = '/home/ps/Guo/Project/sam-hq-main/train/data/cascade_psp/MSRA_10K-split'

img_folder = os.path.join(des_folder, 'img')
gt_folder = os.path.join(des_folder, 'gt')


os.makedirs(img_folder, exist_ok=True)
os.makedirs(gt_folder, exist_ok=True)


file_names = os.listdir(source_folder)

for file_name in file_names:
    source_path = os.path.join(source_folder, file_name)
    if "jpg" in file_name:
        des_path = os.path.join(img_folder, file_name)
        shutil.copy2(source_path, des_path)
    elif 'png' in file_name:
        des_path = os.path.join(gt_folder, file_name)
        shutil.copy2(source_path, des_path)
        
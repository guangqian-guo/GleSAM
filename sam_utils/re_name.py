import os

source_folder = "/home/ps/Guo/Project/sam-hq-main/train/data/DIS5K/DIS-VD-1536/im"

file_names = os.listdir(source_folder)

for file_name in file_names:
    old_path = os.path.join(source_folder, file_name)
    
    new_name = file_name.replace("_real", "")
    
    new_path = os.path.join(source_folder, new_name)
    
    os.rename(old_path, new_path)
    
    
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


# 读取图像
img_dir = '/home/ps/Guo/Project/sam-hq-main/train/data/COD/TrainDataset/Imgs'
save_dir = '/home/ps/Guo/Project/sam-hq-main/train/data/COD/TrainDataset/high-freq-imgs-enhance'
os.makedirs(save_dir, exist_ok=True)
# image_path = "/home/ps/Guo/Project/sam-hq-main/train/data/COD/TrainDataset/Imgs/COD10K-CAM-3-Flying-61-Katydid-4137.jpg"
# image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
img_list = os.listdir(img_dir)
for img_name in img_list:
    img_path = os.path.join(img_dir, img_name)
    image = cv2.imread(img_path)

    # 进行离散余弦变换（DCT）并分别对每个通道进行处理
    dct_result = np.zeros_like(image, dtype=np.float32)
    for i in range(3):  # 0: Blue, 1: Green, 2: Red
        dct_result[:, :, i] = cv2.dct(np.float32(image[:, :, i]))

    # 挑选要保留的频率成分
    keep_percentage = 0.1  # 保留前10%的频率成分
    rows, cols, _ = image.shape
    dct_result[int(rows * keep_percentage):, :, :] = 0
    dct_result[:, int(cols * keep_percentage):, :] = 0

    # 进行逆离散余弦变换（IDCT）
    filtered_image = np.zeros_like(image, dtype=np.uint8)
    for i in range(3):
        filtered_image[:, :, i] = cv2.idct(dct_result[:, :, i])

    enhanced_img = cv2.addWeighted(image, 0.5, filtered_image, 0.5, 0)  # fusion
    cv2.imwrite(os.path.join(save_dir, img_name), enhanced_img)
    
    # 可视化原始图像和过滤后的图像
    # plt.figure(figsize=(10, 5))

    # plt.subplot(1, 2, 1)
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt.title("Original Image")
    # plt.axis("off")

    # plt.subplot(1, 2, 2)
    # plt.imshow(cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB))
    # plt.title("Filtered Image")
    # plt.axis("off")

    # plt.show()


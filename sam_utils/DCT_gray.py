import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


# 读取图像
img_dir = '/home/ps/Guo/Project/sam-hq-main/train/data/COD/TrainDataset/Imgs'
save_dir = '/home/ps/Guo/Project/sam-hq-main/train/data/COD/TrainDataset/mix-freq-imgs'
os.makedirs(save_dir, exist_ok=True)
# image_path = "/home/ps/Guo/Project/sam-hq-main/train/data/COD/TrainDataset/Imgs/COD10K-CAM-3-Flying-61-Katydid-4137.jpg"
# image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
img_list = os.listdir(img_dir)
for img_name in img_list:
    img_path = os.path.join(img_dir, img_name)
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # 进行离散余弦变换
    dct_result = cv2.dct(np.float32(image))

    # 挑选要保留的频率成分
    keep_percentage = 0.1  # 保留前10%的频率成分
    rows, cols = dct_result.shape
    high = dct_result
    high[-1*int(rows * keep_percentage):, :] = 0
    high[:, -1*int(cols * keep_percentage):] = 0
    
    dct_result[int(rows * keep_percentage):, :] = 0
    dct_result[:, int(cols * keep_percentage):] = 0
    
    # 进行逆离散余弦变换
    filtered_image = cv2.idct(high+dct_result)

    cv2.imwrite(os.path.join(save_dir, img_name), filtered_image)
    

    # 可视化原始图像、频域图和过滤后的图像
    # plt.figure(figsize=(12, 4))

    # plt.subplot(1, 3, 1)
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt.title("Original Image")
    # plt.axis("off")

    # plt.subplot(1, 3, 2)
    # plt.imshow(np.log(np.abs(dct_result) + 1), cmap="gray")
    # plt.title("DCT Spectrum")
    # plt.axis("off")

    # plt.subplot(1, 3, 3)
    # plt.imshow(filtered_image, cmap="gray")
    # plt.title("Filtered Image")
    # plt.axis("off")

    # plt.show()

# # 进行离散余弦变换（DCT）并分别对每个通道进行处理
# dct_result = np.zeros_like(image, dtype=np.float32)
# for i in range(3):  # 0: Blue, 1: Green, 2: Red
#     dct_result[:, :, i] = cv2.dct(np.float32(image[:, :, i]))

# # 挑选要保留的频率成分
# keep_percentage = 0.5  # 保留前10%的频率成分
# rows, cols, _ = image.shape
# dct_result[int(rows * keep_percentage):, :, :] = 0
# dct_result[:, int(cols * keep_percentage):, :] = 0

# # 进行逆离散余弦变换（IDCT）
# filtered_image = np.zeros_like(image, dtype=np.uint8)
# for i in range(3):
#     filtered_image[:, :, i] = cv2.idct(dct_result[:, :, i])

# # 可视化原始图像和过滤后的图像
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


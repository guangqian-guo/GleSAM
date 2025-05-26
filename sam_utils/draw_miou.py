# import matplotlib.pyplot as plt

# # 准备数据
# x = ['1 point', '3 point ', '5 point', '10 point', 'point+box']
# y1 = [2, 3, 5, 7, 11]
# y2 = [1, 2, 4, 6, 10]

# # 创建折线图
# plt.figure(figsize=(8, 6))  # 设置图形大小
# plt.plot(x, y1, marker='o', color='blue', linewidth=2, label='SAM')  # 设置线条颜色、样式、宽度和标签
# plt.plot(x, y2, marker='s', color='red', linewidth=2, label='Ours')   # 设置线条颜色、样式、宽度和标签

# # 添加标题和标签
# plt.title('Line Plot Example', fontsize=16)  # 设置标题和字体大小
# plt.xlabel('prompt num', fontsize=14)            # 设置X轴标签和字体大小
# plt.ylabel('mIoU', fontsize=14)            # 设置Y轴标签和字体大小

# # 设置图例
# plt.legend(loc='upper left', fontsize=12)    # 设置图例位置和字体大小

# # 设置网格线
# plt.grid(True, linestyle='--', alpha=0.5)    # 显示网格线，设置线条样式和透明度

# # 调整坐标轴范围
# plt.xlim(0, 6)  # 设置X轴范围
# plt.ylim(0, 12) # 设置Y轴范围

# # 显示图形
# plt.show()

import numpy as np
import matplotlib.pyplot as plt

# 数据
metrics = ['SAM', 'HQ-SAM', 'HQ-SAM++','Ours']
metrics = ['SAM2 (Baseline)', 'ClearSeg2 (Ours)']

thresholds = ['1 point', '2 point' , '3 points', '5 points', '8 points', '10 points']
values = np.array([
    # [43.8, 62.1, 67.1, 69.5, 68.4, 69.2],
    # [43.3, 64.3, 71.4, 75.9, 76.7, 77.5],
    # [44.5, 64.0, 71.4, 76.0, 76.8, 76.9],
    # [49.2, 65.4, 71.2, 76.2, 77.5, 78.5],
    # # miou  of CSD2K
    # [0.390, 0.423, 0.427, 0.430, 0.421, 0.411],
    # [0.409, 0.498, 0.540,  0.584, 0.611, 0.622],
    
    # # miou of unseen-set
    # [0.504,  0.570, 0.586, 0.592, 0.587, 0.583],
    # [0.484, 0.617, 0.682, 0.734, 0.767, 0.777]

    # # mbiou of unseen-set
    # [0.405, 0.459, 0.473, 0.479, 0.473, 0.468],
    # [0.406, 0.509, 0.562, 0.608, 0.638, 0.647]
    
    # SAM-ecssd
    # [0.2921, 0.5802, 0.6156, 0.6284, 0.6235, 0.6249],
    # [0.3175, 0.7113, 0.7944, 0.8213, 0.8277, 0.8287]

    # SAM2-ecssd
    [0.4239, 0.6745, 0.7292, 0.7650, 0.7795, 0.7783],
    [0.4538, 0.7555, 0.7976, 0.8373, 0.8430, 0.8359]
])

# 绘图
plt.figure(figsize=(8, 6))
plt.ylim(0.4, 0.9)
marker = ['o', '^', 's', 'D']
# color = ['red', 'green', '','#FFFFE0', '#D8BFD8','#ADD8E6','#00008B']
for i, metric in enumerate(metrics):
    plt.plot(thresholds, values[i], marker=marker[i], markersize=8, linewidth=3,  label=metric)
    # for x, y in zip(thresholds, values[i]):
    #     plt.text(x, y, f'{y: .3f}', ha='center', va='bottom', fontsize=16)
    
plt.xlabel('Number of point prompts', fontsize=18)
plt.ylabel('IoU', fontsize=18)
# plt.title('Metrics vs Threshold')
plt.legend(fontsize=16)
plt.grid(False)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.savefig('mIoU-sam2.png')
plt.show()



# import matplotlib.pyplot as plt

# # 数据
# x_values = ['1', '2', '3', '5', '10']
# sam_values = [61, 83, 86, 88, 89]
# ours_values = [62, 81, 83, 86, 87]

# # 绘图
# plt.figure(figsize=(8, 6))

# plt.plot(x_values, sam_values, marker='o', color='lightblue', linestyle='-', linewidth=2, label='SAM')
# plt.plot(x_values, ours_values, marker='o', color='lightcoral', linestyle='-', linewidth=2, label='Ours')

# plt.xlabel('Prompt num.', fontsize=12)
# plt.ylabel('mIoU', fontsize=12)
# plt.title('mIoU vs Prompt num.', fontsize=14)
# plt.legend(fontsize=12)
# # plt.grid(True)
# plt.tight_layout()  # 调整布局，防止文字重叠
# plt.show()
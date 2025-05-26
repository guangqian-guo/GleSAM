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
metrics = ['SAM (Baseline)', 'ClearSeg', 'SAM2 (Baseline)','ClearSeg2']

thresholds = ['1', '3', '5', '7', '9']
values = np.array([
    [0.4763, 0.4763, 0.4763, 0.4763, 0.4763],
    [0.5751, 0.5821, 0.5851, 0.5786, 0.5733],
    [0.5625, 0.5625, 0.5625, 0.5625, 0.5625],
    [0.5906, 0.6150, 0.6165, 0.5949, 0.5950 ]
])

# 绘图
plt.figure(figsize=(6, 6))
plt.ylim(0.3, 0.7)
marker = ['o', '^', 's', 'D']
# color = ['red', 'green', ,'#FFFFE0', '#D8BFD8','#ADD8E6','#00008B']
for i, metric in enumerate(metrics):
    plt.plot(thresholds, values[i], marker=marker[i], markersize=8, linewidth=3,  label=metric)
    # for x, y in zip(thresholds, values[i]):
    #     plt.text(x, y, f'{y: .3f}', ha='center', va='bottom', fontsize=16)



plt.xlabel('k', fontsize=20)
plt.ylabel('IoU', fontsize=20)
plt.legend(fontsize=16, frameon=True)
plt.grid(True, linestyle='--', alpha=0.7)

plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.savefig('k_coco.png', dpi=300)
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
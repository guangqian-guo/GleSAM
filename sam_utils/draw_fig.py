# import matplotlib.pyplot as plt
# import numpy as np

# # 数据
# tau_values = [0.6, 0.7, 0.8, 0.9]
# mIoU_values = [60.00, 60.28, 60.20, 60.03]
# CL_05_values = [74.03, 74.47, 74.33, 74.07]
# CL_07_values = [41.05, 41.87, 41.96, 41.54]
# CL_09_values = [2.10, 2.13, 2.14, 2.11]

# # 设置每个柱状图的宽度
# bar_width = 0.2

# # 设置图表标题和标签
# plt.title('Performance Metrics')
# plt.xlabel('Threshold (τ)')
# plt.ylabel('Metric Values')

# # 设置 x 轴刻度
# tau_indices = np.arange(len(tau_values))

# # 绘制柱状图
# plt.bar(tau_indices - bar_width * 1.5, mIoU_values, width=bar_width, label='mIoU')
# plt.bar(tau_indices - bar_width * 0.5, CL_05_values, width=bar_width, label='CL@0.5')
# plt.bar(tau_indices + bar_width * 0.5, CL_07_values, width=bar_width, label='CL@0.7')
# plt.bar(tau_indices + bar_width * 1.5, CL_09_values, width=bar_width, label='CL@0.9')

# # 设置 x 轴刻度标签
# plt.xticks(tau_indices, tau_values)

# # 添加图例
# plt.legend()

# # 显示图表
# plt.show()

# import matplotlib.pyplot as plt
# import numpy as np

# # 数据
# thresholds = [0.6, 0.7, 0.8, 0.9]
# mIoU_values = [60.00, 60.28, 60.20, 60.03]
# CL_05_values = [74.03, 74.47, 74.33, 74.07]
# CL_07_values = [41.05, 41.87, 41.96, 41.54]
# CL_09_values = [2.10, 2.13, 2.14, 2.11]

# # 设置每个柱状图的宽度
# bar_width = 0.2

# # 设置图表标题和标签
# plt.title('Performance Metrics at Different Thresholds')
# plt.xlabel('Metrics')
# plt.ylabel('Metric Values')

# # 设置 x 轴刻度
# metric_indices = np.arange(4)

# # 绘制柱状图
# plt.bar(metric_indices - bar_width * 1.5, [mIoU_values[0], CL_05_values[0], CL_07_values[0], CL_09_values[0]], width=bar_width, label=f'Threshold={thresholds[0]}')
# plt.bar(metric_indices - bar_width * 0.5, [mIoU_values[1], CL_05_values[1], CL_07_values[1], CL_09_values[1]], width=bar_width, label=f'Threshold={thresholds[1]}')
# plt.bar(metric_indices + bar_width * 0.5, [mIoU_values[2], CL_05_values[2], CL_07_values[2], CL_09_values[2]], width=bar_width, label=f'Threshold={thresholds[2]}')
# plt.bar(metric_indices + bar_width * 1.5, [mIoU_values[3], CL_05_values[3], CL_07_values[3], CL_09_values[3]], width=bar_width, label=f'Threshold={thresholds[3]}')

# # 设置 x 轴刻度标签
# plt.xticks(metric_indices, ['mIoU', 'CL@0.5', 'CL@0.7', 'CL@0.9'])

# # 添加图例
# plt.legend()

# # 显示图表
# plt.show()


import matplotlib.pyplot as plt
import numpy as np

# 数据
# thresholds = [0.6, 0.7, 0.8, 0.9]
# mIoU_values = [60.00, 60.28, 60.20, 60.03]
# CL_05_values = [74.03, 74.47, 74.33, 74.07]
# CL_07_values = [41.05, 41.87, 41.96, 41.54]
# CL_09_values = [2.10, 2.13, 2.14, 2.11]

# # 设置图表标题和标签
# plt.title('Performance Metrics at Different Thresholds')
# plt.xlabel('Metrics')
# plt.ylabel('Metric Values')

# # 设置 x 轴刻度
# metric_indices = np.arange(4)

# # 绘制折线图
# plt.plot(metric_indices, mIoU_values, marker='o', label='mIoU')
# plt.plot(metric_indices, CL_05_values, marker='o', label='CL@0.5')
# plt.plot(metric_indices, CL_07_values, marker='o', label='CL@0.7')
# plt.plot(metric_indices, CL_09_values, marker='o', label='CL@0.9')

# # 设置 x 轴刻度标签
# plt.xticks(metric_indices, ['Threshold=0.6', 'Threshold=0.7', 'Threshold=0.8', 'Threshold=0.9'])

# # 添加图例
# plt.legend()

# # 显示图表
# plt.show()


import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# 数据
stress = np.array([0, 30, 60, 90, 120, 150, 180, 210])
Q345B = np.array([0, -0.12, -0.21, -0.275, -0.3, -0.27, -0.21, -0.15])
CrMnV = np.array([0, -0.01, -0.03, -0.05, -0.05, -0.04, -0.03, -0.02])
MnV = np.array([0, 0.002, -0.02, -0.04, -0.05, -0.06, -0.06, -0.07])

# 样条插值
x_interp = np.linspace(np.min(stress), np.max(stress), 1000)
Q345B_interp = CubicSpline(stress, Q345B, bc_type='natural')(x_interp)
CrMnV_interp = CubicSpline(stress, CrMnV, bc_type='natural')(x_interp)
MnV_interp = CubicSpline(stress, MnV, bc_type='natural')(x_interp)

# 绘图
plt.figure()

plt.plot(stress, Q345B, 'o', label='Q345B')
plt.plot(x_interp, Q345B_interp, label='Q345B (Interpolated)')

plt.plot(stress, CrMnV, 's', label='42CrMnV')
plt.plot(stress, MnV, 'd', label='48MnV')
plt.plot(x_interp, CrMnV_interp, label='42CrMnV (Interpolated)')
plt.plot(x_interp, MnV_interp, label='48MnV (Interpolated)')

plt.xlabel('Time (s)')
plt.ylabel('Stress (MPa)')
plt.title('Stress vs Time')
plt.legend()
plt.grid(False)  # 关闭背景网格线
plt.show()
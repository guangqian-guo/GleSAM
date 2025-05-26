import matplotlib.pyplot as plt

# 示例数据
model_size = [300, 350, 1000, 1700, 2400, 3100]
dice_coefficient = [0.32, 0.39, 0.34, 0.40, 0.38, 0.44]
fps = [6.77, 6.23, 2.90, 2.80, 1.78, 1.62]
size=[1,1,1,1,1,1]
labels = ['SAM(ViT-B)', 'VNS-SAM(ViT-B)', 'SAM(ViT-L)', 'VNS-SAM(ViT-L)', 'SAM(ViT-H)', 'VNS-SAM(ViT-H)']
colors = ['blue', 'red', 'blue', 'red', 'blue', 'red']
edge_color = ['b', 'r', 'b', 'r', 'b', 'r']
plt.figure(figsize=(9, 6))

# 绘制气泡图
for i in range(len(model_size)):
    plt.scatter(model_size[i], dice_coefficient[i], s=size[i]*2500, color=colors[i], alpha=0.6, edgecolors=edge_color[i], linewidth=2)
    plt.text(model_size[i], dice_coefficient[i]-0.007, f'{fps[i]} \n fps', color='w', fontsize=12, ha='center')
    plt.text(model_size[i], dice_coefficient[i]+0.023, labels[i], fontsize=12, ha='center')

plt.xlim(0, 3600 )
plt.ylim(0.25, 0.5)


plt.xlabel('Model Size [M]')
plt.ylabel('mean IoU')
# plt.title('Comparison of performance, speed, and model size among various SAM and RobustSAM variants')
plt.grid(True)
plt.savefig('./compare.jpg')
# plt.show()
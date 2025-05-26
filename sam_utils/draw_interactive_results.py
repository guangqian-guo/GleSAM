
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
# Data for the first plot
k = ['1 point', '2 point' , '3 points', '5 points', '8 points', '10 points']
SAM_baseline = [0.2921, 0.5802, 0.6156, 0.6284, 0.6235, 0.6249]
ClearSeg = [0.3175, 0.7113, 0.7944, 0.8213, 0.8277, 0.8287]
SAM2_baseline = [0.4239, 0.6745, 0.7292, 0.7650, 0.7795, 0.7783]
ClearSeg2 = [0.4538, 0.7555, 0.7976, 0.8373, 0.8430, 0.8359]


# SAM_baseline = [0.6051, 0.6051,0.6051, 0.6051, 0.6051]
# ClearSeg = [0.7467, 0.7809, 0.7844, 0.7518, 0.7393]
# SAM2_baseline = [0.7150, 0.7150, 0.7150, 0.7150, 0.7150]
# ClearSeg2 = [0.7824, 0.7890, 0.7976, 0.7759, 0.7729]
    
# Plot configuration to enhance aesthetics similar to the second plot
plt.figure(figsize=(10, 6))

# Line styles and markers
# plt.plot(k, SAM_baseline, label='SAM (Baseline)', marker='o', markersize=8, linestyle='--', color='#6C7A89', linewidth=2.5)  #7851A9  
# plt.plot(k, ClearSeg, label='GenSAM (Ours)', markersize=8, marker='^',linestyle='--', color='#2E8B57', linewidth=2.5)  #A52A2A
plt.plot(k, SAM2_baseline, label='SAM2 (Baseline)', marker='s', markersize=8, color='#6C7A89', linestyle='--',linewidth=2.5)
plt.plot(k, ClearSeg2, label='GenSAM2 (Ours)', markersize=8, marker='D',color='#2E8B57', linestyle='--', linewidth=2.5)
plt.xticks(np.array(k))

# Title and labels with enhanced formatting
# plt.title('Comparison of Different Segmentation Methods', fontsize=14, weight='bold')
plt.xlabel('Number of point prompts', fontsize=16)
plt.ylabel('IoU', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
# Legend with improved styling
plt.legend(fontsize=12, frameon=True, loc='lower right',  fancybox=True, shadow=True, borderpad=1)  #loc='lower right', 
# Grid with dashed style for better readability
plt.grid(visible=True, linestyle='--', alpha=0.7)

# Set axis limits for better visualization
plt.ylim(0.4, 0.9)
# plt.ylim(0.45, 0.85)
# Show the plot
plt.tight_layout()
plt.savefig('interactive_sam2.png', dpi=300, bbox_inches='tight', pad_inches=0.1)


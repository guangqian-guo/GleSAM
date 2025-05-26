
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
# Data for the first plot
k = [1, 3, 5, 7, 9]

# COCO
SAM_baseline = [0.4763, 0.4763, 0.4763, 0.4763, 0.4763]
ClearSeg = [0.5751, 0.5821, 0.5851, 0.5786, 0.5733]
SAM2_baseline = [0.5625, 0.5625, 0.5625, 0.5625, 0.5625]
ClearSeg2 = [0.5906, 0.6150, 0.6165, 0.5949, 0.5950 ]

# ECSSD
SAM_baseline = [0.6051, 0.6051,0.6051, 0.6051, 0.6051]
ClearSeg = [0.7467, 0.7809, 0.7844, 0.7518, 0.7393]
SAM2_baseline = [0.7150, 0.7150, 0.7150, 0.7150, 0.7150]
ClearSeg2 = [0.7824, 0.7890, 0.7976, 0.7759, 0.7729]
    
# Plot configuration to enhance aesthetics similar to the second plot
plt.figure(figsize=(6, 6))

# Line styles and markers
plt.plot(k, SAM_baseline, label='SAM (Baseline)', marker='o', markersize=8, linestyle='--', color='#6C7A89', linewidth=2.5)
plt.plot(k, ClearSeg, label='GleSAM (Ours)', markersize=8, marker='^',linestyle='--', color='#2E8B57', linewidth=2.5)
plt.plot(k, SAM2_baseline, label='SAM2 (Baseline)', marker='s', markersize=8, color='#7851A9', linestyle='--',linewidth=2.5)
plt.plot(k, ClearSeg2, label='GleSAM2 (Ours)', markersize=8, marker='D',color='#A52A2A', linestyle='--', linewidth=2.5)
plt.xticks(np.array(k))

# Title and labels with enhanced formatting
# plt.title('Comparison of Different Segmentation Methods', fontsize=14, weight='bold')
plt.xlabel('k', fontsize=16)
plt.ylabel('IoU', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
# Legend with improved styling
plt.legend(fontsize=12, frameon=True, loc='lower right',  fancybox=True, shadow=True, borderpad=1)  #loc='lower right', 
# Grid with dashed style for better readability
plt.grid(visible=True, linestyle='--', alpha=0.7)

# Set axis limits for better visualization
# plt.ylim(0.3, 0.7)
plt.ylim(0.45, 0.85)
# Show the plot
plt.tight_layout()
plt.savefig('ecssd_improved.png', dpi=300, bbox_inches='tight', pad_inches=0.1)


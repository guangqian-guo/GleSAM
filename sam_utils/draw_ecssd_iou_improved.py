import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
# Data for the first plot
k = [1, 3, 5, 7, 9]
SAM_baseline = [0.6051, 0.6051,0.6051, 0.6051, 0.6051]
ClearSeg = [0.7467, 0.7809, 0.7844, 0.7518, 0.7393]
SAM2_baseline = [0.7150, 0.7150, 0.7150, 0.7150, 0.7150]
ClearSeg2 = [0.7824, 0.7890, 0.7976, 0.7759, 0.7729]
    
# Plot configuration to enhance aesthetics similar to the second plot
plt.figure(figsize=(6, 6))

# Line styles and markers
plt.plot(k, SAM_baseline, label='SAM (Baseline)', marker='o', markersize=8, linestyle='--', color='#6C7A89', linewidth=1.5)
plt.plot(k, ClearSeg, label='GenSAM', markersize=8, marker='^',linestyle='--', color='#2E8B57', linewidth=1.5)
plt.plot(k, SAM2_baseline, label='SAM2 (Baseline)', marker='s', markersize=8, color='#7851A9', linestyle='--',linewidth=1.5)
plt.plot(k, ClearSeg2, label='GenSAM2', markersize=8, marker='D',color='#A52A2A', linestyle='--', linewidth=1.5)
plt.xticks(np.array(k))

# Title and labels with enhanced formatting
# plt.title('Comparison of Different Segmentation Methods', fontsize=14, weight='bold')
plt.xlabel('k', fontsize=12)
plt.ylabel('IoU', fontsize=12)

# Legend with improved styling
plt.legend(fontsize=10, frameon=True, fancybox=True, shadow=True, borderpad=1)  #loc='lower right', 
# Grid with dashed style for better readability
plt.grid(visible=True, linestyle='--', alpha=0.7)

# Set axis limits for better visualization
plt.ylim(0.55, 0.85)

# Show the plot
plt.tight_layout()
plt.savefig('plot.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
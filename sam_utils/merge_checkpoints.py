'''
25.1.25
合并decoder和encoder的权重。
'''

import os

import torch

decoder = torch.load('/home/ps/Guo/Project/GleSAM-code/work-dir/GleSAM2/ft-decoder/gle-decoder.pth')

sam = torch.load('/home/ps/Guo/Project/GleSAM-code/work-dir/sam2_hiera_base_plus.pt')

glesam = sam
# print(glesam['model'].keys())

for name, param in decoder.items():
    print(name)
    name = name.replace('module.', '')
    assert name in glesam['model'].keys()
    glesam['model'][name] = param


torch.save(glesam, '/home/ps/Guo/Project/GleSAM-code/work-dir/GleSAM2/ft-decoder/glesam2-base.pth')

    # glesam[name] = param
    
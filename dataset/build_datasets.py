"""
build lr dataset
by guogq 
2024/10/8
"""


# train set
dataset_MSRA_train = {
    "name": "MSRA",
    "im_dir": "datas/MSRA_10K",
    "gt_dir": "datas/MSRA_10K",
    "im_ext": ".jpg",
    "gt_ext": ".png"
}

dataset_Thin_train = {
    "name": "Thin",
    "im_dir": "datas/train/ThinObject5K/images_train",
    "gt_dir": "datas/train/ThinObject5K/masks_train",
    "im_ext": ".jpg",
    "gt_ext": ".png"
}

dataset_LVIS_train = {
    "name": "LVIS",
    "im_dir": "datas/RobustSeg/train/LVIS/LVIS_train/images",
    "gt_dir": "datas/RobustSeg/train/LVIS/LVIS_train/masks",
    "im_ext": ".jpg",
    "gt_ext": ".png"
}


# lr val sets
# seen sets
dataset_Thin_lr4_val = {
    "name": "thin_val",
    "im_dir": "datas/thin_object_detection/ThinObject5K-RealLR-ds4-orisize/lr", 
    # "im_dir": "datas/thin_object_detection/ThinObject5K-RealLR-ds-random/lr", 
    "gt_dir": "datas/thin_object_detection/ThinObject5K/masks_test",
    "im_ext": ".jpg",
    "gt_ext": ".png"
}


dataset_ECSSD_lr4_val = {
    "name": "ECSSD_val",
    "im_dir": "/home/ps/Guo/Project/GleSAM-code/datas/test/ecssd-RealLR-ds4-orisize/lr",
    # "im_dir": "datas/cascade_psp/ecssd-RealLR-ds-random-orisize/lr",
    "gt_dir": "/home/ps/Guo/Project/sam-hq-main/train/data/cascade_psp/ecssd",
    "im_ext": ".jpg",
    "gt_ext": ".png"
}


dataset_coco_lr4_val = {
    "name": "coco_lr_val",
    "im_dir": "datas/RobustSeg/val/coco/images/RealLR-ds4-orisize/lr",
    # "im_dir": "datas/RobustSeg/val/coco/images/RealLR-ds-random-orisize/lr",
    "gt_dir": "datas/RobustSeg/val/coco/masks_random",
    "im_ext": ".jpg",
    "gt_ext": ".png"
}


dataset_lvis_lr4_val = {
    "name": "lvis_lr_val",
    "im_dir": "datas/RobustSeg/val/LVIS_val/RealLR-ds4-orisize/lr",
    # "im_dir": "datas/RobustSeg/val/LVIS_val/RealLR-ds-random-orisize/lr",
    "gt_dir": "datas/RobustSeg/val/LVIS_val/masks_random",
    "im_ext": ".jpg",
    "gt_ext": ".png"
}


dataset_DUTS_lr_val = {
    "name": "DUTS",
    "im_dir": "datas/cascade_psp/DUTS-TE-RealLR-ds2-orisize/lr",
    "gt_dir": "datas/cascade_psp/DUTS-TE",
    "im_ext": ".jpg",
    "gt_ext": ".png"
}


dataset_Thin_lr2_val = {
    "name": "thin_val",
    "im_dir": "datas/test/Thin/lr", 
    # "im_dir": "datas/thin_object_detection/ThinObject5K-RealLR-ds-random/lr", 
    "gt_dir": "/home/ps/Guo/Project/sam-hq-main/train/data/thin_object_detection/ThinObject5K/masks_test",
    "im_ext": ".jpg",
    "gt_ext": ".png"
}


dataset_ECSSD_lr2_val = {
    "name": "ECSSD_val",
    "im_dir": "datas/test/ecssd-RealLR-ds2-orisize/lr",
    # "im_dir": "datas/cascade_psp/ecssd-RealLR-ds-random-orisize/lr",
    "gt_dir": "/home/ps/Guo/Project/sam-hq-main/train/data/cascade_psp/ecssd",
    "im_ext": ".jpg",
    "gt_ext": ".png"
}


dataset_coco_lr2_val = {
    "name": "coco_lr_val",
    "im_dir": "datas/RobustSeg/val/coco/images/RealLR-ds2-orisize/lr",
    # "im_dir": "datas/RobustSeg/val/coco/images/RealLR-ds-random-orisize/lr",
    "gt_dir": "datas/RobustSeg/val/coco/masks_random",
    "im_ext": ".jpg",
    "gt_ext": ".png"
}


dataset_lvis_lr2_val = {
    "name": "lvis_lr_val",
    "im_dir": "datas/test/LVIS/lr",
    "gt_dir": "/home/ps/Guo/dataset/LVIS/test/masks",
    "im_ext": ".jpg",
    "gt_ext": ".png"
}


dataset_Thin_lr1_val = {
    "name": "thin_val",
    "im_dir": "datas/thin_object_detection/ThinObject5K-RealLR-ds1-orisize/lr", 
    # "im_dir": "datas/thin_object_detection/ThinObject5K-RealLR-ds-random/lr", 
    "gt_dir": "datas/thin_object_detection/ThinObject5K/masks_test",
    "im_ext": ".jpg",
    "gt_ext": ".png"
}

dataset_ECSSD_lr1_val = {
    "name": "ECSSD_val",
    "im_dir": "datas/cascade_psp/ecssd-RealLR-ds1-orisize/lr",
    # "im_dir": "datas/cascade_psp/ecssd-RealLR-ds-random-orisize/lr",
    "gt_dir": "datas/cascade_psp/ecssd",
    "im_ext": ".jpg",
    "gt_ext": ".png"
}


dataset_coco_lr1_val = {
    "name": "coco_lr_val",
    "im_dir": "datas/RobustSeg/val/coco/images/RealLR-ds1-orisize/lr",
    # "im_dir": "datas/RobustSeg/val/coco/images/RealLR-ds-random-orisize/lr",
    "gt_dir": "datas/RobustSeg/val/coco/masks_random",
    "im_ext": ".jpg",
    "gt_ext": ".png"
}


dataset_lvis_lr1_val = {
    "name": "lvis_lr_val",
    "im_dir": "datas/RobustSeg/val/LVIS_val/RealLR-ds1-orisize/lr",
    # "im_dir": "datas/RobustSeg/val/LVIS_val/RealLR-ds-random-orisize/lr",
    "gt_dir": "datas/RobustSeg/val/LVIS_val/masks_random",
    "im_ext": ".jpg",
    "gt_ext": ".png"
}


dataset_demo_val = {
    "name": "demo_val",
    "im_dir": "datas/demo/images",
    # "im_dir": "datas/RobustSeg/val/LVIS_val/RealLR-ds-random-orisize/lr",
    "gt_dir": "datas/demo/masks",
    "im_ext": ".jpg",
    "gt_ext": ".png"
}


# robust sam degrade
dataset_Thin_robust_degrade_val = {
    "name": "thin_val",
    "im_dir": "datas/thin_object_detection/ThinObject5K-robust-degrade", 
    # "im_dir": "datas/thin_object_detection/ThinObject5K-RealLR-ds-random/lr", 
    "gt_dir": "datas/thin_object_detection/ThinObject5K/masks_test",
    "im_ext": ".jpg",
    "gt_ext": ".png"
}


dataset_lvis_robust_degrade_val = {
    "name": "lvis_lr_val",
    "im_dir": "datas/RobustSeg/val/LVIS_val/robust_degrade",
    # "im_dir": "datas/RobustSeg/val/LVIS_val/RealLR-ds-random-orisize/lr",
    "gt_dir": "datas/RobustSeg/val/LVIS_val/masks_random",
    "im_ext": ".jpg",
    "gt_ext": ".png"
}


dataset_ECSSD_robust_degrade_val = {
    "name": "ECSSD_val",
    "im_dir": "datas/cascade_psp/ecssd-robust-degrade",
    # "im_dir": "datas/cascade_psp/ecssd-RealLR-ds-random-orisize/lr",
    "gt_dir": "datas/cascade_psp/ecssd",
    "im_ext": ".jpg",
    "gt_ext": ".png"
}


dataset_coco_robust_degrade_val = {
    "name": "coco_lr_val",
    "im_dir": "datas/RobustSeg/val/coco/images/robust-degrade",
    # "im_dir": "datas/RobustSeg/val/coco/images/RealLR-ds-random-orisize/lr",
    "gt_dir": "datas/RobustSeg/val/coco/masks_random",
    "im_ext": ".jpg",
    "gt_ext": ".png"
}


dataset_DUTS_robust_degrade_val = {
    "name": "DUTS",
    "im_dir": "datas/cascade_psp/DUTS-TE-robust-degrade",
    "gt_dir": "datas/cascade_psp/DUTS-TE",
    "im_ext": ".jpg",
    "gt_ext": ".png"
}


# real world
dataset_bdd10K_val = {
    "name": "bdd10K_val",
    "im_dir": "datas/RobustSeg/val/BDD100K/bdd100k_val/images/10k/val",
    "gt_dir": "datas/RobustSeg/val/BDD100K/bdd100k_val/masks",
    "im_ext": ".jpg",
    "gt_ext": ".png"
}


dataset_LISU_val = {
    "name": "LISU",
    "im_dir": "datas/test/LISU-8-9-10-11/img",
    "gt_dir": "datas/test/LISU-8-9-10-11/gt",
    "im_ext": ".png",
    "gt_ext": ".png"
}


# hr val sets
dataset_Thin_val = {
    "name": "thin_val",
    "im_dir": "datas/thin_object_detection/ThinObject5K/images_test",
    "gt_dir": "datas/thin_object_detection/ThinObject5K/masks_test",
    "im_ext": ".jpg",
    "gt_ext": ".png"
}


dataset_ECSSD_val = {
    "name": "ECSSD_val",
    "im_dir": "/home/ps/Guo/dataset/ECSSD/Imgs",
    "gt_dir": "/home/ps/Guo/dataset/ECSSD/GT",
    "im_ext": ".jpg",
    "gt_ext": ".png"
}


dataset_coco_val = {
    "name": "coco_lr_val",
    # "im_dir": "datas/RobustSeg/val/coco/images/RealLR-ds4-orisize/lr",
    "im_dir": "datas/RobustSeg/val/coco/images/clear",
    "gt_dir": "datas/RobustSeg/val/coco/masks_random",
    "im_ext": ".jpg",
    "gt_ext": ".png"
}


dataset_lvis_val = {
    "name": "lvis_lr_val",
    # "im_dir": "datas/RobustSeg/val/LVIS_val/RealLR-ds4-orisize/lr",
    "im_dir": "datas/RobustSeg/val/LVIS_val/clear",
    "gt_dir": "datas/RobustSeg/val/LVIS_val/masks_random",
    "im_ext": ".jpg",
    "gt_ext": ".png"
}


# osediff hr val sets
dataset_ECSSD_lr_osediff_val = {
    "name": "ECSSD_val",
    "im_dir": "datas/cascade_psp/ecssd-RealLR-ds2-orisize-osediff",
    "gt_dir": "datas/cascade_psp/ecssd",
    "im_ext": ".jpg",
    "gt_ext": ".png"
}


dataset_Thin_lr_osediff_val = {
    "name": "thin_val",
    "im_dir": "datas/thin_object_detection/ThinObject5K-RealLR-ds2-orisize-osediff",
    "gt_dir": "datas/thin_object_detection/ThinObject5K/masks_test",
    "im_ext": ".jpg",
    "gt_ext": ".png"
}


dataset_coco_lr_osediff_val = {
    "name": "coco_lr_val",
    "im_dir": "datas/RobustSeg/val/coco/images/RealLR-ds2-orisize-osediff",
    # "im_dir": "datas/RobustSeg/val/coco/images/RealLR-ds-random-orisize/lr",
    "gt_dir": "datas/RobustSeg/val/coco/masks_random",
    "im_ext": ".jpg",
    "gt_ext": ".png"
}


dataset_lvis_lr_osediff_val = {
    "name": "lvis_lr_val",
    "im_dir": "datas/RobustSeg/val/LVIS_val/RealLR-ds2-orisize-osediff",
    # "im_dir": "datas/RobustSeg/val/LVIS_val/RealLR-ds-random-orisize/lr",
    "gt_dir": "datas/RobustSeg/val/LVIS_val/masks_random",
    "im_ext": ".jpg",
    "gt_ext": ".png"
}


# diffbir hr val sets
dataset_ECSSD_lr_diffbir_val = {
    "name": "ECSSD_val",
    "im_dir": "datas/cascade_psp/ecssd-RealLR-ds1-orisize-diffbir",
    "gt_dir": "datas/cascade_psp/ecssd",
    "im_ext": ".jpg",
    "gt_ext": ".png"
}


dataset_Thin_lr_diffbir_val = {
    "name": "thin_val",
    "im_dir": "/home/ps/Guo/Project/GleSAM-code/work-dir/diffbir-results/Thin",
    "gt_dir": "/home/ps/Guo/Project/sam-hq-main/train/data/thin_object_detection/ThinObject5K/masks_test",
    "im_ext": ".jpg",
    "gt_ext": ".png"
}


dataset_coco_lr_diffbir_val = {
    "name": "coco_lr_val",
    "im_dir": "datas/RobustSeg/val/coco/images/RealLR-ds1-orisize-diffbir",
    # "im_dir": "datas/RobustSeg/val/coco/images/RealLR-ds-random-orisize/lr",
    "gt_dir": "datas/RobustSeg/val/coco/masks_random",
    "im_ext": ".jpg",
    "gt_ext": ".png"
}

dataset_lvis_lr_diffbir_val = {
    "name": "lvis_lr_val",
    "im_dir": "/home/ps/Guo/Project/GleSAM-code/work-dir/diffbir-results/LVIS",
    "gt_dir": "/home/ps/Guo/dataset/LVIS/test/masks",
    "im_ext": ".jpg",
    "gt_ext": ".png"
}

# promptir hr val sets
dataset_ECSSD_lr_promptir_val = {
    "name": "ECSSD_val",
    "im_dir": "datas/cascade_psp/ecssd-RealLR-ds1-orisize-promptir",
    "gt_dir": "datas/cascade_psp/ecssd",
    "im_ext": ".png",
    "gt_ext": ".png"
}

dataset_Thin_lr_promptir_val = {
    "name": "thin_val",
    "im_dir": "/home/ps/Guo/Project/PromptIR-main/output/thin",
    "gt_dir": "/home/ps/Guo/Project/sam-hq-main/train/data/thin_object_detection/ThinObject5K/masks_test",
    "im_ext": ".png",
    "gt_ext": ".png"
}

dataset_coco_lr_promptir_val = {
    "name": "coco_lr_val",
    "im_dir": "datas/RobustSeg/val/coco/images/RealLR-ds1-orisize-promptir",
    # "im_dir": "datas/RobustSeg/val/coco/images/RealLR-ds-random-orisize/lr",
    "gt_dir": "datas/RobustSeg/val/coco/masks_random",
    "im_ext": ".png",
    "gt_ext": ".png"
}

dataset_lvis_lr_promptir_val = {
    "name": "lvis_lr_val",
    "im_dir": "/home/ps/Guo/Project/PromptIR-main/output/LVIS",
    "gt_dir": "/home/ps/Guo/dataset/LVIS/test/masks",
    "im_ext": ".png",
    "gt_ext": ".png"
}


def build_lrseg_train():
    train_dataset = [dataset_MSRA_train, dataset_Thin_train]
    return train_dataset


def build_lrseg_large_train():
    train_dataset = [dataset_MSRA_train, dataset_Thin_train, dataset_LVIS_train]
    train_dataset = [dataset_MSRA_train, dataset_Thin_train]
    return train_dataset


def build_lrseg_val():
    val_dataset = [dataset_Thin_lr4_val, dataset_lvis_lr4_val, dataset_ECSSD_lr4_val, dataset_coco_lr4_val]
    val_dataset = [dataset_ECSSD_lr4_val, dataset_coco_lr4_val, dataset_ECSSD_lr2_val, dataset_coco_lr2_val, dataset_ECSSD_lr1_val, dataset_coco_lr1_val]
    val_dataset = [dataset_LISU_val]
    val_dataset = [dataset_Thin_lr2_val, dataset_lvis_lr2_val, dataset_ECSSD_lr2_val]
    return val_dataset


def build_robust_degrade_val():
    val_dataset =[dataset_ECSSD_robust_degrade_val, dataset_coco_robust_degrade_val]
    return val_dataset


def build_clear_val():
    val_dataset = [dataset_ECSSD_val]
    return val_dataset


def build_osediff_val():
    val_dataset = [dataset_Thin_lr_osediff_val, dataset_lvis_lr_osediff_val, dataset_ECSSD_lr_osediff_val, dataset_coco_lr_osediff_val]
    return val_dataset


def build_diffbir_val():
    val_dataset = [ dataset_Thin_lr_diffbir_val, dataset_lvis_lr_diffbir_val, dataset_ECSSD_lr_diffbir_val, dataset_coco_lr_diffbir_val]
    val_dataset = [dataset_Thin_lr_diffbir_val, dataset_lvis_lr_diffbir_val]
    return val_dataset


def build_promptir_val():
    val_dataset = [dataset_Thin_lr_promptir_val, dataset_lvis_lr_promptir_val, dataset_ECSSD_lr_promptir_val, dataset_coco_lr_promptir_val]
    val_dataset = [dataset_Thin_lr_promptir_val, dataset_lvis_lr_promptir_val]
    return val_dataset


dataset_registry = {
    "lrseg_train": build_lrseg_train,
    "lrseg_large_train": build_lrseg_large_train,
    "lrseg_val": build_lrseg_val,
    "robust_deg_val": build_robust_degrade_val,
    "clear_val": build_clear_val,
    "osediff_val": build_osediff_val,
    "diffbir_val": build_diffbir_val,
    "promptir_val": build_promptir_val
}
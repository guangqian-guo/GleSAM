import argparse 
import shutil 
import torch 
import time 
from datetime import datetime
import os
from tqdm import tqdm
import yaml
import pytz
from omegaconf import OmegaConf
import numpy as np

from accelerate.utils import set_seed
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, CLIPTextModel
from diffusers.optimization import get_scheduler
from torchvision import transforms
from torchvision.utils import make_grid
from einops import rearrange
from diffusers import StableDiffusionPipeline
from peft.utils import get_peft_model_state_dict
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.utils import (
    convert_state_dict_to_diffusers,
    convert_unet_state_dict_to_peft,
)
import sam_utils.misc as misc
from models.sd_unifined_model_CRE_addlora import SDUniModel
from ram.models.ram_lora import ram
from utils.common import instantiate_from_config
from utils.others import cycle 
from ram import inference_ram as inference

from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide 
from utils.loss import *
import cv2
import matplotlib.cm as cm
import torch.nn.functional as F
import py_sod_metrics
import matplotlib.pyplot as plt
import random
import logging


def show_anns(masks, input_point, input_box, input_label, filename, image, ious, boundary_ious):
    if len(masks) == 0:
        return

    for i, (mask) in enumerate(masks):
        plt.figure(figsize=(10,10))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        if input_box is not None:
            show_box(input_box, plt.gca())
        if (input_point is not None) and (input_label is not None): 
            show_points(input_point, input_label, plt.gca())

        plt.axis('off')
        plt.savefig(filename+'_'+str(i)+'.png',bbox_inches='tight',pad_inches=-0.1)
        plt.close()

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=675):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='yellow', facecolor=(0,0,0,0), lw=4))    


def gray_to_color(img_tensor):
    img_numpy = img_tensor.numpy()
    # 使用 matplotlib 的颜色映射将灰度图像转换为伪彩色图像
    imgs = []
    for img_np in img_numpy:
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        img_np = np.uint8(255 * img_np)
        img = cv2.applyColorMap(img_np, cv2.COLORMAP_JET)
        imgs.append(img)
    img_colored = np.array(imgs)

    # 将彩色图像转换回 [C, W, H] 格式
    return torch.from_numpy(np.transpose(img_colored/255.0, (0, 3, 1, 2))).float()


def prepare_feat_for_saving(image_tensor):
    image_tensor = torch.mean(image_tensor, dim=1).data.cpu() # B W H
    image_tensor = F.interpolate(image_tensor.unsqueeze(1), size=(320, 320), mode='bilinear', align_corners=False)
    image_tensor = gray_to_color(image_tensor.squeeze(1))   # B 3 W H
    grid_image = make_grid(image_tensor, nrow=int(np.sqrt(image_tensor.size(0))), padding=2)  # 3 H W

    image = grid_image.permute(1, 2, 0).mul(255).byte().numpy()

    return image


def prepare_images_for_saving(image_tensor):
    grid_image = make_grid(image_tensor.cpu(), nrow=int(np.sqrt(image_tensor.size(0))), padding=2)
    image = grid_image.permute(1, 2, 0).mul(255).byte().numpy()
    return image


def compute_iou(preds, target):
    assert target.shape[1] == 1, 'only support one mask per image now'
    if(preds.shape[2]!=target.shape[2] or preds.shape[3]!=target.shape[3]):
        postprocess_preds = F.interpolate(preds, size=target.size()[2:], mode='bilinear', align_corners=False)
    else:
        postprocess_preds = preds
    iou = 0
    for i in range(0,len(preds)):
        iou = iou + misc.mask_iou(postprocess_preds[i],target[i])
    return iou / len(preds)


def compute_boundary_iou(preds, target):
    assert target.shape[1] == 1, 'only support one mask per image now'
    if(preds.shape[2]!=target.shape[2] or preds.shape[3]!=target.shape[3]):
        postprocess_preds = F.interpolate(preds, size=target.size()[2:], mode='bilinear', align_corners=False)
    else:
        postprocess_preds = preds
    iou = 0
    for i in range(0,len(preds)):
        iou = iou + misc.boundary_iou(target[i],postprocess_preds[i])
    return iou / len(preds)


def compute_pixel_accuracy(pred_mask, gt_mask):
    """
    Calculate pixel accuracy (PA) between a predicted mask and a ground truth mask.
    """
    if(pred_mask.shape[2]!=gt_mask.shape[2] or pred_mask.shape[3]!=gt_mask.shape[3]):
        pred_mask = F.interpolate(pred_mask, size=gt_mask.size()[2:], mode='bilinear', align_corners=False)
    gt_mask = gt_mask[0][0].detach().cpu().numpy()
    pred_mask = pred_mask[0][0].detach().cpu().numpy()
    # Threshold to binary mask if needed
    pred_mask = (pred_mask > 0.).astype(np.uint8)
    gt_mask = (gt_mask > 128).astype(np.uint8)
    correct_pixels = np.sum(gt_mask == pred_mask)
    total_pixels = gt_mask.size
    pixel_accuracy = correct_pixels / total_pixels
    return torch.tensor(pixel_accuracy).to('cuda')


def compute_dice_similarity_coefficient(pred_mask, gt_mask):
    """
    Calculate Dice coefficient between a predicted mask and a ground truth mask.
    """
    if(pred_mask.shape[2]!=gt_mask.shape[2] or pred_mask.shape[3]!=gt_mask.shape[3]):
        pred_mask = F.interpolate(pred_mask, size=gt_mask.size()[2:], mode='bilinear', align_corners=False)
    pred_mask = pred_mask.detach().cpu().numpy()
    gt_mask = gt_mask.detach().cpu().numpy()
    # Threshold to binary mask if needed
    pred_mask = (pred_mask > 0.).astype(np.uint8)
    gt_mask = (gt_mask > 128).astype(np.uint8)
    intersection = np.sum(np.logical_and(gt_mask, pred_mask))  # TP
    total_pixels = np.sum(gt_mask) + np.sum(pred_mask)  # Number of pixels in both masks
    dsc = (2.0 * intersection) / total_pixels if total_pixels > 0 else 0
    return torch.tensor(dsc).to('cuda')


class Trainer:
    def __init__(self, args):
        self.args = args

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True 

        accelerator_project_config = ProjectConfiguration(logging_dir=args.log_path)
        accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision="no",
            log_with="tensorboard",
            project_config=accelerator_project_config,
            kwargs_handlers=None,
            dispatch_batches=False
        )

        set_seed(args.seed + accelerator.process_index)

        print(accelerator.state)

        if accelerator.is_main_process:
            tz = pytz.timezone('Asia/Shanghai')
            current_time = datetime.now(tz).strftime("%Y-%m-%d_%H-%M-%S")
            # output_path = os.path.join(args.output_path, f"{current_time}_seed{args.seed}")
            output_path = args.output_path
            os.makedirs(output_path, exist_ok=True)

            args_dict = vars(args)
            args_file_path = os.path.join(output_path, 'config.yaml')
            with open(args_file_path, 'w') as yaml_file:
                yaml.dump(args_dict, yaml_file, default_flow_style=False)
            self.output_path = output_path
            self.writer = SummaryWriter(log_dir=output_path)
            print(f"TensorBoard log directory: {output_path}")

        self.model = SDUniModel(args, accelerator).to(accelerator.device)
        
        self.max_grad_norm = args.max_grad_norm
        self.step = 0

        self.tokenizer = AutoTokenizer.from_pretrained(args.model_id, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
                args.model_id, subfolder="text_encoder"
            ).to(accelerator.device)
        self.text_encoder.requires_grad_(False)

        dataset_cfg = OmegaConf.load(args.dataset_cfg)

        #============v3==============================================
        from dataset.build_datasets import dataset_registry
        from dataset.utils import load_img_gt_list
        train_set = dataset_cfg['params'].pop('train_set')
        val_set = dataset_cfg['params'].pop('val_set')
        val_dataset_cfg = dataset_cfg.copy()
        train_dataset_cfg = dataset_cfg.copy()
        
        # train set
        train_img_list, train_gt_list = load_img_gt_list(dataset_registry[train_set](), is_train=True)
        train_dataset_cfg['params']['dataset'] = {"img": train_img_list, "gt": train_gt_list}
        dataset = instantiate_from_config(train_dataset_cfg)
        decoder_dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        self.decoder_dataloader = accelerator.prepare(decoder_dataloader)
        
        # val
        self.val_dataloaders = []
        val_img_lists, val_gt_lists = load_img_gt_list(dataset_registry[val_set](), is_train=False)
        for i in range(len(val_img_lists)):
            val_dataset_cfg['params']['dataset'] = {"img": val_img_lists[i], "gt": val_gt_lists[i]}
            val_dataset = instantiate_from_config(val_dataset_cfg)
            # val_datasets.append(val_dataset)
            val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
            val_dataloader = accelerator.prepare(val_dataloader)
            self.val_dataloaders.append(val_dataloader)
        #===========================================================

        # self.DAPE = ram(pretrained=args.ram_path, 
        #                 pretrained_condition=args.ram_ft_path,
        #                 image_size=384,
        #                 vit='swin_l'
        #             ).to(accelerator.device)
        # self.DAPE.requires_grad_(False)
        # self.DAPE.eval()

        self.uncond_embedding = self.encode_prompt([""])

        # add mask decoder dataloader, optim, and scheduler =======================
        # ============= init SAM ======================================
        self.sam = sam_model_registry['vit_l_joint'](opt=None, checkpoint=args.sam_ckpt).to(accelerator.device)
        learnable_params = sum(p.numel() for p in self.sam.parameters() if p.requires_grad)
        print(learnable_params)
        self.sam_transform = ResizeLongestSide(self.sam.image_encoder.img_size)
        # ======================================================================

        self.optimizer_decoder = torch.optim.Adam([param for param in self.sam.parameters() if param.requires_grad], lr=args.generator_lr, weight_decay=1e-5)
        self.decoder_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer_decoder, 10)
        self.loss_dice, self.loss_focal = DiceLoss(smooth=1e-6, reduction='mean'), FocalLoss(gamma=2, alpha=0.25)
        (self.sam, self.optimizer_decoder, self.decoder_lr_scheduler) = accelerator.prepare(self.sam, self.optimizer_decoder, self.decoder_lr_scheduler)
        # ==========================================================

        self.accelerator = accelerator
        self.train_iters = args.train_iters
        self.batch_size = args.batch_size
        self.log_iters = args.log_iters
        self.log_loss = args.log_loss

        self.max_checkpoint = args.max_checkpoint

        self.dfake_gen_update_ratio = args.dfake_gen_update_ratio

        if args.checkpoint_path is not None:
            self.load(args.checkpoint_path)

        self.previous_time = None 
        self.no_save = args.no_save

    # @torch.no_grad()
    # def get_prompt(self, image):
    #     ram_transforms = transforms.Compose([
    #         transforms.Resize((384, 384)),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #     ])
    #     lq = ram_transforms(image)
    #     captions = inference(lq, self.DAPE)
    #     return captions


    @torch.no_grad()
    def encode_prompt(self, prompt_batch):
        prompt_embeds_list = []
        for caption in prompt_batch:
            text_input_ids = self.tokenizer(
                caption, max_length=self.tokenizer.model_max_length,
                padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids
            prompt_embeds = self.text_encoder(text_input_ids.to(self.text_encoder.device))[0]
            prompt_embeds_list.append(prompt_embeds)
        prompt_embeds = torch.concat(prompt_embeds_list, dim=0)
        return prompt_embeds

    def unwrap_model(self, model):
        model = self.accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    def load(self, checkpoint_path):
        # this is used for non-fsdp models.
        self.step = int(checkpoint_path.replace("/", "").split("_")[-1])   # reload step
        print(self.accelerator.load_state(checkpoint_path, strict=False))
        self.accelerator.print(f"Loaded checkpoint from {checkpoint_path}")

    def save(self):
        # NOTE: we save the checkpoints to only one place
        # 1. output_path: save the latest few ckpts, this is assumed to be a permanent storage
        # training states 
        # If FSDP is used, we only save the model parameter as I haven't figured out how to save the optimizer state without oom yet, help is appreciated.
        # Otherwise, we use the default accelerate save_state function 
        # TODO: Only save the lora layers, this need define load/save_model_hook, 
        # TODO: then `accelerator.register_load/save_state_pre_hook(load/save_model_hook)`
        
        if self.fsdp:
            feedforward_state_dict = self.fsdp_state_dict(self.model.feedforward_model)
            guidance_model_state_dict = self.fsdp_state_dict(self.model.guidance_model)

        if self.accelerator.is_main_process:
            # overwrite if exists
            if os.path.exists(os.path.join(self.output_path, f"checkpoint_{self.step:06d}")):
                shutil.rmtree(os.path.join(self.output_path, f"checkpoint_{self.step:06d}"))

            output_path = os.path.join(self.output_path, f"checkpoint_{self.step:06d}")
            os.makedirs(output_path, exist_ok=True)
            print(f"start saving checkpoint to {output_path}")

            if self.fsdp:
                torch.save(feedforward_state_dict, os.path.join(output_path, f"pytorch_model.bin"))
                del feedforward_state_dict
                torch.save(guidance_model_state_dict, os.path.join(output_path, f"pytorch_model_1.bin"))
                del guidance_model_state_dict
            else:
                self.accelerator.save_state(output_path)

            unwrapped_unet = self.unwrap_model(self.model.feedforward_model)
            unet_lora_state_dict = convert_state_dict_to_diffusers(
                get_peft_model_state_dict(unwrapped_unet)
            )

            StableDiffusionPipeline.save_lora_weights(
                save_directory=output_path,
                unet_lora_layers=unet_lora_state_dict,
                safe_serialization=True,
            )

            checkpoints = sorted(
                [folder for folder in os.listdir(self.output_path) if folder.startswith("checkpoint_")]
            )

            if len(checkpoints) > self.max_checkpoint:
                for folder in checkpoints[:-self.max_checkpoint]:
                    shutil.rmtree(os.path.join(self.output_path, folder))
            print("done saving")
        torch.cuda.empty_cache()

    def train_decoder(self):
        self.val_load(args.ckpt_path_for_val)
        step = 0
        best_iou = 0
        # ========= evaluate before training ===========================
        # averaged_results = self.val_decoder(load=False)  
        # if self.accelerator.is_main_process:
        #     logging.info('====> Epoch 0: {}'.format(averaged_results))
        # ==============================================================
        for epoch in range(6): 
            epoch_loss = 0
            for batch in tqdm(self.decoder_dataloader):
                step += 1
                gt, lq, gt_mask, point_prompt, point_label = batch['gt'], batch['lq'], batch['mask'], batch['point_prompt'], batch['point_label']
                ori_mask = batch['ori_mask']
                self.sam.train()
                accelerator = self.accelerator
                visual = False
                COMPUTE_GENERATOR_GRADIENT = False
                
                gt = rearrange(gt, "b h w c -> b c h w").contiguous().float().to(self.accelerator.device)   # rgb chw 0-1 
                lq = rearrange(lq, "b h w c -> b c h w").contiguous().float().to(self.accelerator.device)
                gt_mask = gt_mask.to(self.accelerator.device).float().unsqueeze(1) * 255.0   # 0-255
                point_label = point_label.to(self.accelerator.device)
                point_prompt = point_prompt.to(self.accelerator.device)

                # =============
                gt_lq = torch.cat((gt, lq), dim=0) * 255.0   # Tensor; [0, 1] -> [0, 255]; RGB; HWC; 
                transformed_gt_lq = self.sam_transform.apply_image_torch(gt_lq)

                batched_input = []
                for i in range(gt_lq.shape[0]):
                    data_dict = {}
                    data_dict['image'] = transformed_gt_lq[i]
                    data_dict['original_size'] = gt_lq.shape[-2:]
                    batched_input.append(data_dict)
                    
                batched_output = self.sam(batched_input, multimask_output=False, forward_encoder=True, forward_decoder=False)   # 2 256 64 64
                
                latent = []
                for b_out in batched_output:
                    latent.append(b_out['encoder_embedding'])
                latent = torch.cat(latent, dim=0)

                # latent = batched_output['src']
                hq_feat = latent[:gt.shape[0]]
                lq_feat = latent[gt.shape[0]:]
                del latent, batched_output

                real_train_dict = {
                    "gt_image": gt,
                    "hq_feat": hq_feat
                }

                # prompt = self.get_prompt(lq)   # NOTE: generate prompt using RAM, which is not necessary
                prompt = ['']
                text_embedding = self.encode_prompt(prompt)
                uncond_embedding = self.uncond_embedding.repeat(len(text_embedding), 1, 1)

                generator_loss_dict, generator_log_dict = self.model(
                    lq, lq_feat, text_embedding, uncond_embedding,
                    visual=visual,
                    compute_generator_gradient=COMPUTE_GENERATOR_GRADIENT,
                    real_train_dict=real_train_dict,
                    generator_turn=True,
                    guidance_turn=False
                )

                pred_feat = generator_log_dict['guidance_data_dict']["image"] / args.feat_weight
                            
                # pred mask  realise random prompt
                # prompt_type = ['point', 'noise_mask']
                box_prompt = misc.masks_to_boxes(gt_mask)

                # labels_256 = F.interpolate(gt_mask, size=(256, 256), mode='bilinear')
                # labels_noisemask = misc.masks_noise(labels_256)

                batched_input = []
                for i in range(pred_feat.shape[0]):
                    # input_type = random.choice(prompt_type)
                    data_dict = {}
                    data_dict["embeddings"] = pred_feat[i]
                    
                    data_dict['point_coords'] = self.sam_transform.apply_coords_torch(point_prompt[i], gt_lq.shape[-2:]).unsqueeze(0)
                    data_dict['point_labels'] = point_label[i].unsqueeze(0)
                    # elif input_type == 'box':
                    # data_dict['boxes'] = self.sam_transform.apply_boxes_torch(box_prompt[i:i+1], gt_lq.shape[-2:])
                    # elif input_type == 'noise_mask':
                    #     data_dict['mask_inputs'] = labels_noisemask[i:i+1]
                    data_dict['original_size'] = gt_lq.shape[-2:]
                    
                    batched_input.append(data_dict)

                batched_output = self.sam(batched_input, multimask_output=False, forward_encoder=False, forward_decoder=True)   # 2 256 64 64

                # compute loss  NOTE only support bs=1
                self.optimizer_decoder.zero_grad()
                pred_masks = batched_output[0]["low_res_logits"]
                pred_masks = F.interpolate(pred_masks, gt_mask.shape[-2:], mode="bilinear", align_corners=False)
                
                dice_loss_gt = self.loss_dice(pred_masks, gt_mask / 255.0)
                focal_loss_gt = self.loss_focal(pred_masks, gt_mask / 255.0)       
                mask_loss_gt = focal_loss_gt + dice_loss_gt 

                if accelerator.is_main_process:
                    self.writer.add_scalar("mask_loss_gt", mask_loss_gt, step)
                
                # print(mask_loss_gt.requires_grad)
                # mask_loss_gt.requires_grad_(True)
                # self.accelerator.backward(mask_loss_gt)
                mask_loss_gt.backward()
                self.optimizer_decoder.step()
                
                epoch_loss += mask_loss_gt

                if step % 500 == 0 and accelerator.is_main_process:
                    print("===> Epoch: {}  Step: {} loss: {:.4f}".format(epoch, step, mask_loss_gt))
                    for name, param in self.sam.named_parameters():
                        if param.requires_grad:
                            self.writer.add_histogram('params/{}'.format(name), param, step)
        
            # lr_scheduler
            self.decoder_lr_scheduler.step()

            # eval after each epoch
            # averaged_results = self.val_decoder(load=False)
            # current_iou = averaged_results['val_iou']

            if self.accelerator.is_main_process:
                torch.save(self.sam.state_dict(), os.path.join(self.output_path, "epoch_last.pth"))
                print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss.item() / len(self.decoder_dataloader)))
                # if current_iou > best_iou:
                #     best_iou = current_iou
                #     print('===========> Saved to epoch_best.pth! ==========')
                #     torch.save(self.sam.state_dict(), os.path.join(self.output_path, "epoch_best.pth"))

    def val_decoder(self, load=True):
        if load:
            self.val_load(args.ckpt_path_for_val)
        
        epoch_loss = 0
        for dataloader in self.val_dataloaders:
            step = 0
            metric_logger = misc.MetricLogger(delimiter="  ")
            clear_degree_list = []
            iou_list = []
            total_time = 0
            for batch in tqdm(dataloader):
                step += 1
                gt, lq, gt_mask, point_prompt, point_label = batch['gt'], batch['lq'], batch['mask'], batch['point_prompt'], batch['point_label']
                ori_mask = batch['ori_mask']
            
                self.sam.eval()
                accelerator = self.accelerator
                visual = False
                COMPUTE_GENERATOR_GRADIENT = False
                gt = rearrange(gt, "b h w c -> b c h w").contiguous().float().to(self.accelerator.device)   # rgb chw 0-1 
                lq = gt
                gt_mask = gt_mask.to(self.accelerator.device).float().unsqueeze(1) * 255.0   # 0-255
                ori_mask = ori_mask.to(self.accelerator.device).float().unsqueeze(1) * 255.0   # 0-255
                point_label = point_label.to(self.accelerator.device)
                point_prompt = point_prompt.to(self.accelerator.device)

                # =============
                gt_lq = torch.cat((gt, lq), dim=0) * 255.0   # Tensor; [0, 1] -> [0, 255]; RGB; CHW; 
                transformed_gt_lq = self.sam_transform.apply_image_torch(gt_lq)
                batched_input = []
                for i in range(gt_lq.shape[0]):
                    data_dict = {}
                    data_dict['image'] = transformed_gt_lq[i]
                    data_dict['original_size'] = gt_lq.shape[-2:]
                    batched_input.append(data_dict)

                batched_output = self.sam(batched_input, multimask_output=False, forward_encoder=True, forward_decoder=False)   # 2 256 64 64
                
                latent = []
                for b_out in batched_output:
                    latent.append(b_out['encoder_embedding'])
                latent = torch.cat(latent, dim=0)

                hq_feat = latent[:gt.shape[0]]
                lq_feat = latent[gt.shape[0]:]
                del latent, batched_output


                real_train_dict = {
                    "gt_image": gt,
                    "hq_feat": hq_feat
                }

                # prompt = self.get_prompt(lq)    # NOTE: generate prompt using RAM, which is not necessary
                # print(prompt)
                prompt = ['']
                text_embedding = self.encode_prompt(prompt)
                uncond_embedding = self.uncond_embedding.repeat(len(text_embedding), 1, 1)

                generator_loss_dict, generator_log_dict = self.model(
                    lq, lq_feat, text_embedding, uncond_embedding,
                    visual=visual,
                    compute_generator_gradient=COMPUTE_GENERATOR_GRADIENT,
                    real_train_dict=real_train_dict,
                    generator_turn=True,
                    guidance_turn=False
                )

                pred_feat = generator_log_dict['guidance_data_dict']["image"] / args.feat_weight

                # box_prompt = misc.masks_to_boxes(gt_mask)

                batched_input = []
                for i in range(1):
                    data_dict = {}
                    data_dict["embeddings"] = pred_feat[i]
                    data_dict['point_coords'] = self.sam_transform.apply_coords_torch(point_prompt[i], gt_lq.shape[-2:]).unsqueeze(0)
                    data_dict['point_labels'] = point_label[i].unsqueeze(0)
                    # data_dict['boxes'] = self.sam_transform.apply_boxes_torch(box_prompt[i:i+1], gt_lq.shape[-2:])
                    data_dict['original_size'] = gt_lq.shape[-2:]
                    batched_input.append(data_dict)

                batched_output = self.sam(batched_input, multimask_output=False, forward_encoder=False, forward_decoder=True)   # 2 256 64 64

                # compute loss  NOTE only support bs=1
                pred_masks = batched_output[0]["low_res_logits"]
                
                iou = compute_iou(pred_masks, ori_mask)
                iou_list.append(iou.item())

                boundary_iou = compute_boundary_iou(pred_masks, ori_mask)
                PA = compute_pixel_accuracy(pred_masks, ori_mask)
                Dice = compute_dice_similarity_coefficient(pred_masks, ori_mask)

                if visual:
                    os.makedirs(os.path.join(self.output_path, 'vis'), exist_ok=True)
                    masks_hq_vis = (F.interpolate(pred_masks.detach(), (512, 512), mode="bilinear", align_corners=False) > 0).cpu()
                    img_vis = gt_lq.permute(0,2,3,1)
                    for ii in range(len(gt)):
                        imgs_ii = img_vis[ii].detach().cpu().numpy().astype(dtype=np.uint8)
                        show_iou = torch.tensor(0)
                        show_boundary_iou = torch.tensor(0)
                        save_base = os.path.join(self.output_path, 'vis', '{}.jpg'.format(step))
                        # show_anns(masks_hq_vis[ii], None, labels_box[ii].cpu(), None, save_base , imgs_ii, show_iou, show_boundary_iou)   
                        show_anns(masks_hq_vis[ii], point_prompt[ii].cpu(), None, np.ones(point_prompt.shape[1]), save_base , imgs_ii, show_iou, show_boundary_iou)    

                loss_dict = {"val_iou": iou, "val_boundary_iou": boundary_iou, "PA": PA, "dsc": Dice}
                # print(loss_dict)
                loss_dict_reduced = misc.reduce_dict(loss_dict)   # 多个进程平均
                # print(loss_dict_reduced)
                metric_logger.update(**loss_dict_reduced)   # 这一个应该是求平均
                # print(metric_logger)

            # print(clear_degree_list, iou_list)
            # exit()
            # from datas.dense_map import draw_map
            # draw_map({"clearity": clear_degree_list, "iou": iou})
            # exit()
            metric_logger.synchronize_between_processes()
            print("Averaged stats:", metric_logger)
            resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
            if self.accelerator.is_main_process:
                logging.info('{}'.format(resstat))
        return resstat


    def val_load(self, checkpoint_path):  # val 
        from safetensors.torch import load_file
        state_dict = load_file(checkpoint_path)
        new_state = {}
        for k, v in state_dict.items():
            new_k = k[5:]
            new_k = new_k.split('.weight')[0]+'.default.weight'
            if "lora.up" in new_k:
                # print(new_k)
                new_k = new_k.replace('lora.up', 'lora_B')
            elif "lora.down" in new_k:
                new_k = new_k.replace('lora.down', 'lora_A')
            
            new_state[new_k] = v

        info = self.model.feedforward_model.load_state_dict(new_state, strict=False)
        if self.accelerator.is_main_process:
            print(info)
        # self.accelerator.load_state(checkpoint_path, strict=True)
        # self.accelerator.print(f"Loaded checkpoint from {checkpoint_path}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="")
    parser.add_argument('--ram_path', type=str, default='pretrained-weights/ram_swin_large_14m.pth')
    parser.add_argument('--ram_ft_path', type=str, default='pretrained-weights/DAPE.pth')
    parser.add_argument('--sam_ckpt', type=str, default='pretrained-weights/sam_vit_l_0b3195.pth') 
    parser.add_argument('--feat_weight', type=int, default=1)  # added by guo  给特征加权
    # parser.add_argument('--noise_step', type=int, default=10)  # added by guo  给特征n加oise的步数
    parser.add_argument("--eval", action="store_true") # added by guo
    
    parser.add_argument("--ckpt_path_for_val", type=str, default=None)
    parser.add_argument("--output_path", type=str, default="exp")
    parser.add_argument("--dataset_cfg", type=str, default=None)
    parser.add_argument("--log_path", type=str, default="tb-log")
    parser.add_argument("--train_iters", type=int, default=1000000)
    parser.add_argument("--log_iters", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=114)
    # parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--visual_iters", type=int, default=100)
    parser.add_argument("--max_grad_norm", type=float, default=10.0, help="max grad norm for network")
    parser.add_argument("--warmup_step", type=int, default=500, help="warmup step for network")
    parser.add_argument("--min_step_percent", type=float, default=0.02, help="minimum step percent for training")
    parser.add_argument("--max_step_percent", type=float, default=0.98, help="maximum step percent for training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--use_fp16", action="store_true")
    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--real_guidance_scale", type=float, default=6.0)
    parser.add_argument("--fake_guidance_scale", type=float, default=1.0)
    parser.add_argument("--no_save", action="store_true", help="don't save ckpt for debugging only")
    parser.add_argument("--cache_dir", type=str, default="/mnt/localssd/cache")
    parser.add_argument("--log_loss", action="store_true", help="log loss at every iteration")
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--latent_channel", type=int, default=4)
    parser.add_argument("--max_checkpoint", type=int, default=5)
    parser.add_argument("--dfake_gen_update_ratio", type=int, default=1)
    parser.add_argument("--generator_lr", type=float)
    parser.add_argument("--guidance_lr", type=float)
    parser.add_argument("--spatial_loss", action="store_true")
    parser.add_argument("--cls_on_clean_image", action="store_true")
    parser.add_argument("--gen_cls_loss", action="store_true")
    parser.add_argument("--percep_weight", type=float, default=0)
    parser.add_argument("--gen_cls_loss_weight", type=float, default=1)
    parser.add_argument("--guidance_cls_loss_weight", type=float, default=1)

    parser.add_argument("--generator_ckpt_path", type=str)
    parser.add_argument("--conditioning_timestep", type=int, default=999)
    parser.add_argument("--gradient_checkpointing", action="store_true", help="apply gradient checkpointing for dfake and generator. this might be a better option than FSDP")
    parser.add_argument("--dm_loss_weight", type=float, default=1.0)

    parser.add_argument("--use_x0", action="store_true")
    parser.add_argument("--denoising_timestep", type=int, default=1000)
    parser.add_argument("--num_denoising_step", type=int, default=1)
    parser.add_argument("--denoising_loss_weight", type=float, default=1.0)

    parser.add_argument("--diffusion_gan", action="store_true")
    parser.add_argument("--diffusion_gan_max_timestep", type=int, default=0)
    parser.add_argument("--revision", type=str)

    parser.add_argument("--real_image_path", type=str)
    parser.add_argument("--gan_alone", action="store_true", help="only use the gan loss without dmd")
    parser.add_argument("--backward_simulation", action="store_true")

    parser.add_argument("--generator_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=float, default=8)
    parser.add_argument("--lora_dropout", type=float, default=0.0)

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    assert args.gradient_accumulation_steps == 1, "grad accumulation not supported yet"
    
    return args 

if __name__ == "__main__":
    args = parse_args()

    logging.basicConfig(filename=os.path.join(args.output_path, 'log.log'),
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    trainer = Trainer(args)

    if args.eval:
        trainer.val_decoder(None)
    else:
        trainer.train_decoder()
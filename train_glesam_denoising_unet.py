import sys
print(sys.executable)

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

from models.sd_unifined_model_CRE_addlora import SDUniModel
from ram.models.ram_lora import ram
from utils.common import instantiate_from_config
from utils.others import cycle 
from ram import inference_ram as inference

from segment_anything import sam_model_registry
import cv2
import matplotlib.cm as cm
import torch.nn.functional as F


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
    image = image[..., ::-1]
    return image


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
            output_path = os.path.join(args.output_path, f"{current_time}_seed{args.seed}")
            os.makedirs(output_path, exist_ok=False)

            args_dict = vars(args)
            args_file_path = os.path.join(output_path, 'config.yaml')
            with open(args_file_path, 'w') as yaml_file:
                yaml.dump(args_dict, yaml_file, default_flow_style=False)

            self.output_path = output_path
            os.makedirs(args.log_path, exist_ok=True)
            self.writer = SummaryWriter(log_dir=output_path)
            print(f"TensorBoard log directory: {output_path}")

        self.model = SDUniModel(args, accelerator)
    
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
        # ============= init SAM encoder ======================================
        self.sam_encoder = sam_model_registry['encoder'](checkpoint=args.sam_ckpt).to(accelerator.device)
        self.sam_encoder.requires_grad_(False)
        self.sam_encoder.eval()
        # ======================================================================

        self.uncond_embedding = self.encode_prompt([""])
        dataloader = torch.utils.data.DataLoader(
            dataset, num_workers=args.num_workers, 
            batch_size=args.batch_size, shuffle=False,
            drop_last=True
        )

        dataloader = accelerator.prepare(dataloader)
        self.dataloader = cycle(dataloader)

        # use two dataloader 
        # as the generator and guidance model are trained at different paces 
        guidance_dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        guidance_dataloader = accelerator.prepare(guidance_dataloader)
        self.guidance_dataloader = cycle(guidance_dataloader)

        self.guidance_cls_loss_weight = args.guidance_cls_loss_weight

        self.gen_cls_loss = args.gen_cls_loss 
        self.gen_cls_loss_weight = args.gen_cls_loss_weight 

        self.fsdp = False # Not realized now

        self.optimizer_generator = torch.optim.AdamW(
            [param for param in self.model.feedforward_model.parameters() if param.requires_grad], 
            lr=args.generator_lr, 
            betas=(0.9, 0.999),  # pytorch's default 
            weight_decay=0.01  # pytorch's default 
        )
        
        self.optimizer_guidance = torch.optim.AdamW(
            [param for param in self.model.guidance_model.parameters() if param.requires_grad], 
            lr=args.guidance_lr, 
            betas=(0.9, 0.999),  # pytorch's default 
            weight_decay=0.01  # pytorch's default 
        )

        # 计算 optimizer_generator 优化的参数量
        generator_params = sum(p.numel() for p in self.model.feedforward_model.parameters() if p.requires_grad)
        print(f"Generator Optimizer is optimizing {generator_params} parameters.")

        # 计算 optimizer_guidance 优化的参数量
        guidance_params = sum(p.numel() for p in self.model.guidance_model.parameters() if p.requires_grad)
        print(f"Guidance Optimizer is optimizing {guidance_params} parameters.")


        self.scheduler_generator = get_scheduler(
            "constant_with_warmup",
            optimizer=self.optimizer_generator,
            num_warmup_steps=args.warmup_step,
            num_training_steps=args.train_iters 
        )
        
        self.scheduler_guidance = get_scheduler(
            "constant_with_warmup",
            optimizer=self.optimizer_guidance,
            num_warmup_steps=args.warmup_step,
            num_training_steps=args.train_iters 
        )

        # the self.model is not wrapped in ddp, only its two subnetworks are wrapped 
        (
            self.model.feedforward_model, self.model.guidance_model, self.optimizer_generator,
            self.optimizer_guidance, self.scheduler_generator, self.scheduler_guidance 
        ) = accelerator.prepare(
            self.model.feedforward_model, self.model.guidance_model, self.optimizer_generator, 
            self.optimizer_guidance, self.scheduler_generator, self.scheduler_guidance
        )

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

    @torch.no_grad()
    def get_prompt(self, image):
        ram_transforms = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        lq = ram_transforms(image)
        captions = inference(lq, self.DAPE)
        return captions

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

    def train_one_step(self):
        self.model.train()

        accelerator = self.accelerator

        visual = self.step % self.args.visual_iters == 0

        COMPUTE_GENERATOR_GRADIENT = self.step % self.dfake_gen_update_ratio == 0
        
        if COMPUTE_GENERATOR_GRADIENT: 
            batch = next(self.dataloader)
        else:
            batch = next(self.guidance_dataloader)
        gt, lq, gt_mask, point_prompt, point_label = batch['gt'], batch['lq'], batch['mask'], batch['point_prompt'], batch['point_label']
            
        gt = rearrange(gt, "b h w c -> b c h w").contiguous().float().to(self.accelerator.device)   # rgb chw 0-1 
        lq = rearrange(lq, "b h w c -> b c h w").contiguous().float().to(self.accelerator.device)
            
        # =============
        gt_lq = torch.cat((gt, lq), dim=0)
        latent = self.sam_encoder(gt_lq)   # 2 256 64 64
        
        hq_feat = latent[:gt.shape[0]].to(torch.float)          # torch.float16
        lq_feat = latent[gt.shape[0]:].to(torch.float)          # torch.float16
        
        
        del latent

        real_train_dict = {
            "gt_image": gt,
            "hq_feat": hq_feat
        }

        # prompt = self.get_prompt(lq)   # 文本prompt NOTE: generate prompt using RAM, which is not necessary
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

        # first update the generator if the current step is a multiple of dfake_gen_update_ratio
        generator_loss = 0.0

        if COMPUTE_GENERATOR_GRADIENT:
            if self.args.spatial_loss:
                generator_loss += generator_loss_dict["loss_spatial"]

            if not self.args.gan_alone:
                generator_loss += generator_loss_dict["loss_dm"] * self.args.dm_loss_weight

            if self.args.cls_on_clean_image and self.args.gen_cls_loss:
                generator_loss += generator_loss_dict["gen_cls_loss"] * self.gen_cls_loss_weight
                
            self.accelerator.backward(generator_loss)
            generator_grad_norm = accelerator.clip_grad_norm_(self.model.feedforward_model.parameters(), self.max_grad_norm)
            self.optimizer_generator.step()
            
            # if we also compute gan loss, the classifier may also receive gradient 
            # zero out guidance model's gradient avoids undesired gradient accumulation
            self.optimizer_generator.zero_grad() 
            self.optimizer_guidance.zero_grad()

        self.scheduler_generator.step()

        # update the guidance model (dfake and classifier)
        guidance_loss_dict, guidance_log_dict = self.model(
            lq, lq_feat, text_embedding, uncond_embedding,
            # real_train_dict=real_train_dict,
            compute_generator_gradient=COMPUTE_GENERATOR_GRADIENT,
            generator_turn=False,
            guidance_turn=True,
            guidance_data_dict=generator_log_dict['guidance_data_dict']
        )

        guidance_loss = 0
        guidance_loss += guidance_loss_dict["loss_fake_mean"]
        if self.args.cls_on_clean_image:
            guidance_loss += guidance_loss_dict["guidance_cls_loss"] * self.guidance_cls_loss_weight
        self.accelerator.backward(guidance_loss)
        guidance_grad_norm = accelerator.clip_grad_norm_(self.model.guidance_model.parameters(), self.max_grad_norm)
        self.optimizer_guidance.step()
        self.optimizer_guidance.zero_grad()
        self.optimizer_generator.zero_grad() # zero out the generator's gradient as well
        self.scheduler_guidance.step()

        # combine the two dictionaries
        loss_dict = {**generator_loss_dict, **guidance_loss_dict}
        log_dict = {**generator_log_dict, **guidance_log_dict}

        if accelerator.is_main_process and self.log_loss and (not visual):
            self.writer.add_scalar("loss/loss_fake_mean", guidance_loss_dict['loss_fake_mean'].item(), self.step)
            self.writer.add_scalar("grad_norm/guidance_grad_norm", guidance_grad_norm.item(), self.step)

            if COMPUTE_GENERATOR_GRADIENT:
                self.writer.add_scalar("grad_norm/generator_grad_norm", generator_grad_norm.item(), self.step)
                if self.args.spatial_loss:
                    self.writer.add_scalar("loss/loss_spatial", loss_dict['loss_spatial'].item(), self.step)
                    self.writer.add_scalar("loss/loss_mse", loss_dict['loss_mse'].item(), self.step)
                if not self.args.gan_alone:
                    self.writer.add_scalar("loss/loss_dm", loss_dict['loss_dm'].item(), self.step)
                    self.writer.add_scalar("grad_norm/dmtrain_gradient_norm", log_dict['dmtrain_gradient_norm'], self.step)
                if self.args.gen_cls_loss:
                    self.writer.add_scalar("loss/gen_cls_loss", loss_dict['gen_cls_loss'].item(), self.step)
            if self.args.cls_on_clean_image:
                self.writer.add_scalar("loss/guidance_cls_loss", loss_dict['guidance_cls_loss'].item(), self.step)

        if visual:
            log_dict['lq_feat'] = accelerator.gather(lq_feat * args.feat_weight)  # 低质量特征
            # log_dict['lq_feat'] = accelerator.gather(torch.mean(lq_feat.view(-1, 4, 64, 64, 64), dim=2) * 100)  # 低质量特征
            log_dict['pred_feat'] = accelerator.gather(log_dict['pred_image'])  # 预测的特征
            log_dict['gt_feat'] = accelerator.gather(real_train_dict['gt_latent'])  # hq feat
            log_dict['lq_image'] = accelerator.gather(lq)  
            log_dict['gt_image'] = accelerator.gather(real_train_dict['gt_image']) 
            log_dict['generated_noise'] = accelerator.gather(log_dict['generated_noise'])  

        if accelerator.is_main_process and visual:
            # Add TensorBoard images here if needed
            with torch.no_grad():
                pred_feat = log_dict['pred_feat']
                # pred_feat_grid = prepare_feat_for_saving(pred_feat)
                # self.writer.add_images("feat/pred_feat", pred_feat_grid, self.step, dataformats="HWC")
                self.writer.add_histogram("dist/pred_feat", pred_feat[0], self.step)
                # noise
                generated_noise = log_dict['generated_noise']
                self.writer.add_histogram("dist/noise", generated_noise[0], self.step)
                gt_feat = log_dict['gt_feat']
                # gt_feat_grid = prepare_feat_for_saving(gt_feat)
                # self.writer.add_images("feat/gt_feat", gt_feat_grid, self.step, dataformats="HWC")
                self.writer.add_histogram("dist/gt_feat", gt_feat[0], self.step)
                lq_feat = log_dict['lq_feat']
                # lq_feat_grid = prepare_feat_for_saving(lq_feat)
                # self.writer.add_images("feat/lq_feat", lq_feat_grid, self.step, dataformats="HWC")
                self.writer.add_histogram("dist/lq_feat", lq_feat[0], self.step)
                # gt_image = log_dict['gt_image']
                # gt_image_grid = prepare_images_for_saving(gt_image)
                # self.writer.add_images("image/gt_image", gt_image_grid, self.step, dataformats="HWC")
                # lq_image = log_dict['lq_image']
                # lq_image_grid = prepare_images_for_saving(lq_image)
                # self.writer.add_images("image/lq_image", lq_image_grid, self.step, dataformats="HWC")

        self.accelerator.wait_for_everyone()

    def train(self):
        progress_bar = tqdm(range(self.step, self.train_iters), desc="Training", unit="step")
        for index in progress_bar:
            self.train_one_step()
            if (not self.no_save) and self.step % self.log_iters == 0:
                self.save()
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    self.writer.add_scalar("time/per_iteration", current_time - self.previous_time, self.step)
                    self.previous_time = current_time
            self.step += 1

        # Save the lora layers
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            unet = self.model.feedforward_model.to(torch.float32)
            unwrapped_unet = self.unwrap_model(unet)
            unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrapped_unet))
            StableDiffusionPipeline.save_lora_weights(
                save_directory=self.output_path,
                unet_lora_layers=unet_lora_state_dict,
                safe_serialization=True,
            )
            torch.cuda.empty_cache()
        self.accelerator.end_training()
    
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
        print(info)
        # self.accelerator.load_state(checkpoint_path, strict=True)
        # self.accelerator.print(f"Loaded checkpoint from {checkpoint_path}")

    # val 
    def val(self, ckpt_path):        
        self.val_load(ckpt_path)
        os.makedirs(os.path.join(self.output_path, 'vis/pred_feat'), exist_ok=True)
        os.makedirs(os.path.join(self.output_path, 'vis/gt_feat'), exist_ok=True)
        os.makedirs(os.path.join(self.output_path, 'vis/lq_feat'), exist_ok=True)
        os.makedirs(os.path.join(self.output_path, 'vis/gt_img'), exist_ok=True)
        os.makedirs(os.path.join(self.output_path, 'vis/lq_img'), exist_ok=True)

        progress_bar = tqdm(self.dataloader, desc="val", unit="step")
        for index in progress_bar:
            self.model.eval()
            accelerator = self.accelerator
            visual = True
            COMPUTE_GENERATOR_GRADIENT = False
            batch = next(self.dataloader)
            gt, lq, gt_mask, point_prompt, point_label = batch['gt'], batch['lq'], batch['mask'], batch['point_prompt'], batch['point_label']
            gt = rearrange(gt, "b h w c -> b c h w").contiguous().float().to(self.accelerator.device)   # rgb chw 0-1 
            lq = rearrange(lq, "b h w c -> b c h w").contiguous().float().to(self.accelerator.device)
            
            # =============
            gt_lq = torch.cat((gt, lq), dim=0)
            latent = self.sam_encoder(gt_lq)   # 2 256 64 64
            hq_feat = latent[:gt.shape[0]]
            lq_feat = latent[gt.shape[0]:]
            del latent

            real_train_dict = {
                "gt_image": gt,
                "hq_feat": hq_feat
            }
            prompt = self.get_prompt(lq)
            text_embedding = self.encode_prompt(prompt)
            uncond_embedding = self.uncond_embedding.repeat(len(text_embedding), 1, 1)
            with torch.no_grad():
                generator_loss_dict, generator_log_dict = self.model(
                    lq, lq_feat, text_embedding, uncond_embedding,
                    visual=visual,
                    compute_generator_gradient=COMPUTE_GENERATOR_GRADIENT,
                    real_train_dict=real_train_dict,
                    generator_turn=True,
                    guidance_turn=False
                )
            log_dict = generator_log_dict
            
            if visual:
                log_dict['lq_feat'] = accelerator.gather(-1*lq_feat*args.feat_weight)  # 低质量特征
                # log_dict['lq_feat'] = accelerator.gather(torch.mean(lq_feat.view(-1, 4, 64, 64, 64), dim=2) * 100)  # 低质量特征
                log_dict['pred_feat'] = accelerator.gather(-1*log_dict['pred_image'])  # 预测的特征
                log_dict['gt_feat'] = accelerator.gather(-1*real_train_dict['gt_latent'])  # hq feat
                log_dict['lq_image'] = accelerator.gather(lq)  
                log_dict['gt_image'] = accelerator.gather(real_train_dict['gt_image']) 
                log_dict['generated_noise'] = accelerator.gather(log_dict['generated_noise'])  
            
            if accelerator.is_main_process and visual:
                # Add TensorBoard images here if needed
                with torch.no_grad():
                    pred_feat = log_dict['pred_feat']
                    pred_feat_grid = prepare_feat_for_saving(pred_feat)
                    cv2.imwrite(os.path.join(self.output_path, 'vis/pred_feat/{}.jpg'.format(self.step)), pred_feat_grid)
                    # self.writer.add_images("feat/pred_feat", pred_feat_grid, self.step, dataformats="HWC")
                    # self.writer.add_histogram("dist/pred_feat", pred_feat[0], self.step)
                    # noise
                    # generated_noise = log_dict['generated_noise']
                    # self.writer.add_histogram("dist/noise", generated_noise[0], self.step)
                    gt_feat = log_dict['gt_feat']
                    gt_feat_grid = prepare_feat_for_saving(gt_feat)
                    cv2.imwrite(os.path.join(self.output_path, 'vis/gt_feat/{}.jpg'.format(self.step)), gt_feat_grid)
                    # self.writer.add_images("feat/gt_feat", gt_feat_grid, self.step, dataformats="HWC")
                    # self.writer.add_histogram("dist/gt_feat", gt_feat[0], self.step)
                    lq_feat = log_dict['lq_feat']
                    lq_feat_grid = prepare_feat_for_saving(lq_feat)
                    cv2.imwrite(os.path.join(self.output_path, 'vis/lq_feat/{}.jpg'.format(self.step)), lq_feat_grid)
                    # self.writer.add_images("feat/lq_feat", lq_feat_grid, self.step, dataformats="HWC")
                    # self.writer.add_histogram("dist/lq_feat", lq_feat[0], self.step)
                    gt_image = log_dict['gt_image']
                    gt_image_grid = prepare_images_for_saving(gt_image)
                    cv2.imwrite(os.path.join(self.output_path, 'vis/gt_img/{}.jpg'.format(self.step)), gt_image_grid)
                    # self.writer.add_images("image/gt_image", gt_image_grid, self.step, dataformats="HWC")
                    lq_image = log_dict['lq_image']
                    lq_image_grid = prepare_images_for_saving(lq_image)
                    cv2.imwrite(os.path.join(self.output_path, 'vis/lq_img/{}.jpg'.format(self.step)), lq_image_grid)
                    # self.writer.add_images("image/lq_image", lq_image_grid, self.step, dataformats="HWC")
            self.accelerator.wait_for_everyone()
            self.step += 1

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="")
    parser.add_argument('--ram_path', type=str, default='/home/ps/Guo/Project/GleSAM-code/pretrained-weights/ram_swin_large_14m.pth')
    parser.add_argument('--ram_ft_path', type=str, default='/home/ps/Guo/Project/GleSAM-code/pretrained-weights/DAPE.pth')
    parser.add_argument('--sam_ckpt', type=str, default='/home/ps/Guo/Project/GleSAM-code/pretrained-weights/sam_vit_l_0b3195.pth')
    parser.add_argument('--feat_weight', type=int, default=1)  # added by guo 
    parser.add_argument("--ckpt_path_for_val", type=str, default=None)  
    parser.add_argument("--output_path", type=str, default="exp")
    parser.add_argument("--dataset_cfg", type=str, default=None)
    parser.add_argument("--log_path", type=str, default="tb-log")
    parser.add_argument("--train_iters", type=int, default=100000)
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
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    trainer = Trainer(args)
    trainer.train()

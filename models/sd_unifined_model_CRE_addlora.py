from diffusers import UNet2DConditionModel
from accelerate.utils import broadcast
from peft import LoraConfig
from torch import nn
import torch 
import torch.nn.functional as F
import pyiqa
from einops import rearrange
import numpy as np
import cv2

from models.sd_guidance_model_CRE_addlora import SDGuidance
from utils.others import NoOpContext, get_prev_sample_from_noise, get_x0_from_noise


class SDUniModel(nn.Module):
    def __init__(self, args, accelerator):
        super().__init__()

        self.args = args
        self.accelerator = accelerator
        self.guidance_model = SDGuidance(args, accelerator) 
        self.num_train_timesteps = self.guidance_model.num_train_timesteps
        self.conditioning_timestep = args.conditioning_timestep 
        self.use_fp16 = args.use_fp16 
        self.gradient_checkpointing = args.gradient_checkpointing 
        self.backward_simulation = args.backward_simulation 

        self.denoising_timestep = args.denoising_timestep 
        self.noise_scheduler = self.guidance_model.scheduler
        self.num_denoising_step = args.num_denoising_step 

        self.timestep_interval = self.denoising_timestep//self.num_denoising_step

        self.feedforward_model = UNet2DConditionModel.from_pretrained(
            args.model_id, subfolder="unet"
        ).float()

        conv_in_weight = self.feedforward_model.conv_in.weight.data
        conv_in_bias = self.feedforward_model.conv_in.bias.data
        conv_out_weight = self.feedforward_model.conv_out.weight.data
        conv_out_bias = self.feedforward_model.conv_out.bias.data
        del self.feedforward_model

        self.feedforward_model = UNet2DConditionModel.from_pretrained(
            args.model_id, subfolder="unet", in_channels=256, out_channels=256, low_cpu_mem_usage=False, ignore_mismatched_sizes=True
        ).float()
        
        new_in_weight = conv_in_weight.repeat(1, 64, 1, 1)
        new_out_weight = conv_out_weight.repeat(64, 1, 1, 1)
        new_out_bias = conv_out_bias.repeat(64)

        self.feedforward_model.conv_in.weight.data.copy_(new_in_weight)
        self.feedforward_model.conv_in.bias.data.copy_(conv_in_bias)
        self.feedforward_model.conv_out.weight.data.copy_(new_out_weight)
        self.feedforward_model.conv_out.bias.data.copy_(new_out_bias)

        self.feedforward_model.requires_grad_(False)

        lora_target_modules = [
            "to_q", "to_k", "to_v", "to_out.0",
            "proj_in", "proj_out",
            "ff.net.0.proj", "ff.net.2",
            "conv1", "conv2", "conv_shortcut",
            "downsamplers.0.conv", "upsamplers.0.conv",
            "time_emb_proj", "conv_in", "conv_out"
        ]

        lora_config = LoraConfig(
            r=args.lora_rank,
            target_modules=lora_target_modules,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout
        )
        
        self.feedforward_model.add_adapter(lora_config)

        if self.gradient_checkpointing:
            self.feedforward_model.enable_gradient_checkpointing()

        self.alphas_cumprod = self.guidance_model.alphas_cumprod.to(accelerator.device)
        self.alphas = self.noise_scheduler.alphas.to(accelerator.device)
        self.betas = self.noise_scheduler.betas.to(accelerator.device)
        

        self.network_context_manager = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if self.use_fp16 else NoOpContext()

        if args.spatial_loss:
            if args.percep_weight > 0:
                self.lpips_loss = pyiqa.create_metric('lpips', device=accelerator.device, as_loss=True)

    def decode_image(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample.float().clamp(-1, 1)
        return image 

    def forward(self, lq, lq_feat, text_embedding, uncond_embedding, 
        visual=False, 
        real_train_dict=None,
        compute_generator_gradient=True,
        generator_turn=False,
        guidance_turn=False,
        guidance_data_dict=None
    ):
        assert (generator_turn and not guidance_turn) or (guidance_turn and not generator_turn) 

        if generator_turn:
            timesteps = torch.ones(lq.shape[0], device=lq.device, dtype=torch.long) * self.conditioning_timestep
        
            if compute_generator_gradient:   # 如果更新generator
                with self.network_context_manager:
                    lq_latent = lq_feat * self.args.feat_weight 
                    generated_noise = self.feedforward_model(
                        lq_latent,
                        timesteps.long(),
                        text_embedding
                    ).sample
            
            else:  # 不更新generator
                if self.gradient_checkpointing:
                    self.accelerator.unwrap_model(self.feedforward_model).disable_gradient_checkpointing()

                with torch.no_grad():
                    with self.network_context_manager:
                        lq_latent = lq_feat * self.args.feat_weight  
                        generated_noise = self.feedforward_model(
                            lq_latent,
                            timesteps.long(),
                            text_embedding
                        ).sample

                if self.gradient_checkpointing:
                    self.accelerator.unwrap_model(self.feedforward_model).enable_gradient_checkpointing()

            if self.args.use_x0:   # defaule True
                generated_image = get_x0_from_noise(    # 根据预测的噪声得到去噪的输出  latent
                    lq_latent.double(),
                    generated_noise.double(), self.alphas_cumprod.double(), timesteps
                ).float()

            else:
                generated_image = get_prev_sample_from_noise(
                    lq_latent.double(), 
                    generated_noise.double(),
                    self.alphas.double(),
                    self.betas.double(),
                    timesteps
                ).float()
                

            with torch.no_grad():
                with self.network_context_manager:
                    gt_latent = real_train_dict["hq_feat"] * self.args.feat_weight
                    
                real_train_dict["gt_latent"] = gt_latent

            if compute_generator_gradient:
                generator_data_dict = {
                    "image": generated_image,    # 其实是去噪后的 latent
                    "text_embedding": text_embedding,
                    "uncond_embedding": uncond_embedding,
                    "real_train_dict": real_train_dict
                }

                # avoid any side effects of gradient accumulation
                self.guidance_model.requires_grad_(False)     
                
                loss_dict, log_dict = self.guidance_model(
                    generator_turn=True,
                    guidance_turn=False,
                    generator_data_dict=generator_data_dict
                )

                self.guidance_model.requires_grad_(True)

                if self.args.spatial_loss:   # L_data

                    with self.network_context_manager:
                        spatial_loss = 0
                        mse_loss = F.mse_loss(generated_image, real_train_dict["gt_latent"])
                        spatial_loss += mse_loss
                        loss_dict["loss_mse"] = mse_loss

                    loss_dict["loss_spatial"] = spatial_loss   # spatial loss 就是loss_data

            else:
                loss_dict = {}
                log_dict = {} 

            if visual:
                with torch.no_grad():
                    if self.use_fp16:
                        log_dict["pred_image"] = generated_image.half()
                        log_dict["decoded_gt_image"] = real_train_dict["gt_latent"].half()
                        log_dict["generated_noise"]  = generated_noise.half()
                    else:
                        log_dict["pred_image"] = generated_image
                        log_dict["decoded_gt_image"] = real_train_dict["gt_latent"]
                        log_dict["generated_noise"]  = generated_noise
                        

            log_dict["guidance_data_dict"] = {
                "image": generated_image.detach(),
                "text_embedding": text_embedding.detach(),
                "uncond_embedding": uncond_embedding.detach(),
                "real_train_dict": real_train_dict,
            }

            log_dict['denoising_timestep'] = timesteps

        elif guidance_turn:
            assert guidance_data_dict is not None
            
            loss_dict, log_dict = self.guidance_model(
                generator_turn=False,
                guidance_turn=True,
                guidance_data_dict=guidance_data_dict
            )

        return loss_dict, log_dict  


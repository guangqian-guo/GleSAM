# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type

from .common import LayerNorm2d
from .components import *
from .diffkd_modules import DiffusionModel, NoiseAdapter, AutoEncoder, DDIMPipeline
from .scheduling_ddim import DDIMScheduler

class MaskDecoderDiff(nn.Module):
    def __init__(
        self,
        *,
        opt=None,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        vit_dim: int=1024
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        if opt is not None:
            self.opt = opt

        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )

        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )
        
        # robust output token (ROT)
        self.custom_robust_token = nn.Embedding(self.num_mask_tokens, transformer_dim)
        # corresponding new MLP layer for ROT
        self.robust_mlp = MLP(transformer_dim, transformer_dim, transformer_dim//8, 3) 
   
        # # AMFG for mask features
        # self.fourier_mask_features = MaskFeatureBlock(transformer_dim=transformer_dim)
        # # AMFG for image encoder features                                               
        # self.fourier_first_layer_features = FirstLayerFeatureBlock(vit_dim=vit_dim, transformer_dim=transformer_dim)
        # self.fourier_last_layer_features = LastLayerFeatureBlock(transformer_dim=transformer_dim)
        
        # # AOTG
        self.custom_token_block = TokenBlock(input_dim=self.num_mask_tokens, mlp_dim=transformer_dim // self.num_mask_tokens)        

        # diffusion model
        self.model = DiffusionModel(channels_in=transformer_dim, kernel_size=3)
        self.scheduler = DDIMScheduler(num_train_timesteps=1000, clip_sample=False, beta_schedule="linear")
        # self.noise_adapter = NoiseAdapter(transformer_dim, 3)
        # pipeline for denoising student feature
        self.pipeline = DDIMPipeline(self.model, self.scheduler, None)
        self.proj = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )


    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        encoder_features: torch.Tensor, #TODO:
        robust_token_only: bool = False,
        clear: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        early_features = encoder_features[0].permute(0, 3, 1, 2)

        # pass image features of different level through AMFG
        # complementary_features = self.fourier_first_layer_features(early_features, clear=clear) 
        # final_image_embeddings = self.fourier_last_layer_features(image_embeddings, clear=clear)

        # robust_features = complementary_features + final_image_embeddings # fuse image's complementary features and final embeddings 

        masks, iou_pred, upscaled_embedding_robust, robust_token, noise_pred, noise = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            clear = clear
        )

        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]
        # Prepare output
        return masks, iou_pred, upscaled_embedding_robust, robust_token, noise, noise_pred


    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        clear: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        
        # Concatenate output tokens
        if clear: # original SAM output token
            output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0) 
        
        else: # RobustSAM output token
            output_tokens = torch.cat([self.iou_token.weight, self.custom_robust_token.weight], dim=0) 
        
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)      
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        if clear:
            upscaled_embedding_decoder = self.output_upscaling(src) # decoder output mask features
            # pred noise
            noise_pred, noise = self.ddim_pred(src)
        else:
            src_diff = self.pipeline(batch_size=b,
                                                    device=src.device,
                                                    dtype=src.dtype,
                                                    shape=src.shape[1:],
                                                    feat=src,
                                                    num_inference_steps=5,
                                                    )
            # upscaled_embedding_diff = self.output_upscaling(src_diff)
            upscaled_embedding_diff = self.proj(src_diff)
            noise_pred, noise = None, None

        # robust_features = robust_features.repeat(b,1,1,1)
        # mask_features = self.fourier_mask_features(upscaled_embedding_decoder, clear=clear) # pass original mask features through AMFG
       
        # upscaled_embedding_robust = mask_features + robust_features # fuse image features and mask features

        hyper_in_list: List[torch.Tensor] = []

        for i in range(self.num_mask_tokens): 
            if clear:
                hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
            
            else: # pass ROT through AOTG and corresponding MLP layers for ROT
                token = mask_tokens_out[:, i, :]
                token = self.custom_token_block(token)

                hyper_in_list.append(self.robust_mlp(token))
        
        hyper_in = torch.stack(hyper_in_list, dim=1)

        b, c, h, w = upscaled_embedding_decoder.shape if clear else upscaled_embedding_diff.shape

        # at inference stage, clear=False
        upscaled_embedding = upscaled_embedding_decoder if clear else upscaled_embedding_diff

        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
        robust_token = mask_tokens_out

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)
        
        return masks, iou_pred, src if clear else src_diff, robust_token, noise_pred, noise


    def ddim_pred(self, gt_feat):
        # Sample noise to add to the images
        noise = torch.randn(gt_feat.shape, device=gt_feat.device) #.to(gt_feat.device)
        bs = gt_feat.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (bs,), device=gt_feat.device).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_images = self.scheduler.add_noise(gt_feat, noise, timesteps)
        noise_pred = self.model(noisy_images, timesteps)
        # loss = F.mse_loss(noise_pred, noise)
        return noise_pred, noise


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


# if __name__ == "__main__":
#     decoder = MaskDecoder()

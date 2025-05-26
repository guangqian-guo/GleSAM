# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import Any, Dict, List, Tuple

from functools import partial

from ..modeling import ImageEncoderViT, PromptEncoder, TwoWayTransformer
from collections import OrderedDict
from typing import List, Tuple, Type

from ..modeling.common import LayerNorm2d
from ..modeling.components import *
from ..modeling.diffkd_modules import DiffusionModel, NoiseAdapter, AutoEncoder, DDIMPipeline
from ..modeling.scheduling_ddim import DDIMScheduler

# # added by guo
# from .modeling import MaskDecoderDiff
# from .modeling import MaskDecoderDiffBase

def build_sam_vit_h(opt, checkpoint=None, train=False):
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
        train=train,
        opt=opt,
    )

def build_sam_vit_l(opt, checkpoint=None, train=False):
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
        train=train,
        opt=opt,
    )


def build_sam_vit_b(opt, checkpoint=None, train=False):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
        train=train,
        opt=opt,
    )


sam_model_registry = {
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
}


def _build_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
    train=False,
    opt=None,
):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = Sam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            opt=opt,
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            vit_dim = encoder_embed_dim,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )

    if checkpoint is not None:
        if train == True:
            with open(checkpoint, "rb") as f:
                state_dict = torch.load(f, map_location='cpu')
        
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:]  # remove 'module.' prefix
                    new_state_dict[name] = v

            info = sam.load_state_dict(new_state_dict, strict=False) 
            print(info)
        
        else:
            with open(checkpoint, "rb") as f:
                state_dict = torch.load(f, map_location='cpu')
    
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    if k.startswith('module'):
                        name = k[7:]  # remove 'module.' prefix
                        new_state_dict[name] = v
                    
                    else: 
                        new_state_dict[k] = v
            
            info = sam.load_state_dict(new_state_dict, strict=False)
            print(info)
          
    return sam


class MaskDecoder(nn.Module):
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



class Sam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    def forward(
        self,
        opt,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool,        
        return_logits: bool = False,
        robust_token_only: bool = False,
    ) -> List[Dict[str, torch.Tensor]]:
    
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_input (list(dict)): A list over input images, each a
            dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'image': The image as a torch tensor in 3xHxW format,
                already transformed for input to the model.
              'original_size': (tuple(int, int)) The original size of
                the image before transformation, as (H, W).
              'point_coords': (torch.Tensor) Batched point prompts for
                this image, with shape BxNx2. Already transformed to the
                input frame of the model.
              'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxN.
              'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
                Already transformed to the input frame of the model.
              'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form Bx1xHxW.
          multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.

        Returns:
          (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
              'masks': (torch.Tensor) Batched binary mask predictions,
                with shape BxCxHxW, where B is the number of input prompts,
                C is determined by multimask_output, and (H, W) is the
                original size of the image.
              'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape BxC.
              'low_res_logits': (torch.Tensor) Low resolution logits with
                shape BxCxHxW, where H=W=256. Can be passed as mask input
                to subsequent iterations of prediction.
        """

        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        with torch.no_grad():
            image_embeddings, encoder_features = self.image_encoder(input_images)
        encoder_features = encoder_features[0] 

        outputs = []
        degraded_index = int(0.5 * len(batched_input)) 

        for i, (image_record, curr_embedding, curr_encoder_features) in enumerate(zip(batched_input, image_embeddings, encoder_features)):
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
        
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=points,
                    boxes=image_record.get("boxes", None),
                    masks=image_record.get("mask_inputs", None),
                )            

            clear = True if i < degraded_index else False
            low_res_masks, iou_predictions, robust_embeddings, robust_token, noise_pred, noise = self.mask_decoder( 
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
                encoder_features=curr_encoder_features.unsqueeze(0).unsqueeze(0),
                robust_token_only=robust_token_only,
                clear=clear
            )

    
            masks = self.postprocess_masks(
                low_res_masks,
                input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"],
            )
            
            if i < degraded_index:
              masks = masks > self.mask_threshold 

            outputs.append(
                {
                    "masks": masks,
                    "iou_predictions": iou_predictions,
                    "low_res_logits": low_res_masks,
                    "robust_embeddings": robust_embeddings,
                    "robust_token": robust_token,
                    "noise_pred": noise_pred,
                    "noise": noise
                }
            )

        return outputs
      
    @torch.no_grad()
    def predict(
        self,
        opt,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool,        
        return_logits: bool = False,
        robust_token_only: bool = False,
    ) -> List[Dict[str, torch.Tensor]]:

        input_images = self.preprocess(batched_input[0]['image'])
        image_embeddings, encoder_features = self.image_encoder(input_images)
        encoder_features = encoder_features[0]  # supplementary feature
        
        outputs = []

        for i, (image_record, curr_embedding, curr_encoder_features) in enumerate(zip(batched_input, image_embeddings, encoder_features)):
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=points,
                    boxes=image_record.get("boxes", None),
                    masks=image_record.get("mask_inputs", None),
                )            

            low_res_masks, iou_predictions, robust_embeddings, robust_token = self.mask_decoder( 
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
                encoder_features=curr_encoder_features.unsqueeze(0).unsqueeze(0),
                robust_token_only=robust_token_only,
                clear=False
            )            
            
            masks = self.postprocess_masks(
                low_res_masks,
                input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"],
            )

            masks = masks > self.mask_threshold                 
            
            outputs.append(
                {
                    "masks": masks,
                    "iou_predictions": iou_predictions,
                    "low_res_logits": low_res_masks,
                    "robust_embeddings": robust_embeddings,
                    "robust_token": robust_token
                }
            )

        return outputs
      
    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x




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


# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

from .modeling import ImageEncoderViT,  MaskDecoder, PromptEncoder, Sam, SamJoint, Sam_Robust, MaskDecoder_Robust, TwoWayTransformer
from collections import OrderedDict


from segment_anything.utils.transforms import ResizeLongestSide 


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
    # sam_encoder = SAM_Encoder(
    #     encoder_embed_dim=1024,
    #     encoder_depth=24,
    #     encoder_num_heads=16,
    #     encoder_global_attn_indexes=[5, 11, 17, 23],
    #     checkpoint=checkpoint
    # )
    # return sam_encoder
    return _build_sam(
            encoder_embed_dim=1024,
            encoder_depth=24,
            encoder_num_heads=16,
            encoder_global_attn_indexes=[5, 11, 17, 23],
            checkpoint=checkpoint,
        ) 


def build_sam_vit_l_joint(opt, checkpoint=None, train=False):
    # sam_encoder = SAM_Encoder(
    #     encoder_embed_dim=1024,
    #     encoder_depth=24,
    #     encoder_num_heads=16,
    #     encoder_global_attn_indexes=[5, 11, 17, 23],
    #     checkpoint=checkpoint
    # )
    # return sam_encoder
    return _build_sam_joint(
            encoder_embed_dim=1024,
            encoder_depth=24,
            encoder_num_heads=16,
            encoder_global_attn_indexes=[5, 11, 17, 23],
            checkpoint=checkpoint,
        ) 


def build_sam_vit_b_joint(opt, checkpoint=None, train=False):
    return _build_sam_joint(
            encoder_embed_dim=768,
            encoder_depth=12,
            encoder_num_heads=12,
            encoder_global_attn_indexes=[2, 5, 8, 11],
            checkpoint=checkpoint,
        ) 


def build_sam_vit_h_joint(opt, checkpoint=None, train=False):
    return _build_sam_joint(
            encoder_embed_dim=1280,
            encoder_depth=32,
            encoder_num_heads=16,
            encoder_global_attn_indexes=[7, 15, 23, 31],
            checkpoint=checkpoint,
        ) 


def build_sam_vit_l_robust(opt, checkpoint=None, train=False):
    return _build_sam_robust(
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


def build_sam_encoder(checkpoint):
    sam_encoder = SAM_Encoder(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint
    )
    return sam_encoder



sam_model_registry = {
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_l_joint": build_sam_vit_l_joint,
    "vit_b_joint": build_sam_vit_b_joint,
    "vit_h_joint": build_sam_vit_h_joint,
    "vit_l_robust": build_sam_vit_l_robust,
    "vit_b": build_sam_vit_b,
    "encoder": build_sam_encoder,
}




def _build_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
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
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )       

    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f, map_location='cpu')
        new_state_dict = {}
        for k, v in state_dict:
            if "module." in k:
                new_k = k.replace('module.', '')
            else:
                new_k = k
            new_state_dict[new_k] = v
        
        info = sam.load_state_dict(new_state_dict, strict=False)
        print(info)
    return sam

def _build_sam_joint(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = SamJoint(
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
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )       

    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f, map_location='cpu')
        new_state = {}
        for k, v in state_dict.items():
            if "module" in k:
                k = k.replace("module.", "")
            new_state[k] = v
        info = sam.load_state_dict(new_state, strict=False)
        print(info)
          
    return sam


def _build_sam_robust(
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
    sam = Sam_Robust(
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
        mask_decoder=MaskDecoder_Robust(
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






class SAM_Encoder(nn.Module):
    def __init__(self, 
        checkpoint,
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23]
    ):
        super().__init__()
        prompt_embed_dim = 256
        image_size = 1024
        vit_patch_size = 16
        self.pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1).cuda()
        self.pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1).cuda()


        self.image_encoder=ImageEncoderViT(
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
        )

        self._init_weight(checkpoint=checkpoint)

        self.transform = ResizeLongestSide(self.image_encoder.img_size)

    def _init_weight(self, checkpoint):
        if checkpoint is not None:
            with open(checkpoint, "rb") as f:
                state_dict = torch.load(f, map_location='cpu')
            info = self.load_state_dict(state_dict, strict=False)
        print(info)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std   # -1 1, 

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def forward(self, batched_input):  
        batched_input = batched_input * 255.0   # chw, 0-255, rgb

        # resize to 1024
        batched_input = self.transform.apply_image_torch(batched_input)

        input_images = torch.stack([self.preprocess(x) for x in batched_input], dim=0)
        
        with torch.no_grad():
            image_embeddings, encoder_features = self.image_encoder(input_images)

        return image_embeddings
    
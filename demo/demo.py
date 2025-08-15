import argparse 
import torch 
import os
from tqdm import tqdm
import numpy as np
from accelerate.utils import set_seed
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from transformers import AutoTokenizer, CLIPTextModel
from torchvision import transforms
import sam_utils.misc as misc
from models.sd_unifined_model_CRE_addlora import SDUniModel
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide 
import cv2
import matplotlib.cm as cm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

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
            output_path = args.output_path
            os.makedirs(output_path, exist_ok=True)
        self.model = SDUniModel(args, accelerator).to(accelerator.device)
        self.max_grad_norm = args.max_grad_norm
        self.step = 0
        
        # init tokenizer and text encoder =========================
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_id, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
                args.model_id, subfolder="text_encoder"
            ).to(accelerator.device)
        self.text_encoder.requires_grad_(False)

        self.uncond_embedding = self.encode_prompt([""])

        # add mask decoder dataloader, optim, and scheduler =======================
        # ============= init SAM ======================================
        self.sam = sam_model_registry['vit_l_joint'](opt=None, checkpoint=args.sam_ckpt).to(accelerator.device)
        self.sam_transform = ResizeLongestSide(self.sam.image_encoder.img_size)
        # ======================================================================
        self.sam = accelerator.prepare(self.sam)

        self.accelerator = accelerator

        if args.checkpoint_path is not None:
            self.load(args.checkpoint_path)

        self.previous_time = None 
        self.no_save = args.no_save

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

    def load(self, checkpoint_path):
        # this is used for non-fsdp models.
        self.step = int(checkpoint_path.replace("/", "").split("_")[-1])   # reload step
        print(self.accelerator.load_state(checkpoint_path, strict=False))
        self.accelerator.print(f"Loaded checkpoint from {checkpoint_path}")

    def inference(self, img, point_prompt, point_label):
        """
        img: RGb [0, 1]  CHW
        """ 
        self.val_load(args.ckpt_path_for_val)
        self.sam.eval()
        accelerator = self.accelerator

        img = img.to(self.accelerator.device).float()
        point_label = point_label.to(self.accelerator.device)
        point_prompt = point_prompt.to(self.accelerator.device)
        transformed_img = self.sam_transform.apply_image_torch(img * 255.0)   # [0, 1] -> [0, 255]; RGB; CHW;

        batched_input = []
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

        prompt = ['']
        text_embedding = self.encode_prompt(prompt)
        uncond_embedding = self.uncond_embedding.repeat(len(text_embedding), 1, 1)

        generator_loss_dict, generator_log_dict = self.model(
            lq, lq_feat, text_embedding, uncond_embedding,
            visual=False,
            compute_generator_gradient=False,
            real_train_dict=real_train_dict,
            generator_turn=True,
            guidance_turn=False
        )

        pred_feat = generator_log_dict['guidance_data_dict']["image"] / args.feat_weight

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
    parser.add_argument('--sam_ckpt', type=str, default='pretrained-weights/sam_vit_l_0b3195.pth') 
    parser.add_argument('--feat_weight', type=int, default=1)  # added by guo  给特征加权
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

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    trainer = Trainer(args)
   
    img = torch.randn(1, 3, 512, 512)  # dummy image
    mask = torch.randn(1, 1, 512, 512)  # dummy mask
    point_prompt = torch.tensor([[[100, 100], [200, 200]]])  # dummy points
    point_label = torch.tensor([[1, 0]])

    Trainer.inference(img, point_prompt, point_label)
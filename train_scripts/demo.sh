export WEIGHT=$1
export OUTPUT_PATH=$2
export CHECKPOINT_PATH=$3

accelerate launch --main_process_port 30000 demo.py  \
    --generator_lr 1e-4  \
    --guidance_lr 1e-4 \
    --feat_weight 3 \
    --train_iters 100000 \
    --output_path  work-dir/ \
    --max_checkpoint 2 \
    --batch_size 1 \
    --log_iters 1000 \
    --seed 10 \
    --real_guidance_scale 8 \
    --fake_guidance_scale 1.0 \
    --max_grad_norm 10.0 \
    --model_id "/home/ps/Guo_copy/Guo_checkpoint/sd-1.5" \
    --dataset_cfg "options/lrseg_add_mask_multi_dataset_ms.yaml" \
    --log_loss \
    --dfake_gen_update_ratio 5 \
    --max_step_percent 0.98 \
    --spatial_loss \
    --cls_on_clean_image \
    --gen_cls_loss \
    --percep_weight 2 \
    --gen_cls_loss_weight 5e-3 \
    --guidance_cls_loss_weight 1e-2 \
    --diffusion_gan \
    --use_x0 \
    --diffusion_gan_max_timestep 1000 \
    --denoising_timestep 1000 \
    --conditioning_timestep 1 \
    --backward_simulation \
    --lora_rank 8 \
    --lora_alpha 8 \
    --ckpt_path_for_val trained_models/sd1.5/pytorch_lora_weights.safetensors

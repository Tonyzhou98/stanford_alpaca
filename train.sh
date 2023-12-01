#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --time=20:00:00
#SBATCH --mem-per-cpu=20g
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16


torchrun --master_port=29600 --nproc_per_node=4 train.py \
    --model_name_or_path /scratch/zt1/project/aiwie-prj/user/tonyzhou/llama-2-7b-hf/ \
    --data_path ./alpaca_data.json \
    --bf16 True \
    --output_dir /scratch/zt1/project/aiwie-prj/user/tonyzhou/checkpoint/ \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap offload" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True
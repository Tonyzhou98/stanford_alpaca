#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --time=20:00:00
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4


torchrun --master_port=29500 --nproc_per_node=4 train.py \
    --model_name_or_path /afs/shell.umd.edu/project/aiwie-prj/user/tonyzhou/llama-2-7b-hf/ \
    --data_path ./alpaca_data.json \
    --bf16 True \
    --output_dir /afs/shell.umd.edu/project/aiwie-prj/user/tonyzhou/checkpoint/ \
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
    --deepspeed "./configs/default_offload_opt_param.json" \
    --tf32 True
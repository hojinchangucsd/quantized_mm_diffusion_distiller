#!/bin/bash
cd /home/mmorafah@AD.UCSD.EDU/Progressive_Distillation/diffusion_distiller

nohup \
    python ./q_sample_fid.py \
    --start_index=0  \
    --num_samples=30000 \
    --out_folder=./data/celeba-u_quant-model-out \
    --device=cpu \
        > /dev/null &

ps -aux | grep "q_sample_fid"
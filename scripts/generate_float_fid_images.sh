#!/bin/bash
cd /home/mmorafah@AD.UCSD.EDU/Progressive_Distillation/diffusion_distiller

for si in 0 5000 10000; do
    nohup \
        python ./sample_fid.py \
        --start_index="$si"  \
        --num_samples=5000 \
        --out_folder=./data/celeba-u_float-model-out \
        --device=cuda:0 \
            > /dev/null &
done

for si in 15000 20000 25000; do
    nohup \
        python ./sample_fid.py \
        --start_index="$si"  \
        --num_samples=5000 \
        --out_folder=./data/celeba-u_float-model-out \
        --device=cuda:1 \
            > /dev/null &
done

ps -aux | grep "sample_fid"
#!/bin/bash
cd /home/mmorafah@AD.UCSD.EDU/Progressive_Distillation/diffusion_distiller/

path1=./data/fid_stats/celeba-u_float-fid-stats.npz
path2=./data/fid_stats/celeba_hq_256_fid-stats.npz

python -m pytorch_fid --device cuda:1 "$path1" "$path2"
#!/bin/bash
cd /home/mmorafah@AD.UCSD.EDU/Progressive_Distillation/diffusion_distiller/

dp=./data/fid_stats/celeba_hq_256_fid-stats.npz
sqp=./data/fid_stats/celeba-u_static-quant-fid-stats.npz
qatp=./data/fid_stats/celeba-u_qat-fid-stats.npz
fp=./data/fid_stats/celeba-u_float-fid-stats.npz

python -m pytorch_fid --device cuda:1 "$dp" "$sqp"
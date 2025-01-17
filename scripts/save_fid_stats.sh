#!/bin/bash
cd /home/mmorafah@AD.UCSD.EDU/Progressive_Distillation/diffusion_distiller

imgs_path=./data/celeba-u_qat-quant-model-out/
stats_path=./data/fid_stats/celeba-u_qat-fid-stats.npz

python -m pytorch_fid --device cuda:1 --save-stats $imgs_path $stats_path
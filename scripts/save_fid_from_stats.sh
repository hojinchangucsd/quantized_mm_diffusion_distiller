#!/bin/bash
cd /home/mmorafah@AD.UCSD.EDU/Progressive_Distillation/diffusion_distiller

imgs_path=./data/celeba_hq_256/celeb/
stats_path=./data/fid_stats/celeba_hq_256_fid-stats.npz

python -m pytorch_fid --device cuda:1 --save-stats $imgs_path $stats_path
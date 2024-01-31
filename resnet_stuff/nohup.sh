#!/bin/bash
cd /home/mmorafah@AD.UCSD.EDU/Progressive_Distillation/diffusion_distiller
eval "$(conda shell.bash hook)"
conda activate prog_dist

nohup python ./resnet18_stuff/cifar100_q.py > ./resnet18_stuff/nohup.out &
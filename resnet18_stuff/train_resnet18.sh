#!/bin/bash
cd /home/mmorafah@AD.UCSD.EDU/Progressive_Distillation/diffusion_distiller
eval "$(conda shell.bash hook)"
conda activate FLCP+fb

nohup python ./resnet18_stuff/train_resnet18.py > ./resnet18_stuff/nohup_train_resnet18.out &
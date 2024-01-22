from ml_collections import ConfigDict as Config
import torch

def get_float_config(): 
    fconfig = Config()
    fconfig.device = torch.device("cuda:1")
    fconfig.lr=0.01
    fconfig.momentum=0.9
    fconfig.betas=(0.9, 0.999)
    fconfig.weight_decay=6e-2
    fconfig.cooldown_epochs=10
    fconfig.warmup_epochs=10
    fconfig.warmup_lr=0.000001
    fconfig.smoothing=0.1
    fconfig.ep=300
    fconfig.tr_bs=128
    fconfig.loss_func=torch.nn.CrossEntropyLoss().to(device=fconfig.device)
    fconfig.tb_name='float'
    fconfig.ckpt_path='../checkpoints/CIFAR100/ResNet18_ckpt.pt'
    return fconfig

def get_quant_config(): 
    qconfig = Config()
    qconfig.device = torch.device("cuda:1")
    qconfig.lr=0.01
    qconfig.momentum=0.9
    qconfig.betas=(0.9, 0.999)
    qconfig.weight_decay=6e-2
    qconfig.cooldown_epochs=10
    qconfig.warmup_epochs=10
    qconfig.warmup_lr=0.000001
    qconfig.smoothing=0.1
    qconfig.ep=300
    qconfig.tr_bs=128
    qconfig.loss_func=torch.nn.CrossEntropyLoss().to(device=qconfig.device)
    qconfig.tb_name='qat'
    qconfig.ckpt_path='../checkpoints/CIFAR100/qat_ResNet18_ckpt.pt'
    return qconfig

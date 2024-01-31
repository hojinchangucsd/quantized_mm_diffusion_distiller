from ml_collections import ConfigDict as Config
import torch

def get_float_config(): 
    fconfig = Config()
    fconfig.device = "cuda:0"
    fconfig.lr=0.001
    fconfig.momentum=0.9
    fconfig.weight_decay=1e-4
    fconfig.ep=150
    fconfig.tr_bs=128
    fconfig.loss_func=torch.nn.CrossEntropyLoss().to(fconfig.device)
    fconfig.optimizer='Adam'
    fconfig.tb_name='resnet32_float'
    fconfig.ckpt_path='../checkpoints/CIFAR100/ResNet32_ckpt.pt'
    return fconfig

def get_quant_config(): 
    qconfig = get_float_config()
    qconfig.device = "cuda:0"
    qconfig.loss_func=torch.nn.CrossEntropyLoss().to(qconfig.device)
    qconfig.tb_name+='qat'
    qconfig.ckpt_path='../checkpoints/CIFAR100/qat_ResNet32_ckpt.pt'
    return qconfig
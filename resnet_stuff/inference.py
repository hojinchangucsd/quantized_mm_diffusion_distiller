import os
os.chdir('/home/mmorafah@AD.UCSD.EDU/Progressive_Distillation/diffusion_distiller/resnet_stuff')

import sys
sys.path.append('../')

import torch
from torch.ao.quantization import (
    get_default_qconfig, get_default_qat_qconfig
)
from torch.ao.quantization.quantize_fx import (
    prepare_fx, convert_fx, prepare_qat_fx
)
from torch.ao.quantization import QConfigMapping
from torch.ao.quantization.fx.custom_config import PrepareCustomConfig

from resnet_stuff.resnet_wrapper import resnet_wrapper
import resnet_stuff.resnet_config as resnet_config

import warnings
warnings.filterwarnings(
    action='ignore',
    category=DeprecationWarning,
    module=r'.*'
)
warnings.filterwarnings(
    action='default',
    module=r'torch.ao.quantization'
)
warnings.filterwarnings(
    action='ignore',
    category=UserWarning
)


rw = resnet_wrapper(resnet_config.get_float_config())
rw.load_ckpt()
print(f'ResNet32 -- F model CIFAR100 Acc: {rw.get_acc():.4f}')

rw = resnet_wrapper(resnet_config.get_quant_config())

jit_path = '../checkpoints/CIFAR100/qat_ResNet32_jit.pth'

if os.path.exists(jit_path): 
    qmodel = torch.jit.load(jit_path)
else: 
    backend = "x86"
    device = torch.device('cuda:0')

    qconfig = get_default_qat_qconfig(backend)
    torch.backends.quantized.engine = backend
    qconfig_mapping = QConfigMapping() \
        .set_global(qconfig)
    prepare_fx_custom_map = PrepareCustomConfig()
    example_inputs = (next(iter(rw.train_dl))[0])

    fmodel = rw.model.train().to(device)
    pmodel = prepare_qat_fx(fmodel, qconfig_mapping, example_inputs,
                            prepare_fx_custom_map)
    rw.model = pmodel

    rw.load_ckpt()

    device = "cpu"
    rw.model.eval().to(device)
    qmodel = convert_fx(rw.model)
    qmodel.to(device)
    
    #qmodel = torch.jit.script(qmodel)
    #torch.jit.save(qmodel, jit_path)

rw.model = qmodel
rw.config.device = device
qtest_acc = rw.get_acc()
print(f'ResNet32 -- Q model CIFAR100 Acc: {qtest_acc:.4f}')
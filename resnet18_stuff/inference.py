import os
os.chdir('/home/mmorafah@AD.UCSD.EDU/Progressive_Distillation/diffusion_distiller/resnet18_stuff')

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

from resnet18_wrapper import resnet18_wrapper
import resnet18_config

rw = resnet18_wrapper(resnet18_config.get_quant_config())

jit_path = '../checkpoints/CIFAR100/qat_ResNet18_jit.pth'

if os.path.exists(jit_path): 
    qmodel = torch.jit.load(jit_path)
else: 
    backend = "x86"
    device = torch.device('cuda:1')

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
    
    qmodel = torch.jit.script(qmodel)
    torch.jit.save(qmodel, '../checkpoints/CIFAR100/qat_ResNet18_jit.pth')

rw.model = qmodel
qtest_acc = rw.get_acc('cpu')
print(f'ResNet18 -- Q model CIFAR100 Acc: {qtest_acc:.4f}')
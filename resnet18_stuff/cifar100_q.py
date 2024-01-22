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
from q_helpers import *

from resnet18_wrapper import resnet18_wrapper
import resnet18_config


QAT = True
LOAD_CKPT = False


resnetw = resnet18_wrapper(resnet18_config.get_quant_config())


if LOAD_CKPT: 
    resnetw.load_ckpt()
    
device = torch.device("cuda:1")
backend = "x86"

qconfig = get_default_qconfig(backend) if not QAT else get_default_qat_qconfig(backend)
torch.backends.quantized.engine = backend
qconfig_mapping = QConfigMapping() \
    .set_global(qconfig)
prepare_fx_custom_map = PrepareCustomConfig()
example_inputs = (next(iter(resnetw.train_dl))[0])

if not QAT: 
    fmodel = resnetw.model.eval().to(device)
    pmodel = prepare_fx(fmodel, qconfig_mapping, 
                        example_inputs, prepare_fx_custom_map)
    resnetw.model = pmodel
    resnetw.calibrate(100)

else: 
    fmodel = resnetw.model.train().to(device)
    pmodel = prepare_qat_fx(fmodel, qconfig_mapping, example_inputs,
                            prepare_fx_custom_map)
    resnetw.model = pmodel
    resnetw.train_model()

resnetw.model = fmodel
ftest_acc = resnetw.get_acc(device)
print(f'ResNet18 -- Full model CIFAR100 Acc: {ftest_acc:.4f}')

device = "cpu"
pmodel.eval().to(device)
qmodel = convert_fx(pmodel)
qmodel.to(device)

resnetw.model = qmodel
qtest_acc = resnetw.get_acc(device)
print(f'ResNet18 -- Q model CIFAR100 Acc: {qtest_acc:.4f}')

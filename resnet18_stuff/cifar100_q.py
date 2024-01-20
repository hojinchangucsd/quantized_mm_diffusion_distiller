import os
os.chdir('/home/mmorafah@AD.UCSD.EDU/Progressive_Distillation/diffusion_distiller/resnet18_stuff')

import sys
sys.path.append('../')

import torch
from torch.ao.quantization import (
    get_default_qconfig
)
from torch.ao.quantization.quantize_fx import (
    prepare_fx, convert_fx
)
from torch.ao.quantization import QConfigMapping
from torch.ao.quantization.fx.custom_config import PrepareCustomConfig
from q_helpers import *
from torch.utils.data import DataLoader
from tqdm import tqdm

from resnet18_stuff.resnet import ResNet, BasicBlock
from torchvision.datasets import CIFAR100
from torchvision import transforms
#from torchvision.models import resnet18, ResNet18_Weights

transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
])

train_ds = CIFAR100('../data', train=True, transform=transform_train, download=True)
test_ds = CIFAR100('../data', train=False, transform=transform_test, download=True)

train_dl = DataLoader(dataset=train_ds, batch_size=128, shuffle=True, drop_last=False, \
                     num_workers=0, pin_memory=True)
test_dl = DataLoader(dataset=test_ds, batch_size=128, shuffle=False, drop_last=False, \
                     num_workers=0, pin_memory=True)

num_classes = len(train_ds.classes)

fmodel = ResNet(BasicBlock, [2,2,2,2], num_classes=num_classes)
pt_weights = torch.load('../checkpoints/CIFAR100/ResNet18_ckpt.pt')
fmodel.load_state_dict(pt_weights)
#fmodel = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1,num_classes=num_classes)
device = torch.device("cuda:1")
backend = "x86"

qconfig = get_default_qconfig(backend)
torch.backends.quantized.engine = backend
qconfig_mapping = QConfigMapping() \
    .set_global(qconfig)
prepare_fx_custom_map = PrepareCustomConfig()
example_inputs = (test_ds.__getitem__(0)[0].unsqueeze(dim=0))

fmodel = fmodel.eval().to(device)

pmodel = prepare_fx(fmodel, qconfig_mapping, 
                    example_inputs, prepare_fx_custom_map)

def calibrate(it=100): 
    with torch.no_grad():
        for i in tqdm(range(it), desc=f'Calibration'): 
            pmodel(train_ds.__getitem__(i)[0].to(device).unsqueeze(dim=0))

calibrate()

def get_acc(m): 
    num_correct = 0
    with torch.no_grad(): 
        for inputs, targets in tqdm(test_dl,desc=f'Testing CIFAR100'): 
            inputs, targets = \
                inputs.to(device=device,non_blocking=True), \
                targets.to(device=device,non_blocking=True)
            out = m(inputs)
            num_correct += targets.eq(out.argmax(dim=1)).sum()
    return num_correct / len(test_dl.dataset)

ftest_acc = get_acc(fmodel)
print(f'ResNet18 -- Full model CIFAR100 Acc: {ftest_acc:.4f}')

device = "cpu"
pmodel.eval().to(device)
qmodel = convert_fx(pmodel)
qmodel.to(device)

qtest_acc = get_acc(qmodel)
print(f'ResNet18 -- Q model CIFAR100 Acc: {qtest_acc:.4f}')
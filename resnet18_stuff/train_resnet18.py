import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
from re import findall
import os
os.chdir('/home/mmorafah@AD.UCSD.EDU/Progressive_Distillation/diffusion_distiller/resnet18_stuff')

from resnet import ResNet, BasicBlock
from torchvision.datasets import CIFAR100
from torch.utils.tensorboard import SummaryWriter

from torchvision import transforms

device = torch.device("cuda:1")
lr=0.01
momentum=0.9
betas=(0.9, 0.999)
weight_decay=6e-2
cooldown_epochs=10
warmup_epochs=10
warmup_lr=0.000001
smoothing=0.1
ep=300
tr_bs=128
loss_func=torch.nn.CrossEntropyLoss().cuda(device=device)

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

curr_time = findall(r'^(.*):(?:[+-]?([0-9]*[.])?[0-9]+)$', str(datetime.now()))[0][0].replace(' ','_')
tb_path = os.path.join("../results/CIFAR100", curr_time)

os.makedirs(tb_path,exist_ok=True)
for f in os.listdir(tb_path): 
    os.remove(os.path.join(tb_path,f))
tb = SummaryWriter(tb_path)

backend = "x86"

num_classes = len(train_ds.classes)

model = ResNet(BasicBlock, [2,2,2,2], num_classes=num_classes)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)#, betas=betas, weight_decay=weight_decay)

def train_1epoch():
    losses = []
    for images, labels in train_dl:
        images, labels = \
            images.cuda(device=device,non_blocking=True), \
            labels.cuda(device=device,non_blocking=True)
        model.zero_grad()
        log_probs = model(images)
        loss = loss_func(log_probs, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses

def get_acc(m): 
    num_correct = 0
    with torch.no_grad(): 
        for inputs, targets in test_dl: 
            inputs, targets = \
                inputs.cuda(device=device,non_blocking=True), \
                targets.cuda(device=device,non_blocking=True)
            out = m(inputs)
            num_correct += targets.eq(out.argmax(dim=1)).sum()
    return num_correct / len(test_dl.dataset)

ckpt_path = '../checkpoints/CIFAR100'
os.makedirs(ckpt_path,exist_ok=True)

model = model.train().to(device)
for e in range(ep):
    epoch_losses = train_1epoch()
    for i,l in enumerate(epoch_losses): 
        tb.add_scalar("Loss", l, i+e*len(epoch_losses))
    test_acc = get_acc(model)
    tb.add_scalar("Test Acc", test_acc, e)
    torch.save(model.state_dict(), os.path.join(ckpt_path,'ResNet18_ckpt.pt'))
tb.close()

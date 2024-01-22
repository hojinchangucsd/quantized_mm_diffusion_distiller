import torch
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
from resnet18_config import get_float_config

class resnet18_wrapper(object): 
    def __init__(self, config): 
        self.config = config

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

        self.train_dl = DataLoader(dataset=train_ds, batch_size=128, shuffle=True, drop_last=False, \
                            num_workers=0, pin_memory=True)
        self.test_dl = DataLoader(dataset=test_ds, batch_size=128, shuffle=False, drop_last=False, \
                            num_workers=0, pin_memory=True)

        self.num_classes = len(train_ds.classes)

        curr_time = findall(r'^(.*):(?:[+-]?([0-9]*[.])?[0-9]+)$', str(datetime.now()))[0][0].replace(' ','_')
        tb_path = os.path.join("../results/CIFAR100", self.config.tb_name + '_' + curr_time)

        os.makedirs(tb_path,exist_ok=True)
        for f in os.listdir(tb_path): 
            os.remove(os.path.join(tb_path,f))
        self.tb = SummaryWriter(tb_path)

        self.model = ResNet(BasicBlock, [2,2,2,2], num_classes=self.num_classes)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr, 
                                           betas=self.config.betas, weight_decay=self.config.weight_decay)
    
    def load_ckpt(self): 
        pt_weights = torch.load(self.config.ckpt_path)
        self.model.load_state_dict(pt_weights)

    def train_1epoch(self):
        losses = []
        for images, labels in self.train_dl:
            images, labels = \
                images.cuda(device=self.config.device,non_blocking=True), \
                labels.cuda(device=self.config.device,non_blocking=True)
            self.model.zero_grad()
            log_probs = self.model(images)
            loss = self.config.loss_func(log_probs, labels)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
        return losses

    def get_acc(self, device): 
        self.model = self.model.eval().to(device)
        num_correct = 0
        with torch.no_grad(): 
            for inputs, targets in self.test_dl: 
                inputs, targets = \
                    inputs.to(device=device,non_blocking=True), \
                    targets.to(device=device,non_blocking=True)
                out = self.model(inputs)
                num_correct += targets.eq(out.argmax(dim=1)).sum()
        return num_correct / len(self.test_dl.dataset)

    def train_model(self): 
        model = self.model.train().to(self.config.device)
        for e in range(self.config.ep):
            epoch_losses = self.train_1epoch()
            for i,l in enumerate(epoch_losses): 
                self.tb.add_scalar("Loss", l, i+e*len(epoch_losses))
            test_acc = self.get_acc(self.config.device)
            self.tb.add_scalar("Test Acc", test_acc, e)
            torch.save(model.state_dict(), self.config.ckpt_path)
        self.tb.close()

    def calibrate(self, it): 
        with torch.no_grad():
            for i, (inputs,_) in tqdm(enumerate(self.train_dl), desc=f'Calibration'): 
                inputs = inputs.to(self.config.device)
                self.model(inputs)
                if i == it: break


#rw = resnet18_wrapper(get_float_config())
#rw.train_model()
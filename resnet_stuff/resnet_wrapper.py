import torch
from tqdm import tqdm
from datetime import datetime
from re import findall
import os
os.chdir('/home/mmorafah@AD.UCSD.EDU/Progressive_Distillation/diffusion_distiller/resnet_stuff')

from torch.utils.tensorboard import SummaryWriter

from torchvision.datasets import CIFAR100
from torchvision import transforms
from resnet_config import get_float_config


from cifar_resnet import resnet32 


class resnet_wrapper(object): 
    def __init__(self, config): 
        self.config = config
        
        torch.backends.cudnn.benchmark = True

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

        self.train_dl = torch.utils.data.DataLoader(
            CIFAR100(root='../data/cifar100', train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]), download=False),
            batch_size=config.tr_bs, shuffle=True,
            num_workers=1, pin_memory=True)

        self.test_dl = torch.utils.data.DataLoader(
            CIFAR100(root='../data/cifar100', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]), download=False),
            batch_size=128, shuffle=False,
            num_workers=1, pin_memory=True)

        curr_time = findall(r'^(.*):(?:[+-]?([0-9]*[.])?[0-9]+)$', str(datetime.now()))[0][0].replace(' ','_')
        self.tb_path = os.path.join("../results/CIFAR100", self.config.tb_name + '_' + curr_time)
        self.tb = None

        self.num_classes = len(self.train_dl.dataset.classes)

        self.best_acc = -1.0

        self.model = resnet32(self.num_classes, next(iter(self.train_dl))[0])\
                        .to(device=self.config.device)

        if config.optimizer == 'Adam': 
            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                              lr=config.lr, 
                                              weight_decay=config.weight_decay)
    
    def make_tb(self): 
        os.makedirs(self.tb_path,exist_ok=True)
        for f in os.listdir(self.tb_path): 
            os.remove(os.path.join(self.tb_path,f))
        self.tb = SummaryWriter(self.tb_path)

    def load_ckpt(self): 
        ckpt = torch.load(self.config.ckpt_path)
        pt_weights = ckpt['State Dict']
        self.config.ep -= ckpt['Epoch']
        self.best_acc = ckpt['Test Accuracy']
        self.model.load_state_dict(pt_weights)

    def train_1epoch(self):
        losses = []
        for images, labels in self.train_dl:
            images, labels = \
                images.to(device=self.config.device,non_blocking=True), \
                labels.to(device=self.config.device,non_blocking=True)
            
            log_probs = self.model(images)
            loss = self.config.loss_func(log_probs, labels)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            losses.append(loss.item())

        return losses

    def get_acc(self): 
        self.model = self.model.eval()
        num_correct = 0
        with torch.no_grad(): 
            for inputs, targets in self.test_dl: 
                inputs, targets = \
                    inputs.to(device=self.config.device,non_blocking=True), \
                    targets.to(device=self.config.device,non_blocking=True)
                out = self.model(inputs)
                num_correct += targets.eq(out.argmax(dim=1)).sum()
        return num_correct / len(self.test_dl.dataset)

    def train_model(self): 
        self.make_tb()
        model = self.model.train()
        for e in tqdm(range(self.config.ep),desc='Epochs'):
            epoch_losses = self.train_1epoch()
            for i,l in enumerate(epoch_losses): 
                self.tb.add_scalar("Loss", l, i+e*len(epoch_losses))
            test_acc = self.get_acc()
            self.tb.add_scalar("Test Acc", test_acc, e)
            if test_acc > self.best_acc: 
                self.best_acc = test_acc
                torch.save({'State Dict': model.state_dict(),
                            'Epoch': e,
                            'Test Accuracy': test_acc}, 
                            self.config.ckpt_path)
        self.tb.close()

    def calibrate(self, it): 
        with torch.no_grad():
            for i, (inputs,_) in tqdm(enumerate(self.train_dl), desc=f'Calibration'): 
                inputs = inputs.to(self.config.device)
                self.model(inputs)
                if i == it: break

if __name__ == "__main__": 
    rw = resnet_wrapper(get_float_config())
    #rw.train_model()
    rw.load_ckpt()
    print(f'Float test acc: {rw.get_acc():.2%}')

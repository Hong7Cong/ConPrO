from IPython.display import clear_output 
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import torch
from torchvision.utils import make_grid
from torchvision.io import read_image
# from torchvision import models, transforms
# import sys
# import os
from PIL import Image
import torch
# import torch.functional as F
import numpy as np
clear_output()
# import glob
from utils import *
from datetime import date
from torch.optim import lr_scheduler
import torchvision
from constants import mean, std
from VindrMammoLoader import MammoDataset, MammoCompDataset
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision.ops.focal_loss import sigmoid_focal_loss
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

train_mode ='multiclass'
datalne = {'train':100000, 'val':1000, 'test':2000}
if train_mode =='binary':
    image_datasets = {x: MammoCompDataset(phase=x, datalen=datalne[x], mode="binary_contrastive", seed =22) for x in ['train', 'val', 'test']}
elif train_mode =='multiclass':
    image_datasets = {x: MammoCompDataset(phase=x, datalen=datalne[x], mode="multiclass_contrastive", seed =22) for x in ['train', 'val', 'test']}
else:
    print('wrong mode')
batch_size = {'train':16, 'val':8, 'test':1}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size[x], shuffle=True, num_workers=8, pin_memory = True)
              for x in ['train', 'val', 'test']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val',  'test']}
class_names = ['0','1']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device, class_names)

(inputs, _), classes, _ = next(iter(dataloaders['train']))
out = torchvision.utils.make_grid(inputs)
imshow(out, title=[class_names[x] for x in classes], mean=mean, std=std)



class SiameseNetwork101(nn.Module):
    """
    Siamese neural network
    Modified from: https://hackernoon.com/facial-similarity-with-siamese-networks-in-pytorch-9642aa9db2f7
    Siamese ResNet-101 from Pytorch library
    """ 
    def __init__(self):
        super(SiameseNetwork101, self).__init__()
        # note that resnet101 requires 3 input channels, will repeat grayscale image x3
        self.cnn1 = get_feature_extractor(feature_extractor='resnet50', cotrain=False, simclr= "/home/vishc1/hoang/simCLR-clone/runs/Mar21_17-27-27_huypn168/checkpoint_95_21032024.pth.tar" )
        self.cnn1.fc = nn.Sequential(torch.nn.Linear(2048, 1000),
                                torch.nn.ReLU(),
                                torch.nn.Dropout(0.1),
                                torch.nn.Linear(1000, 256))
    
    def forward_once(self, x):
        output = self.cnn1(x)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    Modified from: https://hackernoon.com/facial-similarity-with-siamese-networks-in-pytorch-9642aa9db2f7

    """ 

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # euclidean_distance = torch.nn.functional.pairwise_distance(output1, output2)
        # loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
        #                               (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        
        cosine_distance = torch.nn.functional.cosine_similarity(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(cosine_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - cosine_distance, min=0.0), 2))
        # loss_contrastive =  torch.nn.NLLLoss()(cosine_distance)

        return loss_contrastive
    


# from torch.optim import lr_scheduler
siamese50simclr = SiameseNetwork101().to(device)
momentum = 0
lr = 0.1
# optimizer_ft = optim.SGD([{'params': siamese50simclr.cnn1.conv1.parameters()},
#                         {'params': siamese50simclr.cnn1.layer1.parameters()},
#                         {'params': siamese50simclr.cnn1.layer2.parameters()},
#                         {'params': siamese50simclr.cnn1.layer3.parameters()},
#                         {'params': siamese50simclr.cnn1.layer4.parameters()},
#                         {'params': siamese50simclr.cnn1.fc.parameters(), 'lr':lr*10}], lr=lr, momentum=momentum)
optimizer_ft = optim.SGD(siamese50simclr.parameters(), lr = lr, momentum= momentum)
loss_fn = ContrastiveLoss(margin=2.0)
scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.5)

for param in siamese50simclr.cnn1.parameters():
    param.requires_grad = True
for param in siamese50simclr.cnn1.fc.parameters():
    param.requires_grad = True


    
valaccmax = 0
trainlosslist = []
trainacclist  = []
currloss = 1000000
for e in range(5):
    training_acc = 0
    val_acc = 0
    training_loss = 0.0
    val_loss = 0.0
    for inputs, labels, _ in tqdm(dataloaders['train'], total=len(dataloaders['train'])):
        siamese50simclr.train()
        inputA = inputs[0].to(device)
        inputB = inputs[1].to(device)
        labels = labels.to(device)
        # zero the parameter gradients
        optimizer_ft.zero_grad()

        output1, output2 = siamese50simclr(inputA, inputB)
        # preds = (torch.max(output1, 1)[1] == torch.max(output2, 1)[1])*1
        loss = loss_fn(output1, output2, labels)
        loss.backward()
        optimizer_ft.step()
        training_loss += loss.item()
        # training_acc += torch.sum(preds == labels.data)
    trainlosslist.append(training_loss)
        # trainacclist.append(training_acc)
        
    for inputs, labels, _ in dataloaders['val']:
        siamese50simclr.eval()
        inputA = inputs[0].to(device)
        inputB = inputs[1].to(device)
        labels = labels.to(device)

        with torch.no_grad():
            output1, output2 = siamese50simclr(inputA, inputB)
            # preds = (torch.max(output1, 1)[1] == torch.max(output2, 1)[1])*1
            loss = loss_fn(output1, output2, labels)
            val_loss += loss.item()
            # val_acc += torch.sum(preds == labels.data)
    val_loss= val_loss / dataset_sizes['val']
    if(val_loss < currloss):
        currloss = val_loss
        today = date.today()
        print(f'save new model at {e}')
        if train_mode == 'binary':
            torch.save(siamese50simclr.state_dict(), f'pretrained/best-supcontrastive50-{today}.pt')
        elif train_mode =='multiclass':
            torch.save(siamese50simclr.state_dict(), f'pretrained/best-multicalss-supcontrastive50-{today}.pt')
        else:
            print('wrong mode')

    #scheduler.step()

    print(f"E{e} With LR {optimizer_ft.param_groups[0]['lr']} training acc: ", "traning loss: ", training_loss / dataset_sizes['train'], "val loss: ", val_loss / dataset_sizes['val'])
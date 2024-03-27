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


datalne = {'train':100000, 'val':1000, 'test':2000}
image_datasets = {x: MammoCompDataset( phase=x, datalen=datalne[x], mode="severity_comparison", seed=22) for x in ['train', 'val', 'test']}
batch_size = {'train':16, 'val':8, 'test':1}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size[x], shuffle=True, num_workers=8)
              for x in ['train', 'val', 'test']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val',  'test']}
class_names = ['0','1']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device, class_names)


class SiameseNetwork101(nn.Module):
    """
    Siamese neural network
    Modified from: https://hackernoon.com/facial-similarity-with-siamese-networks-in-pytorch-9642aa9db2f7
    Siamese ResNet-101 from Pytorch library
    """ 
    def __init__(self):
        super(SiameseNetwork101, self).__init__()
        # note that resnet101 requires 3 input channels, will repeat grayscale image x3
        self.cnn1 = get_feature_extractor(feature_extractor='resnet50', cotrain=False)# , simclr='/mnt/c/Users/PCM/Dropbox/pretrained/SimCLR/checkpoint_10_02102023.pth.tar')
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
    
    
class SeverityModel(nn.Module):
    """
    Siamese neural network
    Modified from: https://hackernoon.com/facial-similarity-with-siamese-networks-in-pytorch-9642aa9db2f7
    Siamese ResNet-101 from Pytorch library
    """ 
    def __init__(self, path2pretrained='pretrained/best-supcontrastive50-2024-03-23.pt'):
        super(SeverityModel, self).__init__()
        # note that resnet101 requires 3 input channels, will repeat grayscale image x3
        self.bestsimese50simclr = SiameseNetwork101()
        state_dict = torch.load(path2pretrained)
        self.bestsimese50simclr.load_state_dict(state_dict)
        self.bestsimese50simclr.cnn1.add_module('fc2',
            nn.Sequential(torch.nn.Linear(256, 256),
                          torch.nn.ReLU(),
                        torch.nn.Dropout(0.1),
                        torch.nn.Linear(256, 256)))
    
    def forward_once(self, x):
        output = self.bestsimese50simclr.cnn1.fc2(self.bestsimese50simclr.cnn1(x))
        return output

    def forward(self, input1, input2, refinput):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        refinput = self.bestsimese50simclr.cnn1(refinput)
        return output1, output2, refinput
    
    
class PreferenceComparisonLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    Modified from: https://hackernoon.com/facial-similarity-with-siamese-networks-in-pytorch-9642aa9db2f7

    """ 

    def __init__(self, margin=2.0):
        super(PreferenceComparisonLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label, ref):
        # euclidean_distance = torch.nn.functional.pairwise_distance(output1, output2)
        # loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
        #                               (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        
        cosine_distanceA = torch.nn.functional.cosine_similarity(output1, ref)
        cosine_distanceB = torch.nn.functional.cosine_similarity(output2, ref)
        loss_comparation = torch.nn.NLLLoss()(torch.nn.Sigmoid()(cosine_distanceA - cosine_distanceB), label)
        # loss_comparation = torch.mean((1-label) * torch.pow(cosine_distance, 2) +
        #                               (label) * torch.pow(torch.clamp(self.margin - cosine_distance, min=0.0), 2))

        return loss_comparation
    
    
smodel = SeverityModel(path2pretrained='pretrained/best-supcontrastive50-2024-03-23.pt').to(device)
momentum = 0
lr = 0.001
optimizer_ft = optim.SGD([{'params': smodel.bestsimese50simclr.cnn1.conv1.parameters()},
                        {'params': smodel.bestsimese50simclr.cnn1.layer1.parameters()},
                        {'params': smodel.bestsimese50simclr.cnn1.layer2.parameters()},
                        {'params': smodel.bestsimese50simclr.cnn1.layer3.parameters()},
                        {'params': smodel.bestsimese50simclr.cnn1.layer4.parameters()},
                        {'params': smodel.bestsimese50simclr.cnn1.fc.parameters()},
                        {'params': smodel.bestsimese50simclr.cnn1.fc2.parameters(), 'lr':lr*10}], lr=lr, momentum=momentum)
loss_fn = PreferenceComparisonLoss()
scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.5)

for param in smodel.bestsimese50simclr.cnn1.parameters():
    param.requires_grad = True
for param in smodel.bestsimese50simclr.cnn1.fc2.parameters():
    param.requires_grad = True
    
    
valaccmax = 0
trainlosslist = []
trainacclist  = []


for e in range(5):
    training_acc = 0
    val_acc = 0
    training_loss_test = 0.0
    curr_loss = 100
    for inputs,ref, labels, _ in tqdm(dataloaders['train']):
        smodel.train()
        inputA = inputs[0].to(device)
        inputB = inputs[1].to(device)
        labels = labels.to(device)
        # zero the parameter gradients
        optimizer_ft.zero_grad()
        #ref = [image_datasets['train'].get_ref_images() for i in range(len(inputA))]
        #ref = torch.stack(ref).to(device)
        ref = ref.to(device)
        output1, output2, refimg = smodel(inputA, inputB, ref)
        # preds = (torch.max(output1, 1)[1] == torch.max(output2, 1)[1])*1
        loss = loss_fn(output1, output2, labels, refimg)
        # cosine_distanceA = torch.nn.functional.cosine_similarity(output1, refimg)
        # cosine_distanceB = torch.nn.functional.cosine_similarity(output2, refimg)
        # gamma = torch.nn.Sigmoid()(cosine_distanceA - cosine_distanceB)
        # loss = torch.nn.NLLLoss()(gamma, labels)
        # loss.requires_grad = True
        loss.backward()
        optimizer_ft.step()
        training_loss_test += loss.item()
        # training_acc += torch.sum(preds == labels.data)
        # trainlosslist.append(training_loss_test)
        # trainacclist.append(training_acc)
        
    # for inputs, labels in dataloaders['val']:
    #     smodel.eval()
    #     inputA = inputs[0].to(device)
    #     inputB = inputs[1].to(device)
    #     labels = labels.to(device)

    #     with torch.no_grad():
    #         output1, output2 = siamese50simclr(inputA, inputB)
    #         preds = (torch.max(output1, 1)[1] == torch.max(output2, 1)[1])*1
    #         # loss = loss_fn(inputA, inputB, labels)
    #         val_acc += torch.sum(preds == labels.data)

    if(training_loss_test < curr_loss):
        curr_loss = training_loss_test
        today = date.today()
        torch.save(smodel.state_dict(), f'pretrained/best-smodel50-wofreeze-{today}.pt')

    scheduler.step()

    print(f"E{e} With LR {optimizer_ft.param_groups[0]['lr']}", "traning loss: ", training_loss_test / dataset_sizes['train'])





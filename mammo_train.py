import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import torch
from torchvision.utils import make_grid
from torchvision.io import read_image
import sys
import os
from PIL import Image
import torch
from utils import *
from datetime import date
from torch.optim import lr_scheduler
import torchvision
from VindrMammoLoader import MammoDataset
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


datalne = {'train':100000, 'val':1000, 'test':1000}
image_datasets = {x: MammoDataset(phase=x, datalen=datalne[x], mode="binary_contrastive", seed =22) for x in ['train', 'val', 'test']}
batch_size = {'train':8, 'val':4, 'test':1}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size[x], shuffle=True, num_workers=8, pin_memory = True)
              for x in ['train', 'val', 'test']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val',  'test']}
class_names = ['0','1']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device, class_names)

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
(inputs, _), classes, _ = next(iter(dataloaders['train']))
out = torchvision.utils.make_grid(inputs)
imshow(out, title=[class_names[x] for x in classes], mean=mean, std=std)

# from torch.optim import lr_scheduler
siamese50simclr = SiameseNetwork101().to(device)
momentum = 0
lr = 0.01
optimizer_ft = optim.SGD([{'params': siamese50simclr.cnn1.fc.parameters()}], lr=lr, momentum=momentum)
loss_fn = ContrastiveLoss(margin=2.0)
scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=2, gamma=0.5)

for param in siamese50simclr.cnn1.parameters():
    param.requires_grad = True
for param in siamese50simclr.cnn1.fc.parameters():
    param.requires_grad = True

valaccmax = 0
trainlosslist = []
trainacclist  = []
currloss = 100
for e in range(100):
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

    if(val_loss < currloss):
        currloss = val_loss
        today = date.today()
        torch.save(siamese50simclr.state_dict(), f'./pretrained/best-contrastive50-{today}.pt')

    scheduler.step()

    print(f"E{e} With LR {optimizer_ft.param_groups[0]['lr']} training acc: ", "traning loss: ", training_loss / dataset_sizes['train'], "val loss: ", val_loss / dataset_sizes['val'])

plt.plot(torch.stack(val_loss).cpu()/1000)
# plt.plot(training_loss_test/1000)

bestsimese50simclr = SiameseNetwork101().to(device)
state_dict = torch.load('./pretrained/best-contrastive50.pt')
bestsimese50simclr.load_state_dict(state_dict)

test_acc = 0
test_embeddings = torch.zeros((0, 256))
test_targets = []

for inputs, labels, args in dataloaders['train']:
        bestsimese50simclr.eval()
        inputA = inputs[0].to(device)
        inputB = inputs[1].to(device)
        labels = labels.to(device)

        with torch.no_grad():
            output1, output2 = bestsimese50simclr(inputA, inputB)
            preds = (torch.max(output1, 1)[1] == torch.max(output2, 1)[1])*1
            # loss = loss_fn(inputA, inputB, labels)
            test_acc += torch.sum(preds == labels.data)
            test_targets.append(args[0])
            test_targets.append(args[1])
            test_embeddings  = torch.cat((test_embeddings, output1.detach().cpu().flatten().unsqueeze(0)), axis=0)
            test_embeddings  = torch.cat((test_embeddings, output2.detach().cpu().flatten().unsqueeze(0)), axis=0)
test_acc

test_targets = torch.stack(test_targets).flatten()
test_embeddings = np.array(test_embeddings)
test_targets = np.array(test_targets)

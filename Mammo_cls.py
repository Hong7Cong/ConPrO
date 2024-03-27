import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import torch
from PIL import Image
import torch
import numpy as np
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
import argparse
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score



class SiameseNetwork101(nn.Module):
    """
    Siamese neural network
    Modified from: https://hackernoon.com/facial-similarity-with-siamese-networks-in-pytorch-9642aa9db2f7
    Siamese ResNet-101 from Pytorch library
    """ 
    def __init__(self):
        super(SiameseNetwork101, self).__init__()
        # note that resnet101 requires 3 input channels, will repeat grayscale image x3
        self.cnn1 = get_feature_extractor(feature_extractor='resnet50', cotrain=False )
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
    def __init__(self, path2pretrained='pretrained/best-contrastive50-2024-03-15.pt'):
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


def Focal_loss(class_logits,  labels):
    if class_logits.numel() == 0:
        return class_logits.new_zeros([1])[0]

    N = class_logits.shape[0]
    K = class_logits.shape[1] 

    target = class_logits.new_zeros(N, K)
    target[range(len(labels)), labels] = 1
    loss = sigmoid_focal_loss(class_logits, target, reduction = 'mean')
    return loss 


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pretrain-mode',  
        default='ConPro',
        type=str, 
        choices= ['ConPro', 'SupCon', 'SupCon5', 'SimClr', 'Imagenet'],
        help='training mode'

    )
    args = vars(parser.parse_args())
    return args

def run_one_epoch(pretrain_mode = 'ConPro', trial=0):

    image_datasets = {x: MammoDataset(phase=x,  seed =22) for x in ['train', 'val', 'test']}
    batch_size = {'train':32, 'val':8, 'test':1}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size[x], shuffle=True, num_workers=8, pin_memory = True)
                for x in ['train', 'val', 'test']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val',  'test']}
    class_names = ['1','2','3', '4', '5']

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device, class_names)
    print(dataset_sizes)

    clf_model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
    if not pretrain_mode == 'Imagenet':
        clf_model.load_state_dict(torch.load(f'pretrained/pretrained-resnet50-{pretrain_mode}.pt'), strict=False)
    clf_model.fc = nn.Sequential(torch.nn.Linear(2048, 1000),
                                    torch.nn.ReLU(),
                                    torch.nn.Dropout(0.1),
                                    torch.nn.Linear(1000, 256),
                                    torch.nn.ReLU(),
                                    torch.nn.Dropout(0.1),
                                    torch.nn.Linear(256, len(class_names)))
    
    y= []
    for _, y1 in tqdm(dataloaders['train']):
        y = y+ y1.tolist()
    print(len(y))
    class_weight = compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y)


    siamese50simclr = clf_model.to(device)
    momentum = 0.9
    lr = 0.1
    optimizer_ft = optim.SGD([{'params': siamese50simclr.fc.parameters()}], lr=lr, momentum=momentum)
    # optimizer_ft = optim.SGD([{'params': siamese50simclr.conv1.parameters()},
    #                         {'params': siamese50simclr.layer1.parameters()},
    #                         {'params':  siamese50simclr.layer2.parameters()},
    #                         {'params': siamese50simclr.layer3.parameters()},
    #                         {'params': siamese50simclr.layer4.parameters()},
    #                         {'params': siamese50simclr.fc.parameters(), 'lr':lr*10}], lr=lr, momentum=momentum)
    # class_weights=torch.tensor(class_weight,dtype=torch.float).to(device)
    # loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    loss_fn= Focal_loss
    scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.5)

    for param in siamese50simclr.parameters():
        param.requires_grad = False
    for param in siamese50simclr.fc.parameters():
        param.requires_grad = True

    


    # bestmodel = siamese50simclr
    f1max = 0
    for e in range(20):
        training_acc = 0
        val_acc = 0
        training_loss_test = 0.0

        for inputs, labels in tqdm(dataloaders['train'], total= len(dataloaders['train'])):
            siamese50simclr.train()
            inputs = inputs.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer_ft.zero_grad()

            outputs = siamese50simclr(inputs)
            _, preds = torch.max(outputs, 1)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer_ft.step()
            training_loss_test += loss.item() * inputs.size(0)
            training_acc += torch.sum(preds == labels.data)
        predlist = []
        labelist = []
        for inputs, labels in dataloaders['val']:
            siamese50simclr.eval()
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                outputs = siamese50simclr(inputs)
                _, preds = torch.max(outputs, 1)
                loss = loss_fn(outputs, labels)
            labelist.append(labels.detach().cpu().numpy()*1)
            predlist.append(preds.detach().cpu().numpy())
            val_acc += torch.sum(preds == labels.data)
        labelist = np.concatenate(labelist).ravel()
        predlist = np.concatenate(predlist).ravel()
        f1 = f1_score(predlist, labelist, average ='macro')
        if(f1 >= f1max):
            f1max = f1
            print(f"New best mode at epoch {e}")
            today = date.today()
            torch.save(siamese50simclr.state_dict(), f'pretrained/best-classification-siamese50simclr-{pretrain_mode}-{trial}.pt')
        torch.save(siamese50simclr.state_dict(), f'pretrained/last-classification-siamese50simclr-{pretrain_mode}-{trial}.pt')
        #scheduler.step()

        print(f"E{e} With LR {optimizer_ft.param_groups[0]['lr']} training acc: ", training_acc.detach().cpu().numpy() / dataset_sizes['train'], "Val acc: ", val_acc.detach().cpu().numpy() / dataset_sizes['val'], "traning loss: ", training_loss_test / dataset_sizes['train'], "f1", f1)

    bestmodel = models.resnet50(weights='ResNet50_Weights.DEFAULT')#get_feature_extractor(feature_extractor='resnet50', cotrain=False)#, simclr='/mnt/c/Users/PCM/Documents/GitHub/pseudopapill/SimCLR/runs/Oct29_21-00-13_DESKTOP-404G4HS/checkpoint_95_29102023.pth.tar')
    bestmodel.fc = nn.Sequential(torch.nn.Linear(2048, 1000),
                                    torch.nn.ReLU(),
                                    torch.nn.Dropout(0.1),
                                    torch.nn.Linear(1000, 256),
                                    torch.nn.ReLU(),
                                    torch.nn.Dropout(0.1),
                                    torch.nn.Linear(256, len(class_names)))
    bestmodel.load_state_dict(torch.load(f'pretrained/best-mutilclass-siamese50-{pretrain_mode}-{trial}.pt'))
    bestmodel.to(device)

    test_acc = 0
    predlist = []
    labelist = []
    problist = []
    # test_embeddings = torch.zeros((0, 2048))
    sedis = 0
    # fextractor = torch.nn.Sequential(*(list(clf_model.children())[:-1]))

    for inputs, labels in dataloaders['test']:
        bestmodel.eval()
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = bestmodel(inputs)
            # emb = fextractor(inputs)
            _, preds = torch.max(outputs, 1)
            # loss = loss_fn(outputs, labels)
            sedis = sedis + torch.sum(torch.exp(torch.abs(labels - torch.max(outputs, 1)[1])))
        problist.append(outputs[:,1].detach().cpu().numpy())
        labelist.append(labels.detach().cpu().numpy()*1)
        predlist.append(preds.detach().cpu().numpy())
        # test_embeddings  = torch.cat((test_embeddings, emb.detach().cpu().flatten().unsqueeze(0)), axis=0)
        test_acc += torch.sum(preds == labels.data)

    labelist = np.concatenate(labelist).ravel()
    problist = np.concatenate(problist).ravel()
    predlist = np.concatenate(predlist).ravel()
    return sedis/dataset_sizes['test'], f1_score(labelist, predlist, average='weighted'), f1_score(labelist, predlist, average='macro')
    # print('MAEE', sedis/dataset_sizes['test'])
    # print('weight F1', f1_score(labelist, predlist, average='weighted'))
    # print('macro F1', f1_score(labelist, predlist, average='macro'))
    # print(classification_report(labelist, predlist, digits=3))

def main(args):

    maeelist = []
    wf1list = []
    mf1list = []

    for i in range(30):
        print('Run #',i)
        maee, wf1, mf1 = run_one_epoch(args['pretrain_mode'], i)
        print('MAEE', maee)
        print('weight F1', wf1)
        print('macro F1', mf1)
        maeelist.append(maee)
        wf1list.append(wf1)
        mf1list.append(mf1)
    print(torch.stack(maeelist).cpu().numpy())
    print(np.array(mf1list))
    print(np.array(wf1list))


if __name__ == '__main__':
    args = parse_opt()
    main(args)





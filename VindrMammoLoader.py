import torch
from torch.utils.data import Dataset
import torchvision
import numpy as np
import cv2
from torch import randint, manual_seed, cuda, backends
import sys
import os
import glob
from PIL import Image
from torchvision.transforms import functional as F
import random
from torchvision import datasets, models, transforms
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import pandas as pd


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


class Breast_crop(object):
    def __init__(self, thres1 = 1e7 , thres2 = 1e7):
        self.thres1= thres1
        self.thres2 = thres2
    def __call__(self, img):
        img1 = np.array(img)
        h, w,_ = img1.shape
        x = img1.sum(axis= (0,2))
        y = np.cumsum(x)
        thres= (y<self.thres1).sum()
        j = max(0, thres-500) 
        x = img1.sum(axis =(1,2))
        y =np.cumsum(np.flip(x))
        thres = (y< self.thres2).sum()
        i = max(1, thres-500)
        img = F.crop(img, 0,j,h-i,w-j)
        return img
data_transforms = {
    'train': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((450,200)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter(),]),p=0.3),
        transforms.RandomApply(torch.nn.ModuleList([transforms.GaussianBlur(kernel_size=3),]),p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'val': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((450,200)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'test': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((450,200)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}

def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True





class MammoDataset(Dataset): 
    def __init__(self, 
                data_path = "../VinDr_Mammo/physionet.org/files/vindr-mammo/1.0.0/images_png/",
                metadata = "../VinDr_Mammo/physionet.org/files/vindr-mammo/1.0.0/breast-level_annotations1.csv",
                phase ='train',
                transform=None,
                seed=None):
        self.phase = phase
        self.data_path= data_path
        if(seed):
            seed_everything(seed)

        self.transform = data_transforms[self.phase] if(transform == None) else transform
        data = pd.read_csv(metadata)
        self.data = data.loc[data['split']== phase].reset_index()
        
    def get_score(self, data, index):
        birads= data['breast_birads'].iloc[index]
        score= eval(birads[-1])
        return score
    def get_path(self, data, index):
        
        image_name = data['image_id'].iloc[index]
        study_id= data['study_id'].iloc[index]
        image_path = os.path.join(self.data_path, study_id+'/'+image_name+ '.png')
        return (image_path)
    def __getitem__(self, index):
        image_path = self.get_path(self.data, index)
        image = cv2.imread(image_path)
        if self.transform:
            image = self.transform(image)
        label = self.get_score(self.data, index) -1
        return image, label 
    
    
    def __len__(self):
        return len(self.data.index)
class MammoCompDataset(Dataset):
    def __init__(self, 
                data_path = "../VinDr_Mammo/physionet.org/files/vindr-mammo/1.0.0/images_png/",
                metadata = "../VinDr_Mammo/physionet.org/files/vindr-mammo/1.0.0/breast-level_annotations1.csv",
                phase="train", 
                mode="binary_contrastive", 
                transform=None,
                datalen=100,
                certain=True,
                seed=None):
        self.phase = phase
        self.datalen = datalen # Number of image pairs for training/testing
        self.certain = certain
        self.mode = mode
        self.data_path= data_path
        if(seed):
            seed_everything(seed)

        self.transform = data_transforms[self.phase] if(transform == None) else transform
        data = pd.read_csv(metadata)
        self.data = data.loc[data['split']== phase].reset_index()
        self.birads=[]
        for i in range(1,6):
            self.birads.append(self.data.loc[self.data['breast_birads']==f'BI-RADS {i}'])
        self.name_of_classes = [1,2,3,4,5]
        self.len_of_classes = [len(self.birads[i].index) for i in range(5)]
        self.paths1 = []
        self.paths2 = []
        self.listi1 = []
        self.listi2 = []
        self.complabels =[]
        curlen = 0
        self.imagesinclass0 = self.birads[0]
        seed_everything(seed)
        while(curlen < self.datalen):
            if(mode == 'multiclass_contrastive'):
                if(random.randint(0, 1) == 0):
                    i1 = random.randint(0, 4)
                    i2 = random.randint(0, 4)
                else:
                    i1 = random.randint(0, 4)
                    i2 = i1
                # pickimageA = randint(0, lenofclass[random_pick_2class[0]], (1,))
                self.paths1.append(self.get_path(self.birads[i1], randint(0, self.len_of_classes[i1], (1,))[0]))
                self.paths2.append(self.get_path(self.birads[i2], randint(0, self.len_of_classes[i2], (1,))[0]))
                self.complabels.append((i1 == i2) * 1)
                curlen = curlen + 1
            elif(mode == 'binary_contrastive'):
                modee = random.randint(0, 3)
                if(modee == 0):
                    i1 = 0
                    i2 = 0
                elif(modee == 1):
                    i1 = random.randint(1, 4)
                    i2 = random.randint(1, 4)
                elif(modee == 2):
                    i1 = 0
                    i2 = random.randint(1, 4)
                else:
                    i2 = 0
                    i1 = random.randint(1, 4)
                # pickimageA = randint(0, lenofclass[random_pick_2class[0]], (1,))
                self.listi1.append(i1)
                self.listi2.append(i2)
                self.paths1.append(self.get_path(self.birads[i1], randint(0, self.len_of_classes[i1], (1,))[0]))
                self.paths2.append(self.get_path(self.birads[i2], randint(0, self.len_of_classes[i2], (1,))[0]))
                self.complabels.append((((i1 == 0) and (i2 == 0)) or ((i1 !=0) and (i2 !=0))) * 1)
                curlen = curlen + 1
            elif(mode == 'severity_comparison'):
                i1 = random.randint(1, 4)
                i2 = random.randint(1, 4)
                # pickimageA = randint(0, lenofclass[random_pick_2class[0]], (1,))
                self.listi1.append(i1)
                self.listi2.append(i2)
                self.paths1.append(self.get_path(self.birads[i1], randint(0, self.len_of_classes[i1], (1,))[0]))
                self.paths2.append(self.get_path(self.birads[i2], randint(0, self.len_of_classes[i2], (1,))[0]))
                self.complabels.append(((i1 > i2)) * 1)
                curlen = curlen + 1
            elif(mode == 'preference_contrastive'):
                if(random.randint(0, 5) > 4):
                    i1 = random.randint(1, 4)
                    i2 = i1
                else:
                    i1 = random.randint(1, 4)
                    i2 = random.randint(1, 4)
                # pickimageA = randint(0, lenofclass[random_pick_2class[0]], (1,))
                self.listi1.append(i1)
                self.listi2.append(i2)
                self.paths1.append(self.get_path(self.birads[i1], randint(0, self.len_of_classes[i1], (1,))[0]))
                self.paths2.append(self.get_path(self.birads[i2], randint(0, self.len_of_classes[i2], (1,))[0]))
                self.complabels.append((((i1 > i2)) * 1) if(i1 != i2) else 2)
                curlen = curlen + 1
            else:
                assert False, f"No mode {mode} found, please try multiclass_contrastive or binary_contrastive"

    def get_score(self, data, index):
        birads= data['breast_birads'].iloc[index.item()]
        score= eval(birads[-1])
        return score
    def get_path(self, data, index):
        
        image_name = data['image_id'].iloc[index.item()]
        study_id= data['study_id'].iloc[index.item()]
        image_path = os.path.join(self.data_path, study_id+'/'+image_name+ '.png')
        return (image_path)
    def __getitem__(self, index):
        imageA = cv2.imread(self.paths1[index])
        imageB = cv2.imread(self.paths2[index])

        label = self.complabels[index]
        
        imageA = self.transform(imageA)
        imageB = self.transform(imageB)

        return (imageA, imageB), label, (self.listi1[index], self.listi2[index])
    
    def get_ref_images(self):
        ref_img = self.get_path(self.imagesinclass0,randint(0, len(self.imagesinclass0), (1,))[0])
        ref_img = cv2.imread(ref_img)
        ref_img = self.transform(ref_img)
        
        return ref_img
    
    def __len__(self):
        return self.datalen
from torch.utils.data import Dataset
from torchvision import datasets
from torch import randint
from constants import data_transforms
import sys
import os
import glob
from PIL import Image
import random

def PapilledemaDataset(data_dir = '/mnt/c/Users/Hong/Dropbox/chla_fundus_croped/',
                           phase = 'train', 
                           task = 'binary_classification'):
    
    if(task=='binary_classification'):
        return datasets.ImageFolder(os.path.join(data_dir, phase), data_transforms[phase])
    elif(task=='multiclass_classification'):
        return datasets.ImageFolder(os.path.join(data_dir, phase), data_transforms[phase])
    else:
        assert False, f'No task {task} found'

                
class PapilSeverityDataset(Dataset):
    def __init__(self, 
                data_dir = '/mnt/c/Users/Hong/Dropbox/chla_fundus_croped/', 
                phase='train', 
                mode='multiclass_contrastive', 
                datalen = 100):
        self.data_dir = data_dir
        self.phase = phase
        self.datalen = datalen
        self.transform = data_transforms[phase]

        numberofclasses = glob.glob(f'/mnt/c/Users/PCM/Dropbox/chla_fundus_croped/severity/{phase}/*')
        lenofclass = {}
        imagesinclass = {}
        for i in range(len(numberofclasses)):
            imagesinclass[i] = glob.glob(numberofclasses[i] + '/*')
            lenofclass[i] = len(glob.glob(numberofclasses[i] + '/*'))

        self.imagesinclass0 = imagesinclass[0]
        self.paths1 = []
        self.paths2 = []
        self.listi1 = []
        self.listi2 = []
        self.paths0 = []
        self.complabels = []
        curlen = 0
        while(curlen < self.datalen):
            if(mode == 'multiclass_contrastive'):
                if(random.randint(0, 1) == 0):
                    i1 = random.randint(0, 5)
                    i2 = random.randint(0, 5)
                else:
                    i1 = random.randint(0, 5)
                    i2 = i1
                # pickimageA = randint(0, lenofclass[random_pick_2class[0]], (1,))
                self.paths1.append(imagesinclass[i1][randint(0, lenofclass[i1], (1,))[0]])
                self.paths2.append(imagesinclass[i2][randint(0, lenofclass[i2], (1,))[0]])
                self.complabels.append((i1 == i2) * 1)
                curlen = curlen + 1
            elif(mode == 'binary_contrastive'):
                if(random.randint(0, 1) == 0):
                    i1 = 0
                    i2 = 0
                else:
                    i1 = 0
                    i2 = random.randint(1, 5)
                # pickimageA = randint(0, lenofclass[random_pick_2class[0]], (1,))
                self.listi1.append(i1)
                self.listi2.append(i2)
                self.paths1.append(imagesinclass[i1][randint(0, lenofclass[i1], (1,))[0]])
                self.paths2.append(imagesinclass[i2][randint(0, lenofclass[i2], (1,))[0]])
                self.complabels.append(((i1 == 0) and (i2 == 0)) * 1)
                curlen = curlen + 1
            elif(mode == 'severity_comparison'):
                i1 = random.randint(1, 5)
                i2 = random.randint(1, 5)
                # pickimageA = randint(0, lenofclass[random_pick_2class[0]], (1,))
                self.listi1.append(i1)
                self.listi2.append(i2)
                self.paths1.append(imagesinclass[i1][randint(0, lenofclass[i1], (1,))[0]])
                self.paths2.append(imagesinclass[i2][randint(0, lenofclass[i2], (1,))[0]])
                self.complabels.append(((i1 > i2)) * 1)
                curlen = curlen + 1
            elif(mode == 'preference_contrastive'):
                if(random.randint(0, 5) > 4):
                    i1 = random.randint(1, 5)
                    i2 = i1
                else:
                    i1 = random.randint(1, 5)
                    i2 = random.randint(1, 5)
                # pickimageA = randint(0, lenofclass[random_pick_2class[0]], (1,))
                self.listi1.append(i1)
                self.listi2.append(i2)
                self.paths1.append(imagesinclass[i1][randint(0, lenofclass[i1], (1,))[0]])
                self.paths2.append(imagesinclass[i2][randint(0, lenofclass[i2], (1,))[0]])
                self.complabels.append((((i1 > i2)) * 1) if(i1 != i2) else 2)
                curlen = curlen + 1
            else:
                assert False, f"No mode {mode} found, please try multiclass_contrastive or binary_contrastive"
            
    def __getitem__(self, index):
        img1 = self.paths1[index]
        img2 = self.paths2[index]
        target = self.complabels[index]

        if(self.transform):
            img1 = self.transform(Image.open(img1))
            img2 = self.transform(Image.open(img2))

        return (img1, img2), target, (self.listi1[index], self.listi2[index])
    
    def get_ref_images(self):
        ref_img = self.imagesinclass0[randint(0, len(self.imagesinclass0), (1,))[0]]
        if(self.transform):
            ref_img = self.transform(Image.open(ref_img))
        return ref_img
    
    def __len__(self):
        return self.datalen
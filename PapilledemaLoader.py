from torch.utils.data import Dataset
from torchvision import datasets
from constants import data_transforms
import sys
import os


def PapilledemaDataset(data_dir = '/mnt/c/Users/Hong/Dropbox/chla_fundus_croped/',
                           phase = 'train', 
                           task = 'binary_classification'):
    
    if(task=='binary_classification'):
        return datasets.ImageFolder(os.path.join(data_dir, phase), data_transforms[phase])
    elif(task=='multiclass_classification'):
        return datasets.ImageFolder(os.path.join(data_dir, phase), data_transforms[phase])
    else:
        assert False, f'No task {task} found'


class GlaucomaDataset(Dataset):
    def __init__(self, phase='train', task='binary_classification'):
        self.data_dir = '/mnt/c/Users/Hong/Dropbox/chla_fundus_croped/'
        self.phase = phase
        if(task=='binary_classification'):
            return datasets.ImageFolder(os.path.join(self.data_dir, self.phase))
    def __getitem__(self, index):
        return index
    
    def __len__(self):
        return 0
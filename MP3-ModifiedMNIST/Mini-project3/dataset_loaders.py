import pandas as pd
import numpy as np

import torch 
from torch.utils.data.dataset import Dataset

from torch.utils.data.sampler import SubsetRandomSampler
from preprocess import noise_remover

class ModifiedMNISTDataSet(Dataset):
    def __init__(self, train_images_path, train_labels_path, test_images_path):
        # import images (training and test)
        self.images = pd.read_pickle(train_images_path)
        self.test_images = pd.read_pickle(test_images_path)
        
        self.height = self.images.shape[1]
        self.width = self.images.shape[2]
        
        # convert images
        self.images = self.to_tensor(self.images)
        self.test_images = self.to_tensor(self.test_images)
         
        # read and convert label
        self.labels = pd.read_csv(train_labels_path)
        self.labels = torch.tensor(self.labels['Category'].tolist())
        
    # change numpy array to torch tensor of correct size
    def to_tensor(self, images):
        return torch.tensor(images).view(len(images), 1, self.height, self.width)

    # get one test image
    def get_test(self, index):
        return self.test_images[index,:,:,:].view(1,1,self.height,self.width)
    
    # access one image and corresponding image
    def __getitem__(self, index):
        image = self.images[index,:,:,:]
        label = self.labels[index]      
        return image, label

    def __len__(self):
        return len(self.images)
    

## split the indices into train/val sets
def test_val_split(dataset, batch_size, val_split=0.2, shuffle_data=False, seed=0):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(val_split * dataset_size))
    if shuffle_data:
        np.random.seed(seed)
        torch.manual_seed(seed)

        np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    
    return train_sampler, valid_sampler

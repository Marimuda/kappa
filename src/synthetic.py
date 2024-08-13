import torch
import os
import sys
from dataset.vertebra_dataset_factory import VertebraDatasetFactory

# sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

import numpy as np

def mask_volume(vol, shape='sphere'):
    D, H, W = vol.shape
    mask = np.ones((D, H, W), dtype=np.uint8)
    
    if shape == 'sphere':
        center = np.array([np.random.randint(D), np.random.randint(H), np.random.randint(W)])
        radius = np.random.randint(5, min(D, H, W) // 2)
        Z, Y, X = np.ogrid[:D, :H, :W]
        dist_from_center = np.sqrt((Z - center[0])**2 + (Y - center[1])**2 + (X - center[2])**2)
        mask[dist_from_center < radius] = 0
    elif shape == 'cube':
        start = np.array([np.random.randint(D), np.random.randint(H), np.random.randint(W)])
        size = np.random.randint(5, min(D, H, W) // 2)
        end = start + size
        end = np.minimum(end, [D, H, W])
        mask[start[0]:end[0], start[1]:end[1], start[2]:end[2]] = 0
    elif shape == 'cone':
        center = np.array([np.random.randint(D), np.random.randint(H), np.random.randint(W)])
        height = np.random.randint(5, D // 2)
        radius = np.random.randint(5, min(H, W) // 2)
        Z, Y, X = np.ogrid[:D, :H, :W]
        dist_from_center = np.sqrt((Y - center[1])**2 + (X - center[2])**2)
        cone_mask = (Z < center[0] + height) & (dist_from_center < radius * (1 - (Z - center[0]) / height))
        mask[cone_mask] = 0

    return vol * mask



class synDataset(torch.utils.data.Dataset):
    def __init__(self, base_path: str, dataset_type: str, transforms=None):
        self.base_path = base_path
        self.dataset_type = dataset_type
        self.transforms = transforms
        self.crop_factory = VertebraDatasetFactory(base_path=base_path)
        self.crop_dataset = self.crop_factory.create_dataset(dataset_type=dataset_type)
    
    def __len__(self):
        return len(self.crop_dataset)
    
    def __add_artifacts(self, sample):
        choice = ['sphere', 'cube', 'cone']
        shape = np.random.choice(choice)
        sample['image'] = mask_volume(sample['image'], shape=shape)
        return sample

    
    def __getitem__(self, idx):
        sample = self.crop_dataset[idx]
        sample = self.__add_artifacts(sample)
        return sample



def main():

    vol = np.random.rand(100, 100, 100)  
    masked_vol = mask_volume(vol, shape='sphere') 

    breakpoint()
    # base_path = "/dtu/blackhole/14/189044/atde/challenge_data/train"
    # # save_path = "/dtu/blackhole/14/189044/atde/challenge_model/"
    # base_path = '/work3/rapa/challenge_data/train'
    # crop_factory = VertebraDatasetFactory(base_path=base_path)
    # crop_dataset = crop_factory.create_dataset(dataset_type='crop')
    

    testdata = crop_dataset[0]
    breakpoint()

if __name__ == '__main__':
    main()
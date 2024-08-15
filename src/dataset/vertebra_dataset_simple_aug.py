import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from typing import List, Callable, Union
import numpy as np
import os
import random

class VertebraDatasetSimpleAug(Dataset):
    """General dataset class for loading vertebra data, including crops, distance fields, and meshes."""

    def __init__(self, file_paths: List[str], loader: Callable[[str], Union[torch.Tensor, Data]], return_sample_id=False) -> None:
        """Initializes the dataset with file paths and a loader function.

        Args:
            file_paths (List[str]): List of file paths to the data files.
            loader (Callable[[str], Union[torch.Tensor, Data]]): Function to load a data file.
        """
        self.file_paths = file_paths
        self.loader = loader
        self.return_sample_id = return_sample_id
        self.pos_len = len(self.file_paths)
        
        
    def __len__(self) -> int:
        """Returns the number of samples in the dataset.

        Returns:
            int: The number of data samples.
        """
        return 2*len(self.file_paths)

    def __getitem__(self, idx: int) -> Union[torch.Tensor, Data]:
        """Loads a data sample from the dataset.

        Args:
            idx (int): Index of the file to load.

        Returns:
            Union[torch.Tensor, Data]: The loaded data, which could be an image, distance field, or mesh.
        """
        if idx < self.pos_len :
            orig_idx = idx 
            sample_outlier = False
        else :
            orig_idx = idx//2
            sample_outlier = True

        if sample_outlier:
            outliers = ['sphere_outlier_mean_std_inpaint', 'sphere_outlier_water', 'warp_outlier']
            outlier_type = random.choice(outliers)
           
        file_path = self.file_paths[orig_idx]
        label = file_path.__contains__('outlier')
        sample = self.loader(file_path)
        file_name = file_path.split('/')[-1]
        if sample_outlier:
            file_name = file_name.split('.nii.gz')[0]
            file_name = f'{file_name}_{outlier_type}.nii.gz'
            
        path_to_file = file_path.split('/')[:-1]
        path_to_file = '/'.join(path_to_file)

        f1,f2 = file_name.split('crop')
        label_name = f'{f1}crop_label{f2}'
        segmentation = self.loader(os.path.join(path_to_file, label_name))
        
        if self.return_sample_id :
            
            sample_id = file_path.split('/')[-1]
            sample_id = sample_id.split('_')
            sample_id = f'{sample_id[0]}_{sample_id[1]}'
            return sample, segmentation, float(label), sample_id
        else :
            return sample, segmentation, float(label)

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from typing import List, Callable, Union

class VertebraDataset(Dataset):
    """General dataset class for loading vertebra data, including crops, distance fields, and meshes."""

    def __init__(self, file_paths: List[str], loader: Callable[[str], Union[torch.Tensor, Data]]) -> None:
        """Initializes the dataset with file paths and a loader function.

        Args:
            file_paths (List[str]): List of file paths to the data files.
            loader (Callable[[str], Union[torch.Tensor, Data]]): Function to load a data file.
        """
        self.file_paths = file_paths
        self.loader = loader

    def __len__(self) -> int:
        """Returns the number of samples in the dataset.

        Returns:
            int: The number of data samples.
        """
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Union[torch.Tensor, Data]:
        """Loads a data sample from the dataset.

        Args:
            idx (int): Index of the file to load.

        Returns:
            Union[torch.Tensor, Data]: The loaded data, which could be an image, distance field, or mesh.
        """
        file_path = self.file_paths[idx]
        return self.loader(file_path)

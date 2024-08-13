import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

import torch
from dataset.vertebra_dataset_factory import VertebraDatasetFactory
from torch_geometric.data import Data


def inspect_dataset(dataset, dataset_name):
    """Prints out details about the dataset."""
    print(f"--- {dataset_name} Dataset ---")
    print(f"Number of samples: {len(dataset)}")

    if len(dataset) > 0:
        sample = dataset[0]
        if isinstance(sample, torch.Tensor):
            print(f"Sample shape: {sample.shape}")
        elif isinstance(sample, Data):
            print(f"Sample vertices shape: {sample.pos.shape}")
            print(f"Sample faces shape: {sample.face.shape}")
        else:
            print("Unknown data type")

    print()

def main():
    # Paths to your datasets (modify these paths based on your actual data location)
    base_path = "/home/jakupsv/repo/phd/data/challenge_data/train"

    # Create the Crop dataset
    crop_factory = VertebraDatasetFactory(dataset_type='crop', base_path=base_path)
    crop_dataset = crop_factory.create_dataset()
    inspect_dataset(crop_dataset, "Crop")

    # Create the Distance Field dataset
    dist_field_factory = VertebraDatasetFactory(dataset_type='dist_field', base_path=base_path)
    dist_field_dataset = dist_field_factory.create_dataset()
    inspect_dataset(dist_field_dataset, "Distance Field")

    # Create the Mesh dataset
    mesh_factory = VertebraDatasetFactory(dataset_type='mesh', base_path=base_path)
    mesh_dataset = mesh_factory.create_dataset()
    inspect_dataset(mesh_dataset, "Mesh")

if __name__ == "__main__":
    main()

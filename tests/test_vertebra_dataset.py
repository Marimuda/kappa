import pytest
import torch
from torch_geometric.data import Data
from dataset.vertebra_dataset_factory import VertebraDatasetFactory
import logging

logger = logging.getLogger(__name__)

def test_vertebra_dataset_len(crop_dataset_path):
    """Tests that the length of the dataset is correct."""
    factory = VertebraDatasetFactory(dataset_type='crop', base_path=crop_dataset_path)
    dataset = factory.create_dataset()

    assert len(dataset) > 0, "The dataset should not be empty."
    logger.debug(f"Dataset length: {len(dataset)}")


def test_vertebra_dataset_getitem_tensor(crop_dataset_path, dist_field_dataset_path):
    """Tests that the __getitem__ method returns a tensor for crop/dist field data."""
    factory = VertebraDatasetFactory(dataset_type='crop', base_path=crop_dataset_path)
    crop_dataset = factory.create_dataset()

    factory = VertebraDatasetFactory(dataset_type='dist_field', base_path=dist_field_dataset_path)
    dist_field_dataset = factory.create_dataset()

    crop_data = crop_dataset[0]
    dist_data = dist_field_dataset[0]

    assert isinstance(crop_data, torch.Tensor), "Loaded crop data should be a torch.Tensor."
    assert isinstance(dist_data, torch.Tensor), "Loaded distance field data should be a torch.Tensor."

    logger.debug(f"Crop data shape: {crop_data.shape}")
    logger.debug(f"Distance field data shape: {dist_data.shape}")


def test_vertebra_dataset_getitem_data(surfaces_dataset_path):
    """Tests that the __getitem__ method returns a Data object for mesh data."""
    factory = VertebraDatasetFactory(dataset_type='mesh', base_path=surfaces_dataset_path)
    dataset = factory.create_dataset()

    data = dataset[0]

    assert isinstance(data, Data), "Loaded data should be a torch_geometric.data.Data object."
    assert hasattr(data, 'pos'), "Data should have positions attribute."
    assert hasattr(data, 'face'), "Data should have faces attribute."

    logger.debug(f"Mesh vertices shape: {data.pos.shape}")
    logger.debug(f"Mesh faces shape: {data.face.shape}")

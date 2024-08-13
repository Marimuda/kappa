import pytest
import os

@pytest.fixture(scope='module')
def data_base_path():
    """Fixture to provide the base path for the test data."""
    return os.path.join(os.path.dirname(__file__), 'data')

@pytest.fixture(scope='module')
def crop_dataset_path(data_base_path):
    """Fixture to provide the path to the crop dataset."""
    return data_base_path

@pytest.fixture(scope='module')
def dist_field_dataset_path(data_base_path):
    """Fixture to provide the path to the distance field dataset."""
    return data_base_path

@pytest.fixture(scope='module')
def surfaces_dataset_path(data_base_path):
    """Fixture to provide the path to the surfaces dataset."""
    return data_base_path

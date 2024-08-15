import os
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from typing import List, Tuple, Union, Callable, Dict
import logging
from dataset.vertebra_dataset_simple_aug import VertebraDatasetSimpleAug
import vtk 
import itk

logger = logging.getLogger(__name__)

class VertebraDatasetFactoryPositive:
    """
    A factory for creating VertebraDataset objects based on the type of data requested.

    This class supports loading cropped CT scans, distance fields, and surface meshes of vertebrae.
    The specific type of data to load is determined by the `dataset_type` parameter.
    """

    def __init__(self, base_path: str, crop_dir: str = 'crop', dist_field_dir: str = 'dist_field', surfaces_dir: str = 'surfaces',
                 subdirectory: str = '', return_sample_id=False) -> None:
        """
        Initializes the dataset factory with base path and optional subdirectories.

        Args:
            base_path (str): The base directory path where the data is stored.
            crop_dir (str): Subdirectory for cropped images.
            dist_field_dir (str): Subdirectory for distance fields.
            surfaces_dir (str): Subdirectory for surface meshes.
            subdirectory (str): Additional subdirectory (e.g., 'train') to prepend to each data directory.
        """
        self.base_path = base_path
        self.crop_dir = crop_dir
        self.dist_field_dir = dist_field_dir
        self.surfaces_dir = surfaces_dir
        self.subdirectory = subdirectory
        self.return_sample_id = return_sample_id
        self.datasets: Dict[str, Dataset] = {}  # Initialize an empty cache for datasets

    def _initialize_dataset(self, dataset_type: str):
        """
        Initializes dataset paths and sets the loader function based on the dataset type.

        Args:
            dataset_type (str): The type of dataset ('crop', 'dist_field', 'mesh').

        Returns:
            Tuple[str, str, Callable[[str], Union[torch.Tensor, Data]]]: The directory path, file extension, and the loader function for the dataset type.

        Raises:
            ValueError: If an unsupported dataset type is provided.
        """
        if dataset_type == 'crop':
            return os.path.join(self.base_path, self.subdirectory, f'{self.crop_dir}s'), ".nii.gz", self._load_nifti
        elif dataset_type == 'dist_field':
            return os.path.join(self.base_path, self.subdirectory, f'{self.dist_field_dir}s'), ".nii.gz", self._load_nifti
        elif dataset_type == 'mesh':
            return os.path.join(self.base_path, self.subdirectory, self.surfaces_dir), ".vtk", self._load_vtk_mesh
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}. Supported types are 'crop', 'dist_field', or 'mesh'.")

    def _collect_file_paths(self, directory: str, extension: str) -> List[str]:
        """
        Collects file paths with the specified extension from the given directory.

        Args:
            directory (str): Directory to search for files.
            extension (str): File extension to filter files.

        Returns:
            List[str]: List of file paths matching the specified extension.

        Raises:
            FileNotFoundError: If the specified directory does not exist.
        """
        if not os.path.exists(directory):
            logger.error(f"Directory does not exist: {directory}")
            raise FileNotFoundError(f"Directory does not exist: {directory}")

        # NOTE: positive only
        all_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(extension) and 'outlier' not in f]
        files = []
        for f in all_files:
            if not f.__contains__('label'):
                files.append(f)

        logger.debug(f"Collected {len(files)} files with extension {extension} from {directory}.")

        if not files:
            logger.warning(f"No files found with extension {extension} in {directory}.")

        return files

    def _load_nifti(self, file_path: str) -> torch.Tensor:
        """
        Loads a NIfTI file and converts it to a torch.Tensor.

        Args:
            file_path (str): Path to the NIfTI file.

        Returns:
            torch.Tensor: A tensor containing the image data.

        Raises:
            ImportError: If the `itk` module is not found.
            IOError: If the NIfTI file cannot be loaded.
        """
        try:
            image = itk.imread(file_path)
            array = itk.array_from_image(image)
            array = array.astype('float')
           
            logger.debug(f"NIfTI file {file_path} loaded with shape {array.shape}.")
            return torch.tensor(array, dtype=torch.float32)
        except ImportError as e:
            logger.error(f"Failed to import required module for NIfTI loading: {e}")
            raise
        except (itk.itkBase.IOError, FileNotFoundError) as e:
            logger.error(f"Failed to load NIfTI file: {file_path}. Error: {e}")
            raise

    def _load_vtk_mesh(self, file_path: str) -> Data:
        """
        Loads a VTK mesh file and converts it into a torch_geometric Data object.

        Args:
            file_path (str): Path to the VTK file.

        Returns:
            Data: A torch_geometric Data object containing the mesh vertices and faces.

        Raises:
            ImportError: If the `vtk` module is not found.
            RuntimeError: If the VTK file cannot be loaded.
        """
        try:
            reader = vtk.vtkPolyDataReader()
            reader.SetFileName(file_path)
            reader.Update()
            polydata = reader.GetOutput()

            vertices = torch.tensor([polydata.GetPoint(i) for i in range(polydata.GetNumberOfPoints())], dtype=torch.float)
            faces = self._extract_faces(polydata)

            logger.debug(f"VTK mesh {file_path} loaded with {vertices.shape[0]} vertices and {faces.shape[1]} faces.")
            return Data(pos=vertices, face=faces)
        except ImportError as e:
            logger.error(f"Failed to import required module for VTK loading: {e}")
            raise
        except RuntimeError as e:
            logger.error(f"Failed to load VTK mesh file: {file_path}. Error: {e}")
            raise

    def get_dataset(self, dataset_type: str) -> Dataset:
        """
        Returns the dataset for the specified type. If it doesn't exist, create it, cache it, and return it.

        Args:
            dataset_type (str): The type of dataset to return ('crop', 'dist_field', 'mesh').

        Returns:
            Dataset: The dataset object corresponding to the requested data type.
        """
        if dataset_type in self.datasets:
            logger.info(f"Returning cached {dataset_type} dataset.")
            return self.datasets[dataset_type]
        else:
            logger.info(f"{dataset_type.capitalize()} dataset not found in cache. Creating new dataset.")
            self.datasets[dataset_type] = self.create_dataset(dataset_type)
            return self.datasets[dataset_type]

    def create_dataset(self, dataset_type: str) -> Dataset:
        """
        Creates and returns a VertebraDataset object based on the specified data type.

        Args:
            dataset_type (str): The type of dataset to create ('crop', 'dist_field', 'mesh').

        Returns:
            VertebraDataset: A dataset object containing the loaded data.
        """
        data_dir, file_extension, loader = self._initialize_dataset(dataset_type)

        file_paths = self._collect_file_paths(data_dir, file_extension)#[:100]
        return VertebraDatasetSimpleAug(file_paths, loader, return_sample_id=self.return_sample_id)

    def _extract_faces(self, polydata) -> torch.Tensor:
        """
        Extracts face data from VTK polydata and returns it as a tensor.

        Args:
            polydata: VTK polydata containing the mesh information.

        Returns:
            torch.Tensor: A tensor containing the face indices.
        """
        faces = []
        polydata.GetPolys().InitTraversal()
        id_list = vtk.vtkIdList()
        while polydata.GetPolys().GetNextCell(id_list):
            faces.append([id_list.GetId(j) for j in range(id_list.GetNumberOfIds())])
        return torch.tensor(faces, dtype=torch.long).t().contiguous()

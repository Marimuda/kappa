import logging
from pyvistaqt import QtInteractor
from handlers.mesh_handler import MeshHandler
from handlers.volume_handler import VolumeHandler

logger = logging.getLogger(__name__)

class MeshVisualizer:
    def __init__(self, factory, plotter: QtInteractor):
        self.factory = factory
        self.plotter = plotter
        self.mesh_handler = MeshHandler(factory, plotter)
        self.volume_handler = VolumeHandler(factory, plotter)
        self.current_handler = self.mesh_handler  # Start with the mesh handler

    def set_data_type(self, data_type: str):
        """Set the data type for visualization and switch handler."""
        if data_type == "mesh":
            self.current_handler = self.mesh_handler
        elif data_type in ["crop", "dist_field"]:
            self.volume_handler.data_type = data_type
            self.volume_handler.dataset = self.factory.get_dataset(data_type)
            self.current_handler = self.volume_handler

        logger.info(f"Switching to {data_type} visualization.")
        self.current_handler.load_at_index(self.current_handler.current_index)

    def load_next(self):
        """Load the next data in the dataset."""
        self.current_handler.load_next()

    def load_previous(self):
        """Load the previous data in the dataset."""
        self.current_handler.load_previous()

    def load_at_index(self, index: int):
        """Load data at a specific index."""
        self.current_handler.load_at_index(index)

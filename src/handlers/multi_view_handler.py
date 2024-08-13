import logging
import pyvista as pv
from pyvistaqt import QtInteractor
from .mesh_handler import MeshHandler
from .volume_handler import VolumeHandler

logger = logging.getLogger(__name__)

class MultiViewHandler:
    def __init__(self, factory, plotter: QtInteractor):
        self.factory = factory
        self.plotter = plotter
        self.mesh_handler = MeshHandler(factory, plotter)
        self.crop_handler = VolumeHandler(factory, plotter, "crop")
        self.dist_field_handler = VolumeHandler(factory, plotter, "dist_field")

    def setup_multi_view(self):
        """Set up multi-view plotting for comparing different representations."""
        plotter = pv.Plotter(shape=(2, 2))
        
        plotter.subplot(0, 0)
        plotter.add_text("Mesh")
        self.mesh_handler.load_data(self.mesh_handler.current_index)
        
        plotter.subplot(0, 1)
        plotter.add_text("Crop")
        self.crop_handler.load_data(self.crop_handler.current_index)
        
        plotter.subplot(1, 0)
        plotter.add_text("Distance Field")
        self.dist_field_handler.load_data(self.dist_field_handler.current_index)

        plotter.show()

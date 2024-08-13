import numpy as np
import logging
import pyvista as pv
from pyvistaqt import QtInteractor
from torch_geometric.data import Data
from .base_handler import BaseHandler

logger = logging.getLogger(__name__)

class MeshHandler(BaseHandler):
    def __init__(self, factory, plotter: QtInteractor):
        super().__init__()
        self.factory = factory
        self.plotter = plotter
        self.dataset = self.factory.get_dataset("mesh")

    def load_data(self, index):
        """Load and render a mesh by index."""
        logger.info(f"Loading mesh data at index {index}")
        self.current_index = index
        self.data_cache = self.dataset[index]
        self._render_mesh()

    def _render_mesh(self):
        """Render the currently loaded mesh."""
        if isinstance(self.data_cache, Data):
            vertices = self.data_cache.pos.numpy()
            faces = self.data_cache.face.numpy().T
            faces = np.hstack([[3] + list(face) for face in faces])
            mesh = pv.PolyData(vertices, faces)
            self.plotter.clear()
            self.plotter.add_mesh(mesh, show_edges=True, color=np.random.rand(3,))
            self.plotter.view_xy()
            self.plotter.render()

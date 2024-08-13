import logging
import pyvista as pv
import torch
import numpy as np
from .base_handler import BaseHandler

logger = logging.getLogger(__name__)

class VolumeHandler(BaseHandler):
    def __init__(self, factory, plotter: pv.Plotter, data_type="crop"):
        super().__init__()
        self.factory = factory
        self.plotter = plotter
        self.data_type = data_type
        self.dataset = self.factory.get_dataset(self.data_type)

    def load_data(self, index):
        """Load and render a volume by index."""
        logger.info(f"Loading volume data ({self.data_type}) at index {index}")
        self.current_index = index
        self.data_cache = self.dataset[index]

        if self.data_cache is None or not isinstance(self.data_cache, torch.Tensor):
            logger.warning(f"No valid data found for index {index} in {self.data_type} dataset.")
            return

        if self.data_type == "crop":
            self._render_crop()
        elif self.data_type == "dist_field":
            self._render_distance_field()

    def _render_crop(self):
        """Render the currently loaded crop as a 3D volume with a refined opacity map to reduce noise."""
        try:
            volume_data = self.data_cache.numpy()

            # Log the original data range
            logger.info(f"Original data range: min={volume_data.min()}, max={volume_data.max()}")

            # Define clamping range based on typical Hounsfield units for medical imaging
            min_intensity = 0
            max_intensity = 1000

            # Clamp the data to the specified range
            volume_data = np.clip(volume_data, min_intensity, max_intensity)

            # Log the clamped data range
            logger.info(f"Clamped data range: min={volume_data.min()}, max={volume_data.max()}")

            # Check if the data is 4D (e.g., [depth, height, width, channels]) and reduce it to 3D if necessary
            if volume_data.ndim == 4:
                logger.warning(f"Volume data has {volume_data.shape[3]} channels; using the first channel for rendering.")
                volume_data = volume_data[..., 0]  # Use the first channel

            # Ensure the data is now 3D
            if volume_data.ndim != 3:
                raise ValueError(f"Expected a 3D array for volume data, but got {volume_data.ndim}D array instead.")

            # Wrap the volume data for PyVista
            volume = pv.wrap(volume_data)

            # Clear previous render
            self.plotter.clear()

            # Refined opacity mapping
            opacity = [0.0, 0.05, 0.1, 0.3, 0.6, 0.9]  # More aggressive to reduce noise and emphasize bones

            # Use the "bone" colormap for better visualization
            self.plotter.add_volume(
                volume,
                cmap="bone",    # Colormap suitable for medical imaging
                opacity=opacity,  # Apply the refined opacity mapping
                shade=True,       # Enable shading for depth perception
                ambient=0.4,      # Adjust ambient light for overall brightness
                diffuse=0.6,      # Diffuse light for soft shadows
                specular=0.3,     # Specular light to highlight edges
                specular_power=15 # Specular power for sharper highlights
            )

            # Set the view to isometric for a better 3D perspective
            self.plotter.view_isometric()
            self.plotter.render()

        except Exception as e:
            logger.error(f"Error rendering crop volume: {e}")

    def _render_distance_field(self):
        """Render the distance field using isosurfaces for different distance values."""
        try:
            distance_data = self.data_cache.numpy()

            # Wrap the distance field data for PyVista
            distance_field = pv.wrap(distance_data)

            # Clear previous render
            self.plotter.clear()

            # Add an isosurface where the distance is 0 (the actual surface of the vertebra)
            self.plotter.add_mesh(distance_field.contour([0]), cmap="bone", opacity=0.8, label="Surface")

            # Add isosurfaces at other interesting distance levels (e.g., -10 and +10) with a different colormap
            self.plotter.add_mesh(distance_field.contour([-10, 10]), cmap="coolwarm", opacity=0.5, label="Distance Field Contours")

            self.plotter.add_legend()
            self.plotter.view_isometric()
            self.plotter.render()

        except Exception as e:
            logger.error(f"Error rendering distance field: {e}")

import logging
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QSlider, QHBoxLayout, QSizePolicy
from PyQt6.QtCore import Qt, QEvent
from PyQt6.QtGui import QCloseEvent, QKeyEvent
from pyvistaqt import QtInteractor
from viewer.mesh_visualiser import MeshVisualizer
from dataset.vertebra_dataset_factory import VertebraDatasetFactory

logger = logging.getLogger(__name__)

# Stylesheet for modern look
STYLESHEET = """
    QWidget {
        background-color: #2E3440;
    }
    QPushButton {
        background-color: #4C566A;
        color: #ECEFF4;
        border: 1px solid #81A1C1;
        border-radius: 5px;
        padding: 10px;
        font-size: 16px;
    }
    QPushButton:hover {
        background-color: #81A1C1;
        color: #2E3440;
    }
    QPushButton:pressed {
        background-color: #88C0D0;
    }
"""

class MeshViewerUI(QWidget):
    def __init__(self, config: dict, dataset_factory: VertebraDatasetFactory) -> None:
        """Initializes the Viewer UI based on the provided config."""
        super().__init__()
        self.config = config
        self.setWindowTitle(config['window']['title'])
        self.setStyleSheet(STYLESHEET)
        self.resize(config['window']['width'], config['window']['height'])

        # Ensure this widget can receive keyboard focus
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # Initialize the PyVista interactor
        self.plotter = QtInteractor(self)

        # Create the MeshVisualizer and associate it with the plotter
        self.mesh_visualizer = MeshVisualizer(dataset_factory, self.plotter)

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Sets up the main UI elements."""
        main_layout = QHBoxLayout()

        # Add buttons
        self._add_buttons(main_layout)

        # Add PyVista rendering window
        self._add_mesh_visualizer(main_layout)

        # Add vertical slider
        self._add_slider(main_layout)

        self.setLayout(main_layout)
        self.show()
        self._adjust_button_widths()

        # Load the initial mesh
        self.mesh_visualizer.load_next()

        # Set the focus to the widget to ensure it receives key events
        self.setFocus()

    def _add_buttons(self, layout: QHBoxLayout) -> None:
        """Adds buttons to the UI based on the configuration."""
        button_layout = QVBoxLayout()
        for view_name in self.config['buttons']['names']:
            button = QPushButton(view_name)
            button.setFixedHeight(self.height() // self.config['buttons']['height_division'])
            button.clicked.connect(lambda checked, view_name=view_name: self._on_button_click(view_name))
            button_layout.addWidget(button, alignment=Qt.AlignmentFlag.AlignLeft)
        layout.addLayout(button_layout)

    def _add_mesh_visualizer(self, layout: QHBoxLayout) -> None:
        """Adds the PyVista rendering window."""
        mesh_widget = self.plotter.interactor
        mesh_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        mesh_widget.setFixedSize(int(self.width() * 0.8), int(self.height() * 0.9))
        layout.addWidget(mesh_widget, alignment=Qt.AlignmentFlag.AlignCenter)

    def _add_slider(self, layout: QHBoxLayout) -> None:
        """Adds a vertical slider to the UI."""
        slider_config = self.config['slider']
        self.slider = QSlider(Qt.Orientation.Vertical)
        self.slider.setRange(slider_config['min'], slider_config['max'])
        self.slider.setValue(slider_config['initial_value'])
        self.slider.setTickPosition(QSlider.TickPosition.TicksRight)
        self.slider.valueChanged.connect(self._on_slider_value_changed)
        layout.addWidget(self.slider, alignment=Qt.AlignmentFlag.AlignRight)

    def _on_slider_value_changed(self, value: int) -> None:
        """Handles slider value changes to load different data based on the current view."""
        logger.info(f"Slider moved to {value}")
        self.mesh_visualizer.load_at_index(value)

    def _on_button_click(self, view_name: str) -> None:
        """Handles button clicks to change views in the visualizer."""
        logger.info(f"{view_name} button clicked")

        # Map button names to data types
        view_map = {
            "Mesh View": "mesh",
            "Crop View": "crop",
            "Distance Field View": "dist_field"
        }

        if view_name in view_map:
            self.mesh_visualizer.set_data_type(view_map[view_name])

    def _adjust_button_widths(self) -> None:
        """Adjusts the width of the buttons dynamically based on the configuration."""
        for button in self.findChildren(QPushButton):
            button.setFixedWidth(int(self.width() * self.config['buttons']['width_percentage'] / 100))

    def resizeEvent(self, event) -> None:
        """Handle the resize event to adjust button widths dynamically."""
        self._adjust_button_widths()
        super().resizeEvent(event)

    def closeEvent(self, event: QCloseEvent) -> None:
        """Handles the close event of the QWidget."""
        logger.info("Application closed.")
        event.accept()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Handle key press events to navigate through images."""
        if event.key() == Qt.Key_Left:
            logger.info("Left arrow key pressed.")
            self.mesh_visualizer.load_previous()
        elif event.key() == Qt.Key_Right:
            logger.info("Right arrow key pressed.")
            self.mesh_visualizer.load_next()
        else:
            super().keyPressEvent(event)  # Call the base class implementation

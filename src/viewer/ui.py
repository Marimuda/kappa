import logging
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QSlider, QHBoxLayout
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QCloseEvent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
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
    def __init__(self, config: dict) -> None:
        """Initializes the Viewer UI based on the provided config."""
        super().__init__()
        self.config = config
        self.setWindowTitle(config['window']['title'])
        self.setStyleSheet(STYLESHEET)
        self.resize(config['window']['width'], config['window']['height'])
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Sets up the main UI elements."""
        main_layout = QHBoxLayout()

        # Add buttons
        self._add_buttons(main_layout)

        # Add vertical slider
        self._add_slider(main_layout)

        self.setLayout(main_layout)
        self.show()
        self._adjust_button_widths()

    def _add_buttons(self, layout: QHBoxLayout) -> None:
        """Adds buttons to the UI based on the configuration."""
        button_layout = QVBoxLayout()
        for organ in self.config['buttons']['names']:
            button = QPushButton(organ)
            button.setFixedHeight(self.height() // self.config['buttons']['height_division'])
            button.clicked.connect(lambda checked, organ=organ: self._on_button_click(organ))
            button_layout.addWidget(button, alignment=Qt.AlignmentFlag.AlignLeft)
        layout.addLayout(button_layout)

    def _add_slider(self, layout: QHBoxLayout) -> None:
        """Adds a vertical slider to the UI."""
        slider_config = self.config['slider']
        self.slider = QSlider(Qt.Orientation.Vertical)
        self.slider.setRange(slider_config['min'], slider_config['max'])
        self.slider.setValue(slider_config['initial_value'])
        self.slider.setTickPosition(QSlider.TickPosition.TicksRight)
        layout.addWidget(self.slider, alignment=Qt.AlignmentFlag.AlignRight)

    def _on_button_click(self, organ: str) -> None:
        """Logs a message when a button is clicked."""
        logger.info(f"{organ} button clicked")

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

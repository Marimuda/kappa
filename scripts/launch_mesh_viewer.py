import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from dotenv import load_dotenv
from PyQt6.QtWidgets import QApplication
from viewer.ui import MeshViewerUI
from dataset.vertebra_dataset_factory import VertebraDatasetFactory

def main():
    # Load environment variables
    load_dotenv()

    # Get the base path for data from environment variables
    base_path = os.getenv("BASE_PATH", '/default/path/if/not/set')

    # Initialize the dataset factory
    factory = VertebraDatasetFactory(base_path=base_path)

    # PyQt application setup
    app = QApplication(sys.argv)

    # Configuration for the UI
    config = {
        'window': {
            'title': 'Mesh Viewer',
            'width': 1200,
            'height': 800,
        },
        'buttons': {
            'names': ['Mesh View', 'Crop View', 'Distance Field View'],
            'height_division': 10,
            'width_percentage': 20,
        },
        'slider': {
            'min': 0,
            'max': 100,  # Example max value
            'initial_value': 50,
        }
    }

    # Initialize and display the UI
    viewer_ui = MeshViewerUI(config, factory)
    viewer_ui.show()

    # Execute the application
    sys.exit(app.exec())

if __name__ == "__main__":
    main()

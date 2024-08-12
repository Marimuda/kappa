import sys
from PyQt6.QtWidgets import QApplication
from config_loader import load_config
from viewer.ui import MeshViewerUI

def main():
    """Main entry point for the application."""
    config = load_config()  # Load configuration from config.yaml
    app = QApplication(sys.argv)
    window = MeshViewerUI(config)
    sys.exit(app.exec())

if __name__ == '__main__':
    main()

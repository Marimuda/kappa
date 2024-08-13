import logging

logger = logging.getLogger(__name__)

class BaseHandler:
    def __init__(self):
        self.current_index = 0
        self.dataset = None

    def load_next(self):
        """Load the next data in the dataset."""
        if self.current_index < len(self.dataset) - 1:
            self.current_index += 1
            self.load_at_index(self.current_index)

    def load_previous(self):
        """Load the previous data in the dataset."""
        if self.current_index > 0:
            self.current_index -= 1
            self.load_at_index(self.current_index)

    def load_at_index(self, index: int):
        """Load data at a specific index."""
        if 0 <= index < len(self.dataset):
            self.load_data(index)
        else:
            logger.warning(f"Index {index} is out of bounds for the dataset.")
    
    def load_data(self, index: int):
        """Placeholder for loading data, to be implemented by subclasses."""
        raise NotImplementedError("Subclasses should implement this method.")

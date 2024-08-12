import yaml

def load_config(file_path: str = "config.yaml") -> dict:
    """Loads the configuration from a YAML file."""
    with open(file_path, "r") as file:
        return yaml.safe_load(file)

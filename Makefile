# Variables
PYTHON = python

# Run the application
run:
	$(PYTHON) scripts/launch_mesh_viewer.py

# Clean up unnecessary files
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -r {} +
	rm -rf .pytest_cache

# Run tests
test:
	$(PYTHON) -m pytest

# Format the code using black
format:
	$(PYTHON) -m black src/ tests/

# Lint the codebase using flake8
lint:
	$(PYTHON) -m flake8 src/

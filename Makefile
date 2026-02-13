.PHONY: help install clean test examples

VENV_NAME = pysindy_env
PYTHON = python
PIP = pip


help:
	@echo "Available:"
	@echo "  make install    - create venv for pysindye time"
	@echo "  make clean      - delete venv"
	@echo "  test            - run tests"
	@echo "  examples        - run for examples"
	@echo "  fixed_run       - run fixed example"

install: $(VENV_NAME)/bin/activate

$(VENV_NAME)/bin/activate: requirements.txt
	@echo "Creating venv ░"
	python3 -m venv $(VENV_NAME)
	@echo "Installing dependencies ▒"
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .
	@echo "Installation completed!"
	@echo "You can activate env via: source $(VENV_NAME)/bin/activate"

clean:
	rm -rf $(VENV_NAME)
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

test: $(VENV_NAME)/bin/activate
	$(PYTHON) -m pytest tests/

examples: $(VENV_NAME)/bin/activate
	@echo "Examples launch..."
	cd examples && cd tutorial_1 && $(PYTHON) example.py && cd .. && cd tutorial_2 && $(PYTHON) example.py

fixed_run: $(VENV_NAME)/bin/activate
	@echo "run fixed coefficients..."
	cd examples && $(PYTHON) example_fixed.py

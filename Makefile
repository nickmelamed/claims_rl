# Project Makefile

PYTHON = python3
PIP = pip

# Default target

.DEFAULT_GOAL := help

# setup

install:
$(PIP) install -e .

install-dev:
$(PIP) install -e .
$(PIP) install pytest

clean:
find . -name "**pycache**" -delete
find . -name "*.pyc" -delete

# run 

train:
$(PYTHON) scripts/train.py

episode:
$(PYTHON) scripts/run_episode.py

# tests

test:
pytest

# experiments

train-exp:
$(PYTHON) scripts/train.py --exp_name=default

# full reset

reset: clean
rm -rf build dist *.egg-info

# help 

help:
@echo "Available commands:"
@echo "  make install        Install package"
@echo "  make install-dev    Install with dev deps"
@echo "  make train          Run training"
@echo "  make episode        Run single episode"
@echo "  make test           Run tests"
@echo "  make clean          Remove cache files"
@echo "  make reset          Full cleanup"

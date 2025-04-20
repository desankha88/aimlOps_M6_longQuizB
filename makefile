# Makefile
install:
	pip install --upgrade pip && pip install -r requirements/requirements.txt && pip install -r requirements/test_requirements.txt

test:
	python -m pytest tests/test_*.py

all: install test
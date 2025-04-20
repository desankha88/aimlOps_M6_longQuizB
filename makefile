# Makefile 

install: 
    pip install --upgrade pip && pip install -r requirements.txt 
 
test: 
    python -m pytest tests/test_*.py

 all: 
    install test
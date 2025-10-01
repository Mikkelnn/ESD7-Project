#!/bin/bash

#Make venv with packages
uv sync

#run app
cd src
uv run main.py


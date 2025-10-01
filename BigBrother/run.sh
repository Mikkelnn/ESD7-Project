#!/bin/bash

#Make venv with packages
uv sync

#run app
uv run src/main.py

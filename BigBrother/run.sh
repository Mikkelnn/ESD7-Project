#!/bin/bash

#Make venv with packages
uv sync

#run ap
uv run src/main.py
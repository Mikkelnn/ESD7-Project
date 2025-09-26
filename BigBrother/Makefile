
docker_run:
	docker build -f Dockerfile -t bigbrother . && docker run --runtime=nvidia --gpus all bigbrother:latest

run:
	uv run src/main.py

lint:
	uvx ruff check src/ --fix && uvx ty check src/
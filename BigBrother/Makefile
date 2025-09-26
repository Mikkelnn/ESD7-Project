
docker_run:
	docker build -f Dockerfile -t bigbrother . && docker run --gpus all bigbrother:latest

run:
	uv run src/main.py

lint:
	uvx ruff check src/ && ty check src/
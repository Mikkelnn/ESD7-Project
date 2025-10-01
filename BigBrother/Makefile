docker_run:
	docker build -f Dockerfile -t bigbrother . && \
	docker run --runtime=nvidia --gpus all -v $(PWD):/app bigbrother:latest

run:
	uv sync && cd src && uv run main.py && cd ..

lint:
	uvx ruff check src/ && uvx ty check src/
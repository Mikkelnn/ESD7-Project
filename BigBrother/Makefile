docker_run:
	docker build -f Dockerfile -t bigbrother . && \
	docker run --runtime=nvidia --gpus all -v $(PWD):/app bigbrother:latest

run:
	uv sync && cd src && uv run main.py && cd ..

run_ai_lab:
	./run-on-ai-lab.sh

lint:
	uvx ruff check src/ && uvx ty check src/
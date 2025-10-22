run_docker:
	docker build -f docker/Dockerfile -t bigbrother . && \
	docker run --runtime=nvidia --gpus all -v ../$(PWD):/app bigbrother:latest

run_ai_lab:
	./run-on-ai-lab.sh

run:
	uv sync && cd src && uv run main.py && cd ..

lint:
	uvx ruff check src/ && uvx ty check src/
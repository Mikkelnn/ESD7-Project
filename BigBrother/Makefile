docker_run:
	docker build -f Dockerfile -t bigbrother . && \
	docker run --runtime=nvidia --gpus all -v $(PWD):/app bigbrother:latest

run:
	uv run src/main.py

lint:
	uvx ruff check src/ && uvx ty check src/

data_gen:
	uv run gen_test_data.py
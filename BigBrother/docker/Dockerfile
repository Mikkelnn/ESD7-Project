FROM nvidia/cuda:12.6.2-runtime-ubuntu24.04

WORKDIR /app

#in docker folder because run by Makefile outside of docker folder
COPY docker/apt_requirements.txt . 

RUN apt-get update && \
    xargs -a apt_requirements.txt apt-get install -y --no-install-recommends && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

CMD ["make", "run"]
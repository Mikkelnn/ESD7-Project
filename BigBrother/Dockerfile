FROM nvidia/cuda:12.6.2-runtime-ubuntu24.04

WORKDIR /home/user/BigBrother

COPY apt_requirements.txt .

RUN apt-get update && \
    xargs -a apt_requirements.txt apt-get install -y --no-install-recommends && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN useradd -m user

USER user

COPY . .

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/home/user/.local/bin:$PATH"

CMD ["sh", "/home/user/BigBrother/run.sh"]
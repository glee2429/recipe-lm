FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

# git is required by HuggingFace Spaces build hooks
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Non-root user required by HuggingFace Spaces
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Optimise CPU inference for 8 vCPU hardware
ENV OMP_NUM_THREADS=8
ENV MKL_NUM_THREADS=8

WORKDIR /home/user/app

# Install pre-built llama-cpp-python wheel (no compilation)
RUN pip install --no-cache-dir \
        https://huggingface.co/ClaireLee2429/gemma-2b-recipes-gguf/resolve/main/llama_cpp_python-0.3.2-cp311-cp311-linux_x86_64.whl
RUN pip install --no-cache-dir \
        fastapi \
        "uvicorn[standard]" \
        sse-starlette \
        huggingface_hub \
        httpx

COPY --chown=user:user server.py .
COPY --chown=user:user inference.py .

EXPOSE 7860

CMD ["python", "server.py", "--port", "7860"]

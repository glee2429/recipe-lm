FROM python:3.11-alpine

# git for HF Spaces, C++ runtime libs for llama-cpp-python
RUN apk add --no-cache git libstdc++ libgomp libgcc

# Non-root user required by HuggingFace Spaces
RUN adduser -D -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Optimise CPU inference for 8 vCPU hardware
ENV OMP_NUM_THREADS=8
ENV MKL_NUM_THREADS=8

WORKDIR /home/user/app

# Install pre-built llama-cpp-python wheel (musl/Alpine compatible)
RUN pip install --no-cache-dir \
        https://huggingface.co/ClaireLee2429/gemma-2b-recipes-gguf/resolve/main/llama_cpp_python-0.3.2-cp311-cp311-linux_x86_64.whl
RUN pip install --no-cache-dir \
        fastapi \
        "uvicorn[standard]" \
        sse-starlette \
        huggingface_hub \
        httpx \
        gradio

COPY --chown=user:user server.py .
COPY --chown=user:user inference.py .

EXPOSE 7860

CMD ["python", "server.py", "--port", "7860"]

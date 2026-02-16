FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive

# Non-root user required by HuggingFace Spaces
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /home/user/app

# Install serving dependencies (torch already in base image)
RUN pip install --no-cache-dir \
        transformers \
        peft \
        accelerate \
        fastapi \
        "uvicorn[standard]" \
        sse-starlette \
        huggingface_hub \
        httpx

COPY --chown=user:user server.py .
COPY --chown=user:user inference.py .

EXPOSE 7860

CMD ["python", "server.py", \
     "--adapter", "ClaireLee2429/gemma-2b-recipes-lora", \
     "--port", "7860"]

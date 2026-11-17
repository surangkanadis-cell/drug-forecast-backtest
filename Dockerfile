# Hugging Face Space (Docker SDK): Streamlit web app + Chronos-2 on CPU
FROM python:3.12-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    TORCH_COMPILE_DISABLE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY --chown=user requirements-space.txt .
RUN pip install --no-cache-dir --user -r requirements-space.txt

# Download the model at build time so the app starts fast
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('amazon/chronos-2')"

COPY --chown=user . .
RUN mkdir -p data outputs

EXPOSE 7860
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", \
     "--server.headless=true", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]

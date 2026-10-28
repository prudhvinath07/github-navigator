FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

# gitingest clones repositories with git
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs containers as user 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    TIKTOKEN_CACHE_DIR=/home/user/.cache/tiktoken
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt \
    && python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

COPY --chown=user . .

EXPOSE 7860

# One worker so the in-memory sessions are shared; threads for concurrency.
# Long timeout because ingesting a large repository can take a while.
CMD gunicorn --bind 0.0.0.0:${PORT} --workers 1 --threads 8 --timeout 300 app:app

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY openenv.yaml ./
COPY pyproject.toml ./
COPY env.py ./
COPY tasks.py ./
COPY inference.py ./
COPY README.md ./
COPY server ./server

EXPOSE 8000

CMD ["python", "-m", "server.app"]

FROM python:3.10-slim

RUN python3 -m pip install poetry==1.8.0

COPY poetry.lock .
COPY pyproject.toml .

RUN poetry config virtualenvs.create false && \
    poetry install --no-root --only model

RUN poetry config virtualenvs.create false && \
    poetry install --no-root --with base,scheduler

WORKDIR /app

COPY eye_dota/ .

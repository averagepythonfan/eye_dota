FROM python:3.10-slim

RUN python3 -m pip install poetry==1.8.0

COPY poetry.lock .
COPY pyproject.toml .

RUN poetry config virtualenvs.create false && \
    poetry install --no-root --only frontend

RUN python3 -m pip install pydantic

WORKDIR /app

COPY frontend/ .
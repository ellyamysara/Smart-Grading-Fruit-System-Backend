FROM PYTHON:3.11.11-slim

workdir /app

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONBUFFERED 1

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

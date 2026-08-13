FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./
COPY engine ./engine
COPY envs ./envs
COPY agents ./agents
COPY cli ./cli
COPY api ./api
COPY static ./static

RUN pip install --no-cache-dir ".[web]"

EXPOSE 7860

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]

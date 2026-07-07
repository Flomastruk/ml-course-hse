FROM python:3.12-trixie

WORKDIR /workspace

# RUN pip install --upgrade pip
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/

# COPY ../reqs/requirements.txt reqs/
COPY pyproject.toml ./

ENV UV_PROJECT_ENVIRONMENT=/.venv

# RUN pip install -r reqs/requirements.txt
# RUN uv pip install --system -r reqs/requirements.txt --no-cache
RUN uv sync --no-dev

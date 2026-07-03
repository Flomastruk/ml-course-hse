FROM python:3.12-trixie

# RUN pip install --upgrade pip
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/

# COPY ../reqs/requirements.txt reqs/
COPY pyproject.toml uv.lock ./

# RUN pip install -r reqs/requirements.txt
# RUN uv pip install --system -r reqs/requirements.txt --no-cache
RUN uv sync --frozen --no-dev

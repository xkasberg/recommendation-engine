FROM python:3.10-slim

# Install uv (official way)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV UV_LINK_MODE=copy

# Tell uv to place the project environment here (NOT under /app/api)
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR /app/api

# 1) Resolve & cache deps (no project install yet)
COPY api/pyproject.toml /app/api/pyproject.toml

COPY api/uv.lock /app/api/uv.lock
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

# 2) Copy code and install the project into /opt/venv
COPY api/ /app/api/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen


EXPOSE 8080
# Use FastAPI CLI; see compose below for --reload, etc.
CMD ["fastapi", "run", "app.py", "--host", "0.0.0.0", "--port", "8080"]
FROM python:3.10-slim

# Install uv (official way)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV UV_LINK_MODE=copy

# Put the venv in a stable location and on PATH
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR /app/api

# ---- 1) Install and cache ONLY third-party dependencies ----
# Copy lockfile + pyproject and sync without installing the local project
COPY api/pyproject.toml /app/api/pyproject.toml
COPY api/uv.lock /app/api/uv.lock
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

# ---- 2) Install your project WITHOUT deps (re-uses the venv above) ----
# Copy the rest of the source and install only the project code
COPY api/ /app/api/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --no-deps -e .

EXPOSE 8080
CMD ["fastapi", "run", "app.py", "--host", "0.0.0.0", "--port", "8080"]
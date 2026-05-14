# Stage 1: Build React frontend
FROM node:20-alpine AS frontend-builder

WORKDIR /build
COPY frontend/package*.json ./
RUN npm ci --prefer-offline

COPY frontend/ .
# Empty VITE_API_URL means relative URLs — same host as backend
ARG VITE_API_URL=""
ENV VITE_API_URL=$VITE_API_URL
RUN npm run build

# Stage 2: Python runtime with embedded frontend
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY requirements-backend.txt ./
RUN pip install --no-cache-dir -r requirements-backend.txt

COPY backend/ ./backend/
COPY agents/ ./agents/
COPY quantum/ ./quantum/
COPY risk/ ./risk/
COPY coordinator/ ./coordinator/
COPY evaluation/ ./evaluation/
COPY trading/ ./trading/
COPY envs/ ./envs/
COPY __init__.py ./

COPY --from=frontend-builder /build/dist ./frontend/dist

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --loop uvloop"]

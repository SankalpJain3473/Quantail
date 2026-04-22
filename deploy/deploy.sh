#!/bin/bash
# deploy/deploy.sh
# One-command deployment for Quantail
# Usage: ./deploy/deploy.sh [dev|prod]

set -euo pipefail

MODE=${1:-dev}
echo "======================================================"
echo "  QUANTAIL DEPLOYMENT — mode: $MODE"
echo "  Sankalp Jain & Veronica Koval | Columbia University"
echo "======================================================"

# Check dependencies
command -v docker >/dev/null 2>&1 || { echo "Docker not found. Install from https://docker.com"; exit 1; }
command -v docker-compose >/dev/null 2>&1 || command -v docker compose >/dev/null 2>&1 || { echo "Docker Compose not found."; exit 1; }

# Check .env
if [ ! -f .env ]; then
    echo ""
    echo "⚠  No .env file found. Creating from .env.example..."
    cp .env.example .env
    echo "✓  Created .env — please edit it with your secrets before production use"
    echo ""
fi

if [ "$MODE" = "prod" ]; then
    # Production checks
    SECRET=$(grep QUANTAIL_SECRET_KEY .env | cut -d= -f2)
    if [ "$SECRET" = "change-this-to-a-random-64-char-string-in-production" ]; then
        echo "ERROR: Change QUANTAIL_SECRET_KEY in .env before deploying to production!"
        exit 1
    fi
    echo "✓  Security key configured"
fi

# Build and start
echo ""
echo "Building Docker images..."
docker-compose build --no-cache

echo ""
echo "Starting services..."
docker-compose up -d

echo ""
echo "Waiting for health checks..."
sleep 8

# Check backend health
BACKEND_HEALTH=$(curl -sf http://localhost:8000/health 2>/dev/null || echo "FAIL")
if echo "$BACKEND_HEALTH" | grep -q "ok"; then
    echo "✓  Backend healthy"
else
    echo "⚠  Backend not responding — check: docker-compose logs backend"
fi

# Check frontend
FRONTEND_STATUS=$(curl -sf -o /dev/null -w "%{http_code}" http://localhost:3000 2>/dev/null || echo "000")
if [ "$FRONTEND_STATUS" = "200" ]; then
    echo "✓  Frontend healthy"
else
    echo "⚠  Frontend not responding — check: docker-compose logs frontend"
fi

echo ""
echo "======================================================"
echo "  QUANTAIL RUNNING"
echo "  Dashboard:  http://localhost:3000"
echo "  API docs:   http://localhost:8000/api/docs"
echo "  Login:      demo / quantail2026"
echo "======================================================"
echo ""
echo "Useful commands:"
echo "  View logs:    docker-compose logs -f"
echo "  Stop:         docker-compose down"
echo "  Rebuild:      docker-compose up --build"

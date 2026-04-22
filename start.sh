#!/bin/bash
# Run both backend and frontend with one command
# All config comes from .env — no env vars needed in terminal

ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "Starting Quantail..."

# Backend
cd "$ROOT"
/opt/anaconda3/bin/uvicorn backend.main:app --reload --port 8000 &
BACKEND_PID=$!

# Frontend
cd "$ROOT/frontend"
npm run dev &
FRONTEND_PID=$!

echo "Backend  → http://localhost:8000"
echo "Frontend → http://localhost:3002 (or 3000/3001 if available)"
echo ""
echo "Press Ctrl+C to stop both"

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; echo 'Stopped.'" INT TERM
wait

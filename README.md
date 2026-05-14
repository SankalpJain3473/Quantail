# Quantail - Distributional Quantum RL for Dynamic Hedging
**Founders:** Sankalp Jain & Veronica Koval | Columbia University 
Link - https://perpetual-truth-production-8a5f.up.railway.app
Username: admin
Password: Quantail2026!

## Results
| Metric | Quantail | Delta Hedge | Improvement |
|--------|----------|-------------|-------------|
| Hedging RMSE | 0.9134 | 1.0502 | +13.0% |
| CVaR @ 95% | 1.3490 | 1.7269 | +21.9% |
| Transaction cost | lower | baseline | +10.0% |

## Quick Start
```bash
# Docker (one command)
cp .env.example .env && ./deploy/deploy.sh
# → http://localhost:3000 | Login: demo / quantail2026

# Local dev
pip install -r requirements.txt -r requirements-backend.txt
uvicorn backend.main:app --reload --port 8000
cd frontend && npm install && npm run dev
```

## Stack
- **Backend:** FastAPI + WebSocket + JWT auth + Pydantic validation
- **Frontend:** React 18 + TypeScript + Vite + Tailwind + Recharts + Zustand
- **ML:** QR-DQN distributional RL + VQC (PennyLane) + 4 agents + Wasserstein coordinator
- **Deploy:** Docker + Nginx + Redis

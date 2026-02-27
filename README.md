# 🐺 Market Signal Dashboard

Contrarian signal dashboard based on 15yr backtest (2010–2026).

**Confluence signals (3-factor):** AAII Bull-Bear Spread + NAAIM + VIX  
**Macro gate (6-factor):** Rates + DXY + Copper + HY Spread + Yield Curve + SPY > 200MA  
**Standalone:** Fear & Greed, Insider Cluster

## Run Locally

```bash
cd signal_dashboard
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud (Free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account → select repo → set main file: `signal_dashboard/app.py`
4. Add secrets (Settings → Secrets):

```toml
FRED_API_KEY = "your_key_here"
```

Get a free FRED API key at: https://fred.stlouisfed.org/docs/api/api_key.html (takes 2 min)

## Deploy to Render.com

1. Push to GitHub
2. New Web Service → connect repo
3. Build command: `pip install -r signal_dashboard/requirements.txt`
4. Start command: `streamlit run signal_dashboard/app.py --server.port $PORT --server.address 0.0.0.0`
5. Add env vars: `FRED_API_KEY`

## Secrets / Env Vars

| Variable | Required | Description |
|---|---|---|
| `FRED_API_KEY` | Recommended | For real yield curve & HY spread data. Free at fred.stlouisfed.org |
| `TELEGRAM_BOT_TOKEN` | For alerts only | Your OpenClaw bot token |
| `TELEGRAM_CHAT_ID` | For alerts only | Your Telegram chat ID |

## Weekly Manual Updates

AAII and NAAIM publish weekly (AAII: Thursday, NAAIM: Wednesday).  
Use the sidebar overrides to input current week values, or update the local CSV files.

## Backtest Reference

| Signal | N | 3m WR | 12m WR | 12m Avg |
|--------|---|-------|--------|---------|
| VIX > 50 | 19 | 100% | 100% | +57.6% |
| VIX > 40 | 54 | 98% | 100% | +44.2% |
| NAAIM < 25 | 112 | 94% | 100% | +23.3% |
| Confluence ≥ 2 + Macro | 43 | 91% | 100% | +33.1% |
| Confluence ≥ 3 | 25 | 96% | 96% | +18.5% |

#!/bin/bash
# Run this script once to register the daily signal checker cron job
# It will run at 8:00 AM Bangkok time every day

openclaw cron add \
  --name "Daily Market Signal Check" \
  --cron "0 8 * * *" \
  --tz "Asia/Bangkok" \
  --session isolated \
  --message "Run the market signal checker:

python3 /Users/alpha/.openclaw/workspace/signal_dashboard/signal_checker.py

Parse the output carefully:
- If confluence score is ≥ 2 (2 or 3 of AAII + NAAIM + VIX signals firing), send a Telegram alert with full details
- If VIX is above 40 OR NAAIM below 25 as standalone extremes, also alert
- If no significant signals (confluence 0-1, no extremes), reply HEARTBEAT_OK — no alert needed

Format the alert clearly with signal status, confluence score, macro gate score, and what it historically means." \
  --announce \
  --channel telegram \
  --to "372498084"

echo "Cron job registered. Verify with: openclaw cron list"

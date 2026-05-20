# NZX Mean Reversion Dashboard

Jim Simons–style quant terminal for NZ listed companies. Live data from Yahoo Finance (free tier).

**Public URL:** https://nzx-dashboard.streamlit.app

## Metrics

| Metric | Description |
|--------|-------------|
| Z-Score | Deviation from 20-day rolling mean (standard deviations) |
| RSI-14 | Relative Strength Index — <30 oversold, >70 overbought |
| Bollinger %B | Position within 20-day Bollinger Bands |
| Dev20/Dev50 | % price deviation from 20/50 day moving average |
| 52wk High/Low Dist | % below 52-week high / above 52-week low |
| Vol Ratio | 20-day volume vs average volume |

## Colour Key

- 🟢 Green = Oversold / mean reversion buy signal
- 🟡 Amber = Mild premium / slightly rich
- 🔴 Red = Overbought / expensive

## Notes

- Data: Yahoo Finance (no API key needed)
- Z-score uses 20-day window due to 3-month free tier data limit
- Refreshes hourly (Streamlit cloud cache)
- For longer Z-score windows (63-day quarterly), a paid data source (e.g. NZXplorer API) is needed
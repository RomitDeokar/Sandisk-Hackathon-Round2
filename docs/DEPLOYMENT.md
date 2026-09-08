# Deployment Guide — WaferFusion-Cascade Dashboard

The dashboard is a Streamlit app that loads persisted artifacts from `outputs/`
and performs **no retraining**.

## Local / sandbox run

```bash
pip install -r requirements.txt
streamlit run dashboard/app.py
```

`.streamlit/config.toml` binds `0.0.0.0:8501` in headless mode and relaxes
CORS/XSRF so the app is reachable through a reverse proxy or tunnel.

## Process supervision

`supervisord.conf` keeps the dashboard and the Cloudflare tunnel alive with
automatic restarts:

```bash
supervisord -c supervisord.conf          # start both services
supervisorctl -c supervisord.conf status # check state
supervisorctl -c supervisord.conf restart streamlit
```

## Cloudflare exposure

Streamlit is a stateful Python process that requires a long-lived WebSocket
connection, so it cannot run on Cloudflare Pages or Workers (both are
stateless and cannot execute Python, PyTorch or LightGBM). It is instead served
through the Cloudflare edge with a `cloudflared` tunnel, which proxies
WebSockets correctly:

```bash
cloudflared tunnel --url http://localhost:8501
```

For a permanent, named Cloudflare hostname, either
1. add a zone (custom domain) to the account and create a named tunnel, or
2. enable the Workers Paid plan to use Cloudflare Containers.

## Dependencies

`requirements.txt` covers the full pipeline. The dashboard alone needs
`streamlit`, `pandas`, `numpy`, `matplotlib`, `scikit-learn`, plus `torch` and
`lightgbm` (imported via the `sandisk_yield.training` package chain).

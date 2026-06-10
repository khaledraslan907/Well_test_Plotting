
"""
FastAPI webhook receiver for WhatsApp Cloud API.

Run:
    uvicorn whatsapp_webhook_fastapi:app --host 0.0.0.0 --port 8000

Expose while testing:
    ngrok http 8000
"""

import os
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from fastapi import FastAPI, Request, Response

from tmu_parser import parse_tmu_message

VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN", "change-me")
DATA_PATH = Path(os.getenv("TMU_WHATSAPP_CSV", "whatsapp_tmu_data.csv"))

app = FastAPI(title="TMU WhatsApp Webhook")


@app.get("/webhook")
async def verify_webhook(request: Request):
    params = dict(request.query_params)
    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")

    if mode == "subscribe" and token == VERIFY_TOKEN:
        return Response(content=challenge or "", media_type="text/plain")
    return Response(status_code=403)


def extract_text_messages(payload: Dict[str, Any]) -> List[str]:
    texts = []
    for entry in payload.get("entry", []):
        for change in entry.get("changes", []):
            value = change.get("value", {})
            for msg in value.get("messages", []) or []:
                if msg.get("type") == "text":
                    body = (msg.get("text") or {}).get("body", "")
                    if body:
                        texts.append(body)
    return texts


@app.post("/webhook")
async def receive_webhook(request: Request):
    payload = await request.json()
    messages = extract_text_messages(payload)

    rows = []
    for body in messages:
        row = parse_tmu_message(body, source_name="WhatsApp_Cloud_API")
        if any(k in row for k in ["gross_rate_bpd", "oil_rate_stbd", "water_rate_bpd", "whp_psi", "pumping_pressure_psi"]):
            rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        if DATA_PATH.exists():
            old = pd.read_csv(DATA_PATH)
            df = pd.concat([old, df], ignore_index=True, sort=False)
        df.to_csv(DATA_PATH, index=False)

    return {"status": "ok", "parsed_reports": len(rows)}

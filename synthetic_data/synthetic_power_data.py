#!/usr/bin/env python3
"""Generate synthetic power data and POST to /data/update_power_generation_data.

Reads `energy_sources` from the local database (uses `get_db_config()` from `db_operations.py`).
Builds a JSON array of points: { timestamp, device_id, power } and POSTs it to the app.

Usage:
  python synthetic_power_data.py         # runs continuously every minute
  python synthetic_power_data.py --once  # run a single iteration (useful for testing)
"""
from __future__ import annotations
import os
import sys
import time
import random
import argparse
from datetime import datetime
import json

import requests
import psycopg2
from psycopg2.extras import RealDictCursor

# support running from synthetic_data/ by adding repo root to sys.path
try:
    from db_operations import get_db_config
except Exception:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from db_operations import get_db_config


def fetch_sources():
    cfg = get_db_config()
    conn = psycopg2.connect(**cfg, cursor_factory=RealDictCursor)
    try:
        cur = conn.cursor()
        cur.execute("SELECT source_id, source_type, capacity_kw FROM energy_sources")
        return [r for r in cur.fetchall()]
    finally:
        conn.close()


def efficiency_range_for_type(source_type: str) -> tuple[float, float]:
    """Return (min, max) efficiency multiplier for a source type.

    These are heuristics to make the generated power realistic:
      - PV: 0%..60% of capacity (sunlight variability)
      - Wind: 10%..80% (wind varies; rarely full capacity)
      - Battery: 0%..90% (charging/discharging patterns)
      - Biomass: 40%..95% (more stable baseload-like)
      - default: 0%..70%
    """
    t = (source_type or "").strip().lower()
    if "pv" in t or "solar" in t:
        return 0.0, 0.6
    if "wind" in t:
        return 0.1, 0.8
    if "battery" in t:
        return 0.0, 0.9
    if "biomass" in t:
        return 0.4, 0.95
    return 0.0, 0.7


def generate_payload(sources):
    ts = datetime.utcnow().isoformat() + "Z"
    points = []
    for s in sources:
        capacity = float(s.get("capacity_kw") or 0)
        src_type = s.get("source_type") or ""
        lo, hi = efficiency_range_for_type(src_type)
        # Pick a random efficiency and optionally add a small temporal autocorrelation
        eff = random.uniform(lo, hi)
        # Occasionally produce near-peak value (rare)
        if random.random() < 0.02:
            eff = min(1.0, eff + random.uniform(0.2, 0.4))
        power = round(capacity * eff, 4)
        points.append({"timestamp": ts, "device_id": int(s["source_id"]), "power": power})
    return points


def post_payload(
    payload, session_obj, endpoint="http://127.0.0.1:5000/data/update_power_generation_data"
):
    headers = {"Content-Type": "application/json"}
    try:
        resp = session_obj.post(endpoint, headers=headers, data=json.dumps(payload), timeout=10)
        resp.raise_for_status()
        print(f"Posted {len(payload)} points -> {resp.status_code}")
        try:
            print("Response:", resp.json())
        except Exception:
            print("Response text:", resp.text)
    except Exception as e:
        print("Failed to post payload:", e)


def create_session_and_login(base_login_url: str, username: str | None, password: str | None):
    s = requests.Session()
    if username and password:
        try:
            resp = s.post(
                base_login_url, json={"username": username, "password": password}, timeout=10
            )
            resp.raise_for_status()
            print("Logged in successfully; session cookie set")
        except Exception as e:
            print("Login failed:", e)
            raise
    return s


def main(loop: bool, endpoint: str, session_obj: requests.Session):
    print(f"Synthetic power generator starting (target endpoint: {endpoint})")
    while True:
        sources = fetch_sources()
        if not sources:
            print("No sources found in DB; sleeping 60s")
            if not loop:
                break
            time.sleep(60)
            continue

        payload = generate_payload(sources)
        post_payload(payload, session_obj=session_obj, endpoint=endpoint)

        if not loop:
            break
        # sleep until next minute
        time.sleep(60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Run single iteration and exit")
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:5000/data/update_power_generation_data",
        help="Full URL of the ingest endpoint",
    )
    parser.add_argument(
        "--login-url",
        default="http://127.0.0.1:5000/auth/login",
        help="Login URL to obtain session cookie",
    )
    parser.add_argument("--username", help="Optional username to login before posting")
    parser.add_argument("--password", help="Optional password to login before posting")
    args = parser.parse_args()
    # create session and optionally login
    session_obj = create_session_and_login(args.login_url, args.username, args.password)
    main(loop=not args.once, endpoint=args.endpoint, session_obj=session_obj)

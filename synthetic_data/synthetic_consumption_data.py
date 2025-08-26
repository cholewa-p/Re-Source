#!/usr/bin/env python3
"""Generate synthetic consumption readings and POST to /data/update_consumption_readings.

Reads `meter` list from the local database using `get_db_config()` from `db_operations.py`.
Builds a JSON array of points: { timestamp, meter_id, consumption } and POSTs it to the app.

Usage:
  python synthetic_consumption_data.py         # runs continuously every minute
  python synthetic_consumption_data.py --once  # run a single iteration (useful for testing)
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


def fetch_meters():
    cfg = get_db_config()
    conn = psycopg2.connect(**cfg, cursor_factory=RealDictCursor)
    try:
        cur = conn.cursor()
        cur.execute("SELECT meter_id, address_id, property_type, max_load_kw FROM consumption_meters")
        return [r for r in cur.fetchall()]
    finally:
        conn.close()


def consumption_variation(property_type: str) -> tuple[float, float]:
    """Return (min, max) fraction of max_load_kw to use for a reading.

    Residential: 10%..80% (daily cycles)
    Commercial: 20%..95% (higher peak usage)
    Default: 5%..90%
    """
    t = (property_type or "").strip().lower()
    if "res" in t:
        return 0.1, 0.8
    if "com" in t:
        return 0.2, 0.95
    return 0.05, 0.9


def generate_payload(meters):
    ts = datetime.utcnow().isoformat() + "Z"
    points = []
    for m in meters:
        max_load = float(m.get("max_load_kw") or 0)
        prop = m.get("property_type") or ""
        lo, hi = consumption_variation(prop)
        frac = random.uniform(lo, hi)
        # simulate time-of-day effect: higher around local midday/early evening
        tod = datetime.utcnow().hour
        if 17 <= tod <= 20:
            frac = min(1.0, frac + random.uniform(0.05, 0.25))
        consumption = round(max_load * frac, 4)
        points.append({"timestamp": ts, "meter_id": int(m["meter_id"]), "consumption": consumption})
    return points


def post_payload(payload, session_obj, endpoint="http://127.0.0.1:5000/data/update_consumption_readings"):
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
    print(f"Synthetic consumption generator starting (target endpoint: {endpoint})")
    while True:
        meters = fetch_meters()
        if not meters:
            print("No meters found in DB; sleeping 60s")
            if not loop:
                break
            time.sleep(60)
            continue

        payload = generate_payload(meters)
        post_payload(payload, session_obj=session_obj, endpoint=endpoint)

        if not loop:
            break
        time.sleep(60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Run single iteration and exit")
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:5000/data/update_consumption_readings",
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
    session_obj = create_session_and_login(args.login_url, args.username, args.password)
    main(loop=not args.once, endpoint=args.endpoint, session_obj=session_obj)

import os
import json
import logging
from datetime import datetime, timedelta
from configparser import ConfigParser
from typing import List, Dict, Any

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
from flask import request, jsonify, Blueprint, session

from timeseries_model import train_electricity_model, train_aggregated_model, ElectricityProductionModel
import numpy as np
import pandas as pd

# Ensure models directory exists
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

logger = logging.getLogger(__name__)
db_bp = Blueprint("db", __name__)


@db_bp.route("/update_power_generation_data", methods=["POST"])
def update_power_generation_data():
    """Ingest power generation data posted as JSON array.

    Expected payload (list of objects):
    [
      {"timestamp": "2025-08-24T12:00:00Z", "device_id": 42, "power": 1.23},
      ...
    ]
    """
    payload = request.get_json(silent=True)
    if not isinstance(payload, list):
        return jsonify({"error": "Payload must be a JSON array"}), 400

    try:
        cleaned: List[Dict[str, Any]] = []
        for i, row in enumerate(payload):
            if not isinstance(row, dict):
                raise ValueError(f"Element {i} not an object")
            missing = {k for k in ("timestamp", "device_id", "power") if k not in row}
            if missing:
                raise ValueError(f"Missing keys {missing} in element {i}")
            # Parse / validate fields
            ts_raw = row["timestamp"]
            if isinstance(ts_raw, (int, float)):
                # treat as epoch seconds
                ts = datetime.utcfromtimestamp(ts_raw)
            else:
                try:
                    ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
                except Exception:
                    raise ValueError(f"Invalid timestamp format in element {i}: {ts_raw}")
            device_id = int(row["device_id"])  # raises if invalid
            power = float(row["power"])  # raises if invalid
            cleaned.append({"time": ts, "device_id": device_id, "power": power})

        inserted = insert_timeseries_data(cleaned, get_db_config())
        logger.info("Inserted %d rows from IoT device batch", inserted)
        return jsonify({"message": "Operation successful", "rows": inserted}), 200
    except ValueError as ve:
        logger.warning("Validation error: %s", ve)
        return jsonify({"error": str(ve)}), 400
    except Exception:
        logger.exception("Failed to ingest time series batch")
        return jsonify({"error": "Internal server error"}), 500


@db_bp.route("/forecast", methods=["GET"])
def forecast():
    """Role-aware forecast endpoint.

    Modes:
      - Normal user: specify ?source_id=ID (must belong to user). If omit source_id => aggregate all user's sources.
      - Admin (session role == 'admin'): if source_id omitted => aggregate ALL sources; if provided => that single.

    Query params:
      source_id (optional int)
      horizon (int, default 24)
      start / end (ISO timestamps)
    """
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    role = session.get("role", "user")

    source_id_arg = request.args.get("source_id")  # legacy single-source param
    source_ids_arg = request.args.get("source_ids")  # comma-separated list
    horizon = request.args.get("horizon", default="24")
    start_arg = request.args.get("start")
    end_arg = request.args.get("end")

    try:
        horizon_int = int(horizon)
        if horizon_int <= 0 or horizon_int > 7 * 24:
            return jsonify({"error": "horizon must be between 1 and 168 hours"}), 400
        start_dt = datetime.fromisoformat(start_arg) if start_arg else None
        end_dt = datetime.fromisoformat(end_arg) if end_arg else None
        source_id = int(source_id_arg) if source_id_arg is not None else None
        source_ids = None
        if source_ids_arg:
            source_ids = [int(x) for x in source_ids_arg.split(",") if x.strip()]
            # Normalize: if only one provided treat as single
            if len(source_ids) == 1 and source_id is None:
                source_id = source_ids[0]
        if source_id and source_ids:
            return jsonify({"error": "Use either source_id or source_ids, not both"}), 400
    except ValueError:
        return jsonify({"error": "Invalid numeric or datetime format"}), 400

    db_cfg = get_db_config()
    conn = psycopg2.connect(**db_cfg, cursor_factory=RealDictCursor)
    try:
        cur = conn.cursor()
        if role == "admin":
            # Admin path
            if source_ids:
                # Validate each exists (optional performance cost)
                cur.execute(
                    "SELECT source_id FROM energy_sources WHERE source_id = ANY(%s)", (source_ids,)
                )
                found = {r["source_id"] for r in cur.fetchall()}
                missing = set(source_ids) - found
                if missing:
                    return jsonify({"error": f"Sources not found: {sorted(missing)}"}), 404
                model, fc = train_aggregated_model(
                    db_cfg, source_ids, start=start_dt, end=end_dt, horizon_hours=horizon_int
                )
                scope = f'sources_{"_".join(map(str, source_ids))}'
            elif source_id is not None:
                model, fc = train_electricity_model(
                    db_cfg, source_id, start=start_dt, end=end_dt, horizon_hours=horizon_int
                )
                scope = f"source_{source_id}"
            else:
                # all sources
                cur.execute("SELECT source_id FROM energy_sources")
                sid_list = [r["source_id"] for r in cur.fetchall()]
                model, fc = train_aggregated_model(
                    db_cfg, sid_list, start=start_dt, end=end_dt, horizon_hours=horizon_int
                )
                scope = "all_sources"
        else:
            # Normal user path
            username = session.get("username")
            if source_ids:
                cur.execute(
                    """
                    SELECT es.source_id FROM energy_sources es
                    JOIN addresses a ON es.address_id = a.address_id
                    JOIN clients c ON a.client_id = c.client_id
                    JOIN user_accounts ua ON c.client_id = ua.client_id
                    WHERE ua.username = %s AND es.source_id = ANY(%s)
                """,
                    (username, source_ids),
                )
                found = {r["source_id"] for r in cur.fetchall()}
                missing = set(source_ids) - found
                if missing:
                    return (
                        jsonify({"error": f"Sources not found or not owned: {sorted(missing)}"}),
                        404,
                    )
                model, fc = train_aggregated_model(
                    db_cfg, source_ids, start=start_dt, end=end_dt, horizon_hours=horizon_int
                )
                scope = f'user_sources_{"_".join(map(str, source_ids))}'
            elif source_id is not None:
                cur.execute(
                    """
                    SELECT 1 FROM energy_sources es
                    JOIN addresses a ON es.address_id = a.address_id
                    JOIN clients c ON a.client_id = c.client_id
                    JOIN user_accounts ua ON c.client_id = ua.client_id
                    WHERE ua.username = %s AND es.source_id = %s
                """,
                    (username, source_id),
                )
                if cur.fetchone() is None:
                    return jsonify({"error": "Source not found or not owned by user"}), 404
                model, fc = train_electricity_model(
                    db_cfg, source_id, start=start_dt, end=end_dt, horizon_hours=horizon_int
                )
                scope = f"source_{source_id}"
            else:
                # all user's sources
                cur.execute(
                    """
                    SELECT es.source_id FROM energy_sources es
                    JOIN addresses a ON es.address_id = a.address_id
                    JOIN clients c ON a.client_id = c.client_id
                    JOIN user_accounts ua ON c.client_id = ua.client_id
                    WHERE ua.username = %s
                """,
                    (username,),
                )
                sid_list = [r["source_id"] for r in cur.fetchall()]
                if not sid_list:
                    return jsonify({"error": "User has no sources"}), 404
                model, fc = train_aggregated_model(
                    db_cfg, sid_list, start=start_dt, end=end_dt, horizon_hours=horizon_int
                )
                scope = "user_all_sources"
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 404
    except Exception:
        logger.exception("Forecast generation failed")
        return jsonify({"error": "Internal server error"}), 500
    finally:
        conn.close()

    return (
        jsonify(
            {
                "scope": scope,
                "role": role,
                "horizon_hours": horizon_int,
                "forecast": {
                    "timestamp": [ts.isoformat() for ts in fc.index],
                    "power_kw": fc["forecast"].round(4).tolist(),
                    "lower": fc["lower"].round(4).tolist(),
                    "upper": fc["upper"].round(4).tolist(),
                },
            }
        ),
        200,
    )


@db_bp.route("/save_model", methods=["POST"])
def save_model():
    """Persist a trained model to disk and register it in the `models` table.

    Payload JSON:
      { "scope": "..." }
    """
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or "scope" not in payload:
        return jsonify({"error": "Missing payload or scope"}), 400
    scope = payload["scope"]

    try:
        db_cfg = get_db_config()
        model = None
        # Determine how to train based on scope
        if scope.startswith("source_"):
            sid = int(scope.split("_", 1)[1])
            model, _ = train_electricity_model(db_cfg, sid)
        elif scope.startswith("sources_") or scope.startswith("user_sources_"):
            ids = scope.split("_", 1)[1].split("_")
            ids = [int(x) for x in ids if x]
            model, _ = train_aggregated_model(db_cfg, ids)
        elif scope in ("user_all_sources", "all_sources"):
            conn = psycopg2.connect(**db_cfg, cursor_factory=RealDictCursor)
            try:
                cur = conn.cursor()
                if scope == "all_sources":
                    cur.execute("SELECT source_id FROM energy_sources")
                else:
                    username = session.get("username")
                    cur.execute(
                        """
                        SELECT es.source_id FROM energy_sources es
                        JOIN addresses a ON es.address_id = a.address_id
                        JOIN clients c ON a.client_id = c.client_id
                        JOIN user_accounts ua ON c.client_id = ua.client_id
                        WHERE ua.username = %s
                    """,
                        (username,),
                    )
                sid_list = [r["source_id"] for r in cur.fetchall()]
            finally:
                conn.close()
            model, _ = train_aggregated_model(db_cfg, sid_list)
        else:
            return jsonify({"error": "Unknown scope format"}), 400

        if model is None:
            return jsonify({"error": "Model could not be trained for saving"}), 500

        # Save into user-specific subdirectory models/{username}-{user_id}
        user_id = session.get("user_id")
        username = session.get("username")
        if user_id is None or username is None:
            return jsonify({"error": "Unauthorized"}), 401
        safe_user = str(username).replace(os.sep, "_")
        user_dir = os.path.join(MODELS_DIR, f"{safe_user}-{user_id}")
        os.makedirs(user_dir, exist_ok=True)

        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        safe_scope = str(scope).replace(os.sep, "_")
        model_dirname = f"model_{timestamp}_{safe_scope}"
        model_dir = os.path.join(user_dir, model_dirname)
        os.makedirs(model_dir, exist_ok=True)

        filename = f"model_{scope}_{timestamp}.pkl"
        safe_path = os.path.join(model_dir, filename)
        if not os.path.abspath(safe_path).startswith(os.path.abspath(MODELS_DIR)):
            return jsonify({"error": "Invalid filename"}), 400

        # Save model binary
        model.save(safe_path)

        # relative file/dir paths for returning and DB
        rel_dir = os.path.relpath(model_dir, os.path.dirname(__file__))
        rel_file = os.path.join(rel_dir, filename)

        # Persist record in models table with metadata
        try:
            conn2 = psycopg2.connect(**db_cfg, cursor_factory=RealDictCursor)
            try:
                with conn2:
                    cur2 = conn2.cursor()
                    metadata = {
                        "trained_by": {"account_id": user_id, "username": username},
                        "trained_at": datetime.utcnow().replace(tzinfo=None).isoformat() + "Z",
                        "scope": scope,
                        "filename": filename,
                        "file_path": os.path.join(rel_dir, filename),
                    }
                    cur2.execute(
                        """
                        INSERT INTO models (account_id, model_name, model_path, created_at, metadata)
                        VALUES (%s, %s, %s, %s, %s)
                        RETURNING model_id
                        """,
                        (user_id, filename, rel_dir, datetime.utcnow(), json.dumps(metadata)),
                    )
                    row = cur2.fetchone()
                    saved_model_id = row.get("model_id") if row else None
            finally:
                conn2.close()
        except Exception:
            logger.exception("Failed to insert model metadata record")
            # proceed: file is saved, but DB insert failed
            return (
                jsonify({"message": "Model saved to disk, but DB record failed", "path": rel_file}),
                200,
            )

        return (
            jsonify({"message": "Model saved", "path": rel_file, "model_id": saved_model_id}),
            200,
        )
    except Exception:
        logger.exception("Failed to save model")
        return jsonify({"error": "Internal server error"}), 500


def get_db_config(filename="config.ini", section="solar_data"):
    parser = ConfigParser()
    parser.read(filename)

    db_config = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            db_config[param[0]] = param[1]
    else:
        logger.exception(f"Section {section} not found in {filename}")
        raise Exception(f"Section {section} not found in {filename}")
    return db_config


def insert_timeseries_data(rows: List[Dict[str, Any]], db_config: Dict[str, Any]) -> int:
    """Insert batch of time series points.

    rows: list of {"time": datetime, "device_id": int, "power": float}
    Returns number of inserted rows.
    """
    if not rows:
        return 0
    conn = psycopg2.connect(**db_config)
    try:
        with conn, conn.cursor() as cur:
            execute_values(
                cur,
                "INSERT INTO power_generation (time, source_id, power_kw) VALUES %s ON CONFLICT DO NOTHING",
                [(r["time"], r["device_id"], r["power"]) for r in rows],
            )
        logger.info("Inserted %d rows successfully.", len(rows))
        return len(rows)
    finally:
        conn.close()


def insert_consumption_data(rows: List[Dict[str, Any]], db_config: Dict[str, Any]) -> int:
    """Insert batch of consumption readings.

    rows: list of {"timestamp": datetime, "meter_id": int, "consumption": float}
    Returns number of inserted rows.
    """
    if not rows:
        return 0
    conn = psycopg2.connect(**db_config)
    try:
        with conn, conn.cursor() as cur:
            execute_values(
                cur,
                "INSERT INTO consumption_readings (timestamp, meter_id, consumption_kw) VALUES %s ON CONFLICT DO NOTHING",
                [(r["timestamp"], r["meter_id"], r["consumption"]) for r in rows],
            )
        logger.info("Inserted %d consumption rows successfully.", len(rows))
        return len(rows)
    finally:
        conn.close()


@db_bp.route("/production", methods=["GET"])
def production():
    """Return recent production (power_generation) timeseries aggregated across sources.

    Query params:
      hours (int, default 24)
      source_ids (optional comma-separated list)

    Role behavior:
      - admin: can request any sources or omit to get all
      - normal user: when omitted returns user's sources; when provided, only allowed if owned
    """
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    role = session.get("role", "user")
    hours_arg = request.args.get("hours", "24")
    source_ids_arg = request.args.get("source_ids")
    try:
        hours = int(hours_arg)
        if hours <= 0 or hours > 7 * 24:
            return jsonify({"error": "hours must be between 1 and 168"}), 400
    except ValueError:
        return jsonify({"error": "Invalid hours parameter"}), 400

    db_cfg = get_db_config()
    conn = psycopg2.connect(**db_cfg, cursor_factory=RealDictCursor)
    try:
        cur = conn.cursor()
        # Determine source_id list based on role and provided params
        if role == "admin":
            if source_ids_arg:
                source_ids = [int(x) for x in source_ids_arg.split(",") if x.strip()]
            else:
                cur.execute("SELECT source_id FROM energy_sources")
                source_ids = [r["source_id"] for r in cur.fetchall()]
        else:
            username = session.get("username")
            if source_ids_arg:
                candidate = [int(x) for x in source_ids_arg.split(",") if x.strip()]
                cur.execute(
                    """
                    SELECT es.source_id FROM energy_sources es
                    JOIN addresses a ON es.address_id = a.address_id
                    JOIN clients c ON a.client_id = c.client_id
                    JOIN user_accounts ua ON c.client_id = ua.client_id
                    WHERE ua.username = %s AND es.source_id = ANY(%s)
                    """,
                    (username, candidate),
                )
                found = {r["source_id"] for r in cur.fetchall()}
                missing = set(candidate) - found
                if missing:
                    return (
                        jsonify({"error": f"Sources not found or not owned: {sorted(missing)}"}),
                        404,
                    )
                source_ids = candidate
            else:
                cur.execute(
                    """
                    SELECT es.source_id FROM energy_sources es
                    JOIN addresses a ON es.address_id = a.address_id
                    JOIN clients c ON a.client_id = c.client_id
                    JOIN user_accounts ua ON c.client_id = ua.client_id
                    WHERE ua.username = %s
                    """,
                    (username,),
                )
                source_ids = [r["source_id"] for r in cur.fetchall()]
                if not source_ids:
                    return jsonify({"error": "User has no sources"}), 404

        # Aggregate production by timestamp for the last `hours`
        cur.execute(
            "SELECT time AT TIME ZONE 'UTC' AS time_utc, SUM(power_kw) AS production "
            "FROM power_generation "
            "WHERE time >= NOW() - INTERVAL %s AND source_id = ANY(%s) "
            "GROUP BY time_utc ORDER BY time_utc ASC",
            (f"{hours} hours", source_ids),
        )
        rows = cur.fetchall()
        times = [r["time_utc"].isoformat() for r in rows]
        production_vals = [float(r["production"] or 0) for r in rows]
        return jsonify({"Time": times, "Production": production_vals}), 200
    finally:
        conn.close()


@db_bp.route("/usage", methods=["GET"])
def usage():
    """Return recent usage (consumption_readings) timeseries aggregated across meters.

    Query params:
      hours (int, default 24)
      meter_ids (optional comma-separated list)

    Role behavior mirrors /production but operates on meters.
    """
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    role = session.get("role", "user")
    hours_arg = request.args.get("hours", "24")
    meter_ids_arg = request.args.get("meter_ids")
    try:
        hours = int(hours_arg)
        if hours <= 0 or hours > 7 * 24:
            return jsonify({"error": "hours must be between 1 and 168"}), 400
    except ValueError:
        return jsonify({"error": "Invalid hours parameter"}), 400

    db_cfg = get_db_config()
    conn = psycopg2.connect(**db_cfg, cursor_factory=RealDictCursor)
    try:
        cur = conn.cursor()
        if role == "admin":
            if meter_ids_arg:
                meter_ids = [int(x) for x in meter_ids_arg.split(",") if x.strip()]
            else:
                cur.execute("SELECT meter_id FROM consumption_meters")
                meter_ids = [r["meter_id"] for r in cur.fetchall()]
        else:
            username = session.get("username")
            if meter_ids_arg:
                candidate = [int(x) for x in meter_ids_arg.split(",") if x.strip()]
                # validate ownership via users -> clients -> addresses -> meters join
                cur.execute(
                    """
                    SELECT cm.meter_id FROM consumption_meters cm
                    JOIN addresses a ON cm.address_id = a.address_id
                    JOIN clients c ON a.client_id = c.client_id
                    JOIN user_accounts ua ON c.client_id = ua.client_id
                    WHERE ua.username = %s AND cm.meter_id = ANY(%s)
                    """,
                    (username, candidate),
                )
                found = {r["meter_id"] for r in cur.fetchall()}
                missing = set(candidate) - found
                if missing:
                    return (
                        jsonify({"error": f"Meters not found or not owned: {sorted(missing)}"}),
                        404,
                    )
                meter_ids = candidate
            else:
                cur.execute(
                    """
                    SELECT cm.meter_id FROM consumption_meters cm
                    JOIN addresses a ON cm.address_id = a.address_id
                    JOIN clients c ON a.client_id = c.client_id
                    JOIN user_accounts ua ON c.client_id = ua.client_id
                    WHERE ua.username = %s
                    """,
                    (username,),
                )
                meter_ids = [r["meter_id"] for r in cur.fetchall()]
                if not meter_ids:
                    return jsonify({"error": "User has no meters"}), 404

        cur.execute(
            "SELECT timestamp AT TIME ZONE 'UTC' AS time_utc, SUM(consumption_kw) AS usage "
            "FROM consumption_readings "
            "WHERE timestamp >= NOW() - INTERVAL %s AND meter_id = ANY(%s) "
            "GROUP BY time_utc ORDER BY time_utc ASC",
            (f"{hours} hours", meter_ids),
        )
        rows = cur.fetchall()
        times = [r["time_utc"].isoformat() for r in rows]
        usage_vals = [float(r["usage"] or 0) for r in rows]
        return jsonify({"Time": times, "Usage": usage_vals}), 200
    finally:
        conn.close()



# Latest admin model endpoints removed: plotting and minutely forecast are deprecated.


@db_bp.route("/update_consumption_readings", methods=["POST"])
def update_consumption_readings():
    """Ingest consumption readings posted as JSON array.

    Expected payload (list of objects):
      [ {"timestamp": "2025-08-24T12:00:00Z", "meter_id": 5, "consumption": 1.23}, ... ]
    """
    payload = request.get_json(silent=True)
    if not isinstance(payload, list):
        return jsonify({"error": "Payload must be a JSON array"}), 400

    try:
        cleaned: List[Dict[str, Any]] = []
        for i, row in enumerate(payload):
            if not isinstance(row, dict):
                raise ValueError(f"Element {i} not an object")
            missing = {k for k in ("timestamp", "meter_id", "consumption") if k not in row}
            if missing:
                raise ValueError(f"Missing keys {missing} in element {i}")
            ts_raw = row["timestamp"]
            if isinstance(ts_raw, (int, float)):
                ts = datetime.utcfromtimestamp(ts_raw)
            else:
                try:
                    ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
                except Exception:
                    raise ValueError(f"Invalid timestamp format in element {i}: {ts_raw}")
            meter_id = int(row["meter_id"])  # raises if invalid
            consumption = float(row["consumption"])  # raises if invalid
            cleaned.append({"timestamp": ts, "meter_id": meter_id, "consumption": consumption})

        inserted = insert_consumption_data(cleaned, get_db_config())
        logger.info("Inserted %d consumption rows from synthetic generator", inserted)
        return jsonify({"message": "Operation successful", "rows": inserted}), 200
    except ValueError as ve:
        logger.warning("Validation error: %s", ve)
        return jsonify({"error": str(ve)}), 400
    except Exception:
        logger.exception("Failed to ingest consumption batch")
        return jsonify({"error": "Internal server error"}), 500

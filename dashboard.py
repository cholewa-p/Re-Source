from flask import session, render_template, Blueprint, jsonify
import datetime
import psycopg2
from psycopg2.extras import RealDictCursor
from db_operations import get_db_config

# SOURCES = [
#     {"name": "Solar Panels", "power": 300},
#     {"name": "Batteries", "power": 500},
#     {"name": "Water Power Plant", "power": 300}
# ]
sources = []


def get_sources_data():
    """Populate global sources list according to current user's role.

    - Normal user: only sources owned by the user (joined via username)
    - Admin: all sources in the system
    """
    sources.clear()
    role = session.get("role", "user")
    username = session.get("username")
    if username is None:
        return  # not logged in
    conn = psycopg2.connect(**get_db_config(), cursor_factory=RealDictCursor)
    try:
        cur = conn.cursor()
        if role == "admin":
            cur.execute(
                "SELECT es.source_id, es.source_type, es.capacity_kw, a.street, a.city "
                "FROM energy_sources es "
                "JOIN addresses a ON es.address_id = a.address_id "
                "JOIN clients c ON a.client_id = c.client_id "
                "JOIN user_accounts ua ON c.client_id = ua.client_id"
            )
        else:
            cur.execute(
                "SELECT es.source_id, es.source_type, es.capacity_kw, a.street, a.city "
                "FROM energy_sources es "
                "JOIN addresses a ON es.address_id = a.address_id "
                "JOIN clients c ON a.client_id = c.client_id "
                "JOIN user_accounts ua ON c.client_id = ua.client_id "
                "WHERE ua.username = %s",
                (username,),
            )
        for row in cur:
            sources.append(
                {
                    "source_id": row["source_id"],
                    "source_type": row["source_type"],
                    "capacity_kw": row["capacity_kw"],
                    "street": row["street"],
                    "city": row["city"],
                }
            )
    finally:
        conn.close()


dashboard_bp = Blueprint("dashboard", __name__)


@dashboard_bp.route("/get_meters_data", methods=["GET"])
def get_meters_data():
    """Return list of consumption meters with owner/address and last reading.

    JSON: { "meters": [ { "meter_id": int, "address": str, "owner": str,
                         "last_timestamp": iso, "last_consumption": float }, ... ] }
    """
    # Require authenticated user
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    role = session.get("role", "user")
    username = session.get("username")

    db_cfg = get_db_config()
    conn = psycopg2.connect(**db_cfg, cursor_factory=RealDictCursor)
    try:
        cur = conn.cursor()
        if role == "admin":
            cur.execute(
                """
                SELECT
                  cm.meter_id,
                  cm.max_load_kw,
                  COALESCE(a.street, '') || ', ' || COALESCE(a.city, '') AS address,
                  COALESCE(ua.username, '') AS owner,
                  (SELECT cr.timestamp AT TIME ZONE 'UTC' FROM consumption_readings cr
                   WHERE cr.meter_id = cm.meter_id
                   ORDER BY cr.timestamp DESC LIMIT 1) AS last_timestamp,
                  (SELECT cr.consumption_kw FROM consumption_readings cr
                   WHERE cr.meter_id = cm.meter_id
                   ORDER BY cr.timestamp DESC LIMIT 1) AS last_consumption
                FROM consumption_meters cm
                LEFT JOIN addresses a ON cm.address_id = a.address_id
                LEFT JOIN clients c ON a.client_id = c.client_id
                LEFT JOIN user_accounts ua ON c.client_id = ua.client_id
                ORDER BY cm.meter_id
                """
            )
            rows = cur.fetchall()
        else:
            # normal user: only their meters
            cur.execute(
                """
                SELECT
                  COALESCE(a.street, '') || ', ' || COALESCE(a.city, '') AS address,
                  cm.max_load_kw,
                  (SELECT cr.timestamp AT TIME ZONE 'UTC' FROM consumption_readings cr
                   WHERE cr.meter_id = cm.meter_id
                   ORDER BY cr.timestamp DESC LIMIT 1) AS last_timestamp,
                  (SELECT cr.consumption_kw FROM consumption_readings cr
                   WHERE cr.meter_id = cm.meter_id
                   ORDER BY cr.timestamp DESC LIMIT 1) AS last_consumption
                FROM consumption_meters cm
                JOIN addresses a ON cm.address_id = a.address_id
                JOIN clients c ON a.client_id = c.client_id
                JOIN user_accounts ua ON c.client_id = ua.client_id
                WHERE ua.username = %s
                ORDER BY cm.meter_id
                """,
                (username,),
            )
            rows = cur.fetchall()

        meters = []
        for r in rows:
            if role == "admin":
                meters.append(
                    {
                        "meter_id": r["meter_id"],
                        "address": r["address"],
                        "owner": r["owner"],
                        "max_load_kw": float(r["max_load_kw"]) if r.get("max_load_kw") is not None else None,
                        "last_timestamp": r["last_timestamp"].isoformat() if r["last_timestamp"] else None,
                        "last_consumption": float(r["last_consumption"]) if r["last_consumption"] is not None else None,
                    }
                )
            else:
                meters.append(
                    {
                        "address": r["address"],
                        "max_load_kw": float(r["max_load_kw"]) if r.get("max_load_kw") is not None else None,
                        "last_timestamp": r["last_timestamp"].isoformat() if r.get("last_timestamp") else None,
                        "last_consumption": float(r["last_consumption"]) if r.get("last_consumption") is not None else None,
                    }
                )
        return jsonify({"meters": meters}), 200
    finally:
        conn.close()


# Synthetic generator removed — plotting is performed by fetching
# real data from the database via the `/data/usage` and `/data/production` endpoints


@dashboard_bp.route("/dashboard")
def index():
    username = session.get("username", "Guest")
    get_sources_data()
    return render_template("dashboard.html", sources=sources, username=username)

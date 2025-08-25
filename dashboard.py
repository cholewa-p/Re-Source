from flask import session, render_template, Blueprint, jsonify
import datetime
import random
import pandas as pd
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


def generate_data():
    now = datetime.datetime.now()
    times = [now - datetime.timedelta(hours=i) for i in range(24)][::-1]
    usage = [round(random.uniform(0.5, 2.5), 2) for _ in times]
    production = [round(u * random.uniform(0.6, 1.1), 2) for u in usage]
    df = pd.DataFrame(
        {
            "Time": [t.strftime("%Y-%m-%d %H:%M:%S") for t in times],
            "Usage": usage,
            "Production": production,
        }
    )
    return df


@dashboard_bp.route('/data')
def data():
    df = generate_data()
    return jsonify(df.to_dict(orient='list'))


@dashboard_bp.route("/dashboard")
def index():
    username = session.get("username", "Guest")
    get_sources_data()
    return render_template("dashboard.html", sources=sources, username=username)

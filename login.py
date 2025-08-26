from flask import request, jsonify, session, Blueprint, render_template
import psycopg2
from psycopg2.extras import RealDictCursor
from werkzeug.security import check_password_hash
from db_operations import get_db_config
import logging

logger = logging.getLogger(__name__)
login_bp = Blueprint("login", __name__)


@login_bp.route("/login-page", methods=["GET"])
def login_page():
    return render_template("login.html")


@login_bp.route("/login", methods=["POST"])
def login():
    data = request.json
    if not data or "username" not in data or "password" not in data:
        return jsonify({"error": "Brak danych logowania"}), 400

    username = data["username"]
    password = data["password"]

    conn = psycopg2.connect(**get_db_config(), cursor_factory=RealDictCursor)
    cur = conn.cursor()
    cur.execute(
        "SELECT account_id, username, password_hash, role FROM user_accounts WHERE username = %s",
        (username,),
    )
    user = cur.fetchone()
    conn.close()
    if not user or not check_password_hash(user["password_hash"], password):
        return jsonify({"error": "Niepoprawny użytkownik lub hasło"}), 401

    # Tworzymy sesję dla zalogowanego użytkownika
    session.permanent = True
    session["user_id"] = user["account_id"]
    session["username"] = user["username"]
    session["role"] = (
        user.get("role", "user")
        if isinstance(user, dict)
        else user["role"] if "role" in user else "user"
    )
    logger.info("Succesfully logged in.")
    return jsonify({"message": "Zalogowano pomyślnie"}), 200


@login_bp.route("/logout", methods=["POST"])
def logout():
    session.clear()
    logger.info("Succesfully logged out.")
    return jsonify({"message": "Wylogowano"}), 200


@login_bp.route("/me", methods=["GET"])
def me():
    if "user_id" not in session:
        return jsonify({"error": "Nie zalogowany"}), 401
    return (
        jsonify(
            {
                "user_id": session["user_id"],
                "username": session["username"],
                "role": session.get("role", "user"),
            }
        ),
        200,
    )

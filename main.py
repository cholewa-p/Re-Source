from flask import Flask, redirect
from datetime import timedelta
import logging
from logging import StreamHandler, FileHandler, Formatter
from logging.handlers import RotatingFileHandler
import os
from datetime import datetime
from login import login_bp
from dashboard import dashboard_bp
from db_operations import db_bp

app = Flask(__name__)
app.secret_key = "super_tajne_haslo"  # must be set for sessions
app.permanent_session_lifetime = timedelta(hours=1)

# Register the login blueprint
app.register_blueprint(login_bp, url_prefix="/auth")  # all login routes prefixed with /auth
app.register_blueprint(dashboard_bp)
app.register_blueprint(
    db_bp, url_prefix="/data"
)  # /data/update_ts_data , /data/forecast/<source_id>


@app.route("/")
def root():
    # Redirect root to the login page
    return redirect("/auth/login-page")


if __name__ == "__main__":
    # Configure logging: console + file so logs are visible and persisted
    fmt = Formatter("[%(asctime)s] %(levelname)s in %(module)s: %(message)s")

    sh = StreamHandler()
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(fmt)

    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    info_path = os.path.join("logs", f"app_info_{ts}.log")
    error_path = os.path.join("logs", f"app_error_{ts}.log")

    fh_info = RotatingFileHandler(info_path, maxBytes=5 * 1024 * 1024, backupCount=5)
    fh_info.setLevel(logging.INFO)
    fh_info.setFormatter(fmt)

    fh_error = RotatingFileHandler(error_path, maxBytes=5 * 1024 * 1024, backupCount=5)
    fh_error.setLevel(logging.ERROR)
    fh_error.setFormatter(fmt)

    # Attach handlers to Flask app logger
    app.logger.setLevel(logging.DEBUG)
    app.logger.addHandler(sh)
    app.logger.addHandler(fh_info)
    app.logger.addHandler(fh_error)

    # Also attach to root logger so third-party libraries are captured
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(sh)
    root.addHandler(fh_info)
    root.addHandler(fh_error)

    app.run(debug=True)

from flask import Flask, redirect
from datetime import timedelta
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
    app.run(debug=True)

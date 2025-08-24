# Re-Source: Renewable Energy Management & Forecasting Platform

This project is a proof-of-concept web application designed for managing and forecasting electricity production from renewable energy sources. It provides a user-facing dashboard, role-based access, and a backend API for data ingestion and time-series forecasting using SARIMAX models.

## Features

*   **User Authentication:** Secure login/logout system with session management.
*   **Role-Based Access Control (RBAC):** Differentiates between 'user' and 'admin' roles, restricting data access accordingly.
*   **Energy Source Management:** Users can view their registered renewable energy sources. Admins have a global view of all sources in the system.
*   **Time-Series Forecasting:**
    *   API endpoint to generate electricity production forecasts.
    *   Uses SARIMAX models for robust time-series analysis.
    *   Supports forecasting for individual sources or an aggregation of multiple sources.
*   **Model Persistence:** Ability to train and save forecasting models to disk for specific data scopes (e.g., `source_1`, `user_all_sources`).
*   **Data Ingestion API:** A dedicated endpoint to receive and store time-series power generation data from IoT devices or other producers.
*   **Interactive Dashboard:** A simple frontend to visualize data and interact with the forecasting engine.

## Technology Stack

*   **Backend:** Python 3, Flask
*   **Database:** PostgreSQL
*   **Machine Learning:** `statsmodels` (for SARIMAX), `pandas`, `numpy`
*   **Frontend:** HTML, JavaScript, Plotly.js for charting

## Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone git@github.com:cholewa-p/Re-Source.git
    cd Re-Source
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    A `requirements.txt` file is recommended. Based on the imports, the core dependencies are:
    ```bash
    pip install Flask psycopg2-binary pandas numpy statsmodels Werkzeug configparser
    ```

4.  **Configure the Database:**
    Create a `config.ini` file in the root directory with your PostgreSQL credentials. Use the following template:

    ```ini
    [solar_data]
    host = localhost
    port = 5432
    dbname = your_db_name
    user = your_db_user
    password = your_db_password
    ```
5.  **Run the application:**
    ```bash
    flask run
    # or
    python main.py
    ```
    The application will be available at `http://127.0.0.1:5000`. You will be redirected to the login page.

## API Endpoints

The application exposes several API endpoints, primarily for authentication and data operations.

*   `POST /auth/login`: Authenticates a user.
    *   **Payload:** `{"username": "...", "password": "..."}`
*   `POST /auth/logout`: Clears the user session.
*   `GET /dashboard`: Renders the main dashboard page.
*   `POST /data/update_ts_data`: Ingests a batch of time-series data points.
    *   **Payload:** `[{"timestamp": "...", "device_id": ..., "power": ...}]`
*   `GET /data/forecast`: Generates a forecast.
    *   **Query Params:**
        *   `source_id` (int): Forecast for a single source.
        *   `source_ids` (comma-separated ints): Forecast for an aggregation of sources.
        *   `horizon` (int, default: 24): Number of hours to forecast.
*   `POST /data/save_model`: Trains and saves a model based on a scope determined by the forecast request.
    *   **Payload:** `{"scope": "source_1"}`

## Project Structure

```
.
├── dashboard.py         # Dashboard blueprint and logic
├── db_operations.py     # Database operations and data API blueprint
├── login.py             # Authentication blueprint
├── main.py              # Main Flask application
├── timeseries_model.py  # SARIMAX model training and forecasting
├── templates/
│   ├── dashboard.html   # Main dashboard view
│   └── login.html       # Login page
├── config.ini           # (Needs to be created) DB configuration
└── README.md            # This file
```

---

This README provides a solid starting point for anyone looking to understand, set up, and contribute to your project.

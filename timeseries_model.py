"""Time series modeling component for electricity production forecasts.

Fetches historical power generation data from PostgreSQL (table: power_generation)
for a given energy source and fits a SARIMA model (statsmodels required).

Typical usage:

    from db_operations import get_db_config
    from timeseries_model import train_electricity_model

    db_cfg = get_db_config()  # reads config.ini [solar_data]
    model, forecast = train_electricity_model(db_cfg, source_id=1, horizon_hours=24)
    print(forecast.head())

Requirements (install if missing):
    pip install pandas numpy psycopg2-binary statsmodels

"""

from __future__ import annotations
import logging
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List

import pickle
import psycopg2
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Access
# ---------------------------------------------------------------------------


def fetch_timeseries(
    db_config: Dict[str, Any],
    source_id: int,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    tz: Optional[str] = None,
    freq: str = "H",
) -> pd.Series:
    """Fetch power generation time series for a source.

    Parameters:
        db_config: psycopg2 connection kwargs
        source_id: ID of the energy source
        start/end: optional datetime bounds (inclusive)
        tz: timezone string to convert to (data assumed UTC if naive)
        freq: expected frequency (default hourly 'H')

    Returns:
        Pandas Series indexed by timestamp with power_kw float values.
    """
    where_clauses = ["source_id = %s"]
    params = [source_id]
    if start:
        where_clauses.append("time >= %s")
        params.append(start)
    if end:
        where_clauses.append("time <= %s")
        params.append(end)

    where_sql = " AND " + " AND ".join(where_clauses) if where_clauses else ""
    sql = f"""
        SELECT time, power_kw
        FROM power_generation
        WHERE { ' AND '.join(where_clauses) }
        ORDER BY time ASC
    """

    conn = psycopg2.connect(**db_config)
    try:
        df = pd.read_sql(sql, conn, params=params, parse_dates=["time"])
    finally:
        conn.close()

    if df.empty:
        raise ValueError("No data returned for the specified parameters.")

    series = df.set_index("time")["power_kw"].sort_index()

    # Reindex to ensure regular spacing
    full_index = pd.date_range(series.index.min(), series.index.max(), freq=freq)
    series = series.reindex(full_index)
    missing = series.isna().sum()
    if missing:
        logger.warning("Found %d missing %sly points; forward/back filling gaps.", missing, freq)
        series = series.ffill().bfill()

    if tz:
        if series.index.tz is None:
            series.index = series.index.tz_localize("UTC").tz_convert(tz)
        else:
            series.index = series.index.tz_convert(tz)

    return series.astype(float)


# ---------------------------------------------------------------------------
# Modeling
# ---------------------------------------------------------------------------


class ElectricityProductionModel:
    """Wrapper for an hourly SARIMA model for electricity production."""

    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Tuple[int, int, int, int] = (1, 0, 1, 24),
        enforce_stationarity: bool = False,
        enforce_invertibility: bool = False,
    ) -> None:
        self.order = order
        self.seasonal_order = seasonal_order
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        self.model_: SARIMAX | None = None
        self.result_: SARIMAXResults | None = None
        self.train_index_: pd.DatetimeIndex | None = None

    def fit(self, series: pd.Series) -> "ElectricityProductionModel":
        """Fit SARIMAX model and store fitted results."""
        self.train_index_ = series.index
        self.model_ = SARIMAX(
            series,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=self.enforce_stationarity,
            enforce_invertibility=self.enforce_invertibility,
        )
        self.result_ = self.model_.fit(disp=False)
        logger.info("Model fit complete. AIC=%.2f", self.result_.aic)
        return self

    def forecast(self, steps: int, alpha: float = 0.05) -> pd.DataFrame:
        """Produce forecasts with (approximate) confidence intervals.

        Returns DataFrame with columns: forecast, lower, upper.
        """
        # Accept forecasts when a fitted result was loaded from disk (result_ present)
        if self.result_ is None:
            raise RuntimeError("Model not fit yet.")

        # Ensure we have a train_index_ to compute future timestamps. Try to reconstruct
        # from available metadata or from the fitted model object.
        if self.train_index_ is None:
            # attempt to extract from the result's model data
            try:
                data = getattr(self.result_.model, "data", None)
                if data is not None and hasattr(data, "row_labels"):
                    # row_labels may be an array of timestamps
                    self.train_index_ = pd.DatetimeIndex(data.row_labels)
            except Exception:
                pass

        # Fallback: if still missing, use the result's model endog index if available
        if self.train_index_ is None:
            try:
                endog_dates = getattr(self.result_.model, "endog_dates", None)
                if endog_dates is not None:
                    self.train_index_ = pd.DatetimeIndex(endog_dates)
            except Exception:
                pass

        # As a last resort, set train_index_ to current UTC hour so forecasting still produces a timeline
        if self.train_index_ is None or len(self.train_index_) == 0:
            now = pd.Timestamp.utcnow().floor("h")
            self.train_index_ = pd.DatetimeIndex([now])

        # Use lowercase 'h' frequency (pandas deprecation of 'H')
        future_index = pd.date_range(
            self.train_index_[-1] + pd.Timedelta(hours=1), periods=steps, freq="h"
        )

        pred = self.result_.get_forecast(steps=steps)
        mean = pred.predicted_mean
        conf = pred.conf_int(alpha=alpha)
        conf.columns = ["lower", "upper"]
        df = pd.concat([mean.rename("forecast"), conf], axis=1)
        df.index = future_index  # normalize index
        return df

    def save(self, path: str) -> None:
        """Persist model to disk.
        Saves state via statsmodels' native save + metadata sidecar.
        """

        payload = {
            "order": self.order,
            "seasonal_order": self.seasonal_order,
            "train_index_start": self.train_index_[0] if self.train_index_ is not None else None,
            "train_index_end": self.train_index_[-1] if self.train_index_ is not None else None,
            "statsmodels": True,
        }
        self.result_.save(path)
        meta_path = path + ".meta.pkl"
        with open(meta_path, "wb") as f:
            pickle.dump(payload, f)
        logger.info("Saved SARIMAX result to %s and metadata to %s", path, meta_path)

    @staticmethod
    def load(path: str) -> "ElectricityProductionModel":  # pragma: no cover - IO wrapper
        """Load a model produced by save()."""
        obj = ElectricityProductionModel()
        obj.result_ = SARIMAXResults.load(path)
        meta_path = path + ".meta.pkl"
        try:
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            obj.order = meta.get("order", obj.order)
            obj.seasonal_order = meta.get("seasonal_order", obj.seasonal_order)
            # If metadata contains training index bounds, reconstruct the full train index
            start_ts = meta.get("train_index_start")
            end_ts = meta.get("train_index_end")
            if start_ts and end_ts:
                try:
                    start_dt = pd.to_datetime(start_ts)
                    end_dt = pd.to_datetime(end_ts)
                    # Recreate hourly index between start and end
                    obj.train_index_ = pd.date_range(start_dt, end_dt, freq="h")
                except Exception:
                    # Fallback: set train_index_ to the end timestamp only
                    try:
                        obj.train_index_ = pd.DatetimeIndex([pd.to_datetime(end_ts)])
                    except Exception:
                        pass
            elif end_ts:
                try:
                    obj.train_index_ = pd.DatetimeIndex([pd.to_datetime(end_ts)])
                except Exception:
                    pass
        except FileNotFoundError:
            logger.warning("Metadata file %s not found; some attributes may be missing.", meta_path)
        return obj


# ---------------------------------------------------------------------------
# Orchestrator helper
# ---------------------------------------------------------------------------


def train_electricity_model(
    db_config: Dict[str, Any],
    source_id: int,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    horizon_hours: int = 24,
    save_path: Optional[str] = None,
) -> Tuple[ElectricityProductionModel, pd.DataFrame]:
    """End-to-end: fetch data, fit model, forecast horizon, optionally save.

    Returns:
        (model, forecast_dataframe)
    """
    series = fetch_timeseries(db_config, source_id, start, end)
    logger.info(
        "Fetched %d points for source %s (%s -> %s)",
        len(series),
        source_id,
        series.index.min(),
        series.index.max(),
    )

    model = ElectricityProductionModel()
    model.fit(series)
    forecast_df = model.forecast(horizon_hours)

    if save_path:
        model.save(save_path)

    return model, forecast_df


def fetch_and_aggregate_sources(
    db_config: Dict[str, Any],
    source_ids: List[int],
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    freq: str = "H",
) -> pd.Series:
    """Fetch multiple sources and aggregate (sum) power generation.

    Missing timestamps for individual sources are filled prior to summing to avoid bias.
    """
    series_list = []
    for sid in source_ids:
        try:
            s = fetch_timeseries(db_config, sid, start, end, freq=freq)
            series_list.append(s)
        except ValueError:
            continue  # skip empty source
    if not series_list:
        raise ValueError("No data available for provided sources")
    # Align all on union index then sum
    union_index = series_list[0].index
    for s in series_list[1:]:
        union_index = union_index.union(s.index)
    aligned = [s.reindex(union_index).ffill().bfill() for s in series_list]
    total = sum(aligned)
    return total.sort_index()


def train_aggregated_model(
    db_config: Dict[str, Any],
    source_ids: List[int],
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    horizon_hours: int = 24,
) -> Tuple[ElectricityProductionModel, pd.DataFrame]:
    """Train model on aggregated production of multiple sources."""
    series = fetch_and_aggregate_sources(db_config, source_ids, start, end)
    model = ElectricityProductionModel().fit(series)
    fc = model.forecast(horizon_hours)
    return model, fc

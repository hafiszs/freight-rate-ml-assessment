"""
Reusable data-cleaning & feature-engineering utilities for the freight-rate
prediction task. Kept as plain functions (fit on train, applied everywhere)
so the same logic is used for train / internal-validation / validation.csv /
december_chart_inputs.csv, avoiding train/serve skew.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EARTH_RADIUS_MI = 3958.8
N_HARMONICS = 2  # number of sin/cos pairs used to model annual seasonality


# --------------------------------------------------------------------------- #
# Basic cleaning
# --------------------------------------------------------------------------- #
def clean_weight_column(df: pd.DataFrame) -> pd.Series:
    """Fix the sign-error data-quality issue found in EDA: some weights are
    recorded as negative even though their magnitude matches the normal
    (positive) weight distribution. Taking the absolute value corrects this
    without discarding real observations."""
    return df["weight"].abs()


def fit_weight_medians(df: pd.DataFrame, weight_col: str = "weight_clean") -> pd.Series:
    """Median weight per equipment type, used to impute missing weights."""
    return df.groupby("equipment")[weight_col].median()


def impute_weight(df: pd.DataFrame, medians_by_equipment: pd.Series, weight_col: str = "weight_clean") -> pd.Series:
    values = df[weight_col].to_numpy(copy=True)
    missing = df[weight_col].isna().to_numpy()
    if missing.any():
        fallback = medians_by_equipment.reindex(df["equipment"]).to_numpy()
        values[missing] = fallback[missing]
    return pd.Series(values, index=df.index)


# --------------------------------------------------------------------------- #
# City coordinate lookup (needed because december_chart_inputs.csv has no
# lat/lon columns at all -- we reconstruct them from cities seen in
# train/validation, since every city that appears always has one consistent
# lat/lon pair, as confirmed in EDA)
# --------------------------------------------------------------------------- #
def build_city_coords(*frames: pd.DataFrame) -> dict[str, tuple[float, float]]:
    coords: dict[str, tuple[float, float]] = {}
    for frame in frames:
        if {"pickup", "pickup_lat", "pickup_lon"}.issubset(frame.columns):
            for city, lat, lon in frame[["pickup", "pickup_lat", "pickup_lon"]].drop_duplicates().itertuples(index=False):
                coords[city] = (lat, lon)
        if {"delivery", "delivery_lat", "delivery_lon"}.issubset(frame.columns):
            for city, lat, lon in frame[["delivery", "delivery_lat", "delivery_lon"]].drop_duplicates().itertuples(index=False):
                coords[city] = (lat, lon)
    return coords


def attach_coords(df: pd.DataFrame, city_coords: dict[str, tuple[float, float]]) -> pd.DataFrame:
    """Add pickup_lat/pickup_lon/delivery_lat/delivery_lon if not already present,
    looking them up from the known-city table (used for december_chart_inputs.csv)."""
    out = df.copy()
    if "pickup_lat" not in out.columns:
        out["pickup_lat"] = out["pickup"].map(lambda c: city_coords.get(c, (np.nan, np.nan))[0])
        out["pickup_lon"] = out["pickup"].map(lambda c: city_coords.get(c, (np.nan, np.nan))[1])
    if "delivery_lat" not in out.columns:
        out["delivery_lat"] = out["delivery"].map(lambda c: city_coords.get(c, (np.nan, np.nan))[0])
        out["delivery_lon"] = out["delivery"].map(lambda c: city_coords.get(c, (np.nan, np.nan))[1])
    return out


def haversine(lat1, lon1, lat2, lon2) -> np.ndarray:
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * EARTH_RADIUS_MI * np.arcsin(np.sqrt(a))


# --------------------------------------------------------------------------- #
# Seasonal (annual) harmonic regression -- used to (a) impute missing
# market_index/quote_signal values and (b) *generate* those two columns
# entirely for december_chart_inputs.csv, which never has them.
# --------------------------------------------------------------------------- #
def _harmonic_design(day_of_year: np.ndarray, n_harmonics: int = N_HARMONICS) -> np.ndarray:
    cols = [np.ones_like(day_of_year, dtype=float)]
    for k in range(1, n_harmonics + 1):
        angle = 2 * np.pi * k * day_of_year / 365.25
        cols.append(np.sin(angle))
        cols.append(np.cos(angle))
    return np.column_stack(cols)


def fit_seasonal_curve(day_of_year: pd.Series, values: pd.Series, n_harmonics: int = N_HARMONICS) -> np.ndarray:
    mask = values.notna()
    X = _harmonic_design(day_of_year[mask].to_numpy(), n_harmonics)
    y = values[mask].to_numpy()
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef


def predict_seasonal_curve(day_of_year: pd.Series, coef: np.ndarray, n_harmonics: int = N_HARMONICS) -> np.ndarray:
    X = _harmonic_design(day_of_year.to_numpy(), n_harmonics)
    return X @ coef


# --------------------------------------------------------------------------- #
# Full feature engineering pipeline
# --------------------------------------------------------------------------- #
FEATURE_COLUMNS = [
    "distance",
    "weight_final",
    "equipment",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "market_index_final",
    "quote_signal_final",
    "doy_sin",
    "doy_cos",
    "dow",
    "is_weekend",
]
CATEGORICAL_FEATURES = ["equipment", "dow"]


class FeatureRecipe:
    """Holds everything fit on the training set (medians, seasonal curves,
    city coordinates) so it can be applied identically to any other split."""

    def __init__(self) -> None:
        self.weight_medians: pd.Series | None = None
        self.city_coords: dict[str, tuple[float, float]] = {}
        self.market_index_coef: np.ndarray | None = None
        self.quote_signal_coef: np.ndarray | None = None

    def fit(self, train_df: pd.DataFrame, *extra_coord_frames: pd.DataFrame) -> "FeatureRecipe":
        tmp = train_df.copy()
        tmp["weight_clean"] = clean_weight_column(tmp)
        self.weight_medians = fit_weight_medians(tmp)

        self.city_coords = build_city_coords(train_df, *extra_coord_frames)

        date = pd.to_datetime(tmp["date"])
        doy = date.dt.dayofyear
        self.market_index_coef = fit_seasonal_curve(doy, tmp["market_index"])
        self.quote_signal_coef = fit_seasonal_curve(doy, tmp["quote_signal"])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["date"] = pd.to_datetime(out["date"])

        # weight: fix sign, impute missing
        out["weight_clean"] = clean_weight_column(out)
        out["weight_final"] = impute_weight(out, self.weight_medians)

        # coordinates: use given ones, else look up from known cities
        out = attach_coords(out, self.city_coords)

        # seasonal market_index / quote_signal: use actual value when present,
        # otherwise (missing, or column entirely absent e.g. december inputs)
        # fall back to the fitted annual seasonal curve.
        doy = out["date"].dt.dayofyear
        mi_seasonal = predict_seasonal_curve(doy, self.market_index_coef)
        qs_seasonal = predict_seasonal_curve(doy, self.quote_signal_coef)
        if "market_index" in out.columns:
            out["market_index_final"] = out["market_index"].fillna(pd.Series(mi_seasonal, index=out.index))
        else:
            out["market_index_final"] = mi_seasonal
        if "quote_signal" in out.columns:
            out["quote_signal_final"] = out["quote_signal"].fillna(pd.Series(qs_seasonal, index=out.index))
        else:
            out["quote_signal_final"] = qs_seasonal

        # calendar features
        out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
        out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
        out["dow"] = out["date"].dt.dayofweek.astype("category")
        out["is_weekend"] = (out["date"].dt.dayofweek >= 5).astype(int)
        out["equipment"] = out["equipment"].astype("category")

        return out

    def build_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        transformed = self.transform(df)
        return transformed[FEATURE_COLUMNS]

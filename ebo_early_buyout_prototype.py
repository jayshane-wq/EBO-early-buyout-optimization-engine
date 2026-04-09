import io
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="EBO Early Buyout Candidate Tool", layout="wide")

st.title("EBO Early Buyout Candidate Tool")
st.caption(
    "Prototype decision support tool for identifying FHA EBO candidates with the highest likelihood of "
    "re-performing through the 3-payment seasoning period and achieving sensible redelivery economics."
)


# =========================
# Utility functions
# =========================
def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def safe_col(df: pd.DataFrame, column: str, default=0.0):
    if column not in df.columns:
        df[column] = default
    return df[column]


# =========================
# Model weights
# =========================
@dataclass
class ModelWeights:
    ltv: float = 0.11
    cltv: float = 0.12
    payment_history: float = 0.18
    payment_timing: float = 0.10
    current_dq: float = 0.08
    current_note_rate: float = 0.03
    incentive_spread: float = 0.06
    redelivery_economics: float = 0.12
    income_stability: float = 0.07
    macro_stress: float = 0.05
    property_trend: float = 0.04
    occupancy: float = 0.04


DEFAULT_WEIGHTS = ModelWeights()


# =========================
# Normalizers / feature engineering
# =========================
def normalize_equity_signal(cltv: float) -> float:
    if pd.isna(cltv):
        return 0.50
    if cltv < 70:
        return 1.00
    if cltv < 80:
        return 0.90
    if cltv < 90:
        return 0.75
    if cltv < 100:
        return 0.55
    if cltv < 110:
        return 0.30
    return 0.10


def normalize_ltv_signal(ltv: float) -> float:
    if pd.isna(ltv):
        return 0.50
    if ltv < 70:
        return 1.00
    if ltv < 80:
        return 0.90
    if ltv < 90:
        return 0.75
    if ltv < 97:
        return 0.58
    if ltv < 105:
        return 0.30
    return 0.10


def normalize_payment_history(on_time_12m: float, partial_pay_count_12m: float, broken_promise_count_12m: float) -> float:
    on_time_component = clamp(on_time_12m / 12.0, 0, 1)
    partial_penalty = clamp(partial_pay_count_12m / 6.0, 0, 1) * 0.25
    promise_penalty = clamp(broken_promise_count_12m / 6.0, 0, 1) * 0.35
    return clamp(on_time_component - partial_penalty - promise_penalty, 0, 1)


def normalize_payment_timing(avg_days_late_predefault: float) -> float:
    if pd.isna(avg_days_late_predefault):
        return 0.50
    if avg_days_late_predefault <= 0:
        return 1.00
    if avg_days_late_predefault <= 3:
        return 0.88
    if avg_days_late_predefault <= 7:
        return 0.68
    if avg_days_late_predefault <= 15:
        return 0.42
    return 0.15


def normalize_dq(days_delinquent: float) -> float:
    if pd.isna(days_delinquent):
        return 0.40
    if days_delinquent <= 90:
        return 0.90
    if days_delinquent <= 120:
        return 0.75
    if days_delinquent <= 150:
        return 0.52
    if days_delinquent <= 180:
        return 0.30
    return 0.15


def normalize_income_stability(income_type: str, months_on_job: float) -> float:
    income_type = str(income_type).strip().lower()
    type_score_map = {
        "w2": 0.90,
        "salary": 0.95,
        "hourly": 0.75,
        "self-employed": 0.55,
        "gig": 0.35,
        "retired": 0.80,
        "fixed income": 0.82,
        "unknown": 0.50,
    }
    base = type_score_map.get(income_type, 0.50)

    if pd.isna(months_on_job):
        tenure_adj = 0.0
    elif months_on_job >= 60:
        tenure_adj = 0.10
    elif months_on_job >= 24:
        tenure_adj = 0.06
    elif months_on_job >= 12:
        tenure_adj = 0.03
    elif months_on_job < 6:
        tenure_adj = -0.08
    else:
        tenure_adj = 0.0

    return clamp(base + tenure_adj, 0, 1)


def normalize_macro_stress(gas_price_change_pct_90d: float, unemployment_change_pct_90d: float) -> float:
    gas = 0 if pd.isna(gas_price_change_pct_90d) else gas_price_change_pct_90d
    unemp = 0 if pd.isna(unemployment_change_pct_90d) else unemployment_change_pct_90d
    stress_score = 0.65 - (gas * 0.01 * 0.8) - (unemp * 0.45)
    return clamp(stress_score, 0.05, 0.95)


def normalize_property_trend(hpi_change_pct_12m: float) -> float:
    if pd.isna(hpi_change_pct_12m):
        return 0.50
    if hpi_change_pct_12m >= 8:
        return 0.95
    if hpi_change_pct_12m >= 4:
        return 0.85
    if hpi_change_pct_12m >= 0:
        return 0.70
    if hpi_change_pct_12m >= -5:
        return 0.40
    return 0.15


def normalize_occupancy(occupancy_status: str) -> float:
    status = str(occupancy_status).strip().lower()
    mapping = {
        "owner occupied": 0.95,
        "primary": 0.95,
        "second home": 0.55,
        "investor": 0.20,
        "unknown": 0.50,
    }
    return mapping.get(status, 0.50)


def rate_incentive_score(current_note_rate: float, market_rate_forecast_120d: float) -> float:
    if pd.isna(current_note_rate) or pd.isna(market_rate_forecast_120d):
        return 0.50
    spread = market_rate_forecast_120d - current_note_rate
    if spread >= 3.0:
        return 1.00
    if spread >= 2.0:
        return 0.90
    if spread >= 1.0:
        return 0.78
    if spread >= 0.0:
        return 0.62
    if spread >= -1.0:
        return 0.40
    return 0.20


def estimate_redelivery_price(note_rate: float, market_rate: float, slope_per_rate_point: float = 4.5) -> float:
    """
    Demo-only price approximation.
    If note rate is below market, price will fall below par.
    If note rate is above market, price will rise above par.
    """
    if pd.isna(note_rate) or pd.isna(market_rate):
        return np.nan

    price = 100 + ((note_rate - market_rate) * slope_per_rate_point)
    return float(clamp(price, 75, 125))


def price_to_bps(price: float) -> float:
    if pd.isna(price):
        return np.nan
    return float((price - 100) * 100)


def normalize_redelivery_economics(current_gain_bps: float, forecast_gain_bps_120d: float) -> float:
    if pd.isna(current_gain_bps) and pd.isna(forecast_gain_bps_120d):
        return 0.50

    value = np.nanmean([current_gain_bps, forecast_gain_bps_120d])
    if value >= 250:
        return 1.00
    if value >= 100:
        return 0.85
    if value >= 0:
        return 0.70
    if value >= -100:
        return 0.50
    if value >= -250:
        return 0.28
    return 0.10


def score_loan(row: pd.Series, weights: ModelWeights) -> Dict[str, float]:
    equity_cltv = normalize_equity_signal(row.get("cltv", np.nan))
    equity_ltv = normalize_ltv_signal(row.get("ltv", np.nan))
    pay_hist = normalize_payment_history(
        row.get("on_time_payments_12m", 0),
        row.get("partial_pay_count_12m", 0),
        row.get("broken_promise_count_12m", 0),
    )
    pay_time = normalize_payment_timing(row.get("avg_days_late_predefault", np.nan))
    dq = normalize_dq(row.get("days_delinquent", np.nan))
    income = normalize_income_stability(row.get("income_type", "unknown"), row.get("months_on_job", np.nan))
    macro = normalize_macro_stress(
        row.get("gas_price_change_pct_90d", np.nan),
        row.get("unemployment_change_pct_90d", np.nan),
    )
    prop = normalize_property_trend(row.get("hpi_change_pct_12m", np.nan))
    occupancy = normalize_occupancy(row.get("occupancy_status", "unknown"))
    incentive = rate_incentive_score(row.get("current_note_rate", np.nan), row.get("forecast_pmms_120d", np.nan))
    redelivery = normalize_redelivery_economics(
        row.get("current_redelivery_gain_bps", np.nan),
        row.get("forecast_redelivery_gain_bps_120d", np.nan),
    )

    current_note_rate = row.get("current_note_rate", np.nan)
    note_rate_score = 0.50 if pd.isna(current_note_rate) else clamp((7.0 - abs(5.5 - current_note_rate)) / 7.0, 0.2, 1.0)

    weighted_score = (
        equity_ltv * weights.ltv
        + equity_cltv * weights.cltv
        + pay_hist * weights.payment_history
        + pay_time * weights.payment_timing
        + dq * weights.current_dq
        + note_rate_score * weights.current_note_rate
        + incentive * weights.incentive_spread
        + redelivery * weights.redelivery_economics
        + income * weights.income_stability
        + macro * weights.macro_stress
        + prop * weights.property_trend
        + occupancy * weights.occupancy
    )

    reperform_probability = sigmoid((weighted_score - 0.60) * 8)

    partial_claim_probability = sigmoid(
        (
            pay_hist * 0.24
            + dq * 0.20
            + income * 0.18
            + equity_cltv * 0.12
            + macro * 0.06
            + prop * 0.07
            + occupancy * 0.05
            + incentive * 0.08
            - 0.43
        ) * 6
    )

    expected_ebo_success = reperform_probability * partial_claim_probability
    economic_view = np.nanmean([row.get("current_redelivery_gain_bps", np.nan), row.get("forecast_redelivery_gain_bps_120d", np.nan)])

    return {
        "equity_cltv_score": round(equity_cltv, 4),
        "equity_ltv_score": round(equity_ltv, 4),
        "payment_history_score": round(pay_hist, 4),
        "payment_timing_score": round(pay_time, 4),
        "dq_score": round(dq, 4),
        "income_stability_score": round(income, 4),
        "macro_score": round(macro, 4),
        "property_trend_score": round(prop, 4),
        "occupancy_score": round(occupancy, 4),
        "rate_incentive_score": round(incentive, 4),
        "redelivery_econ_score": round(redelivery, 4),
        "weighted_score": round(weighted_score, 4),
        "reperform_probability": round(reperform_probability, 4),
        "partial_claim_probability": round(partial_claim_probability, 4),
        "expected_ebo_success": round(expected_ebo_success, 4),
        "avg_redelivery_gain_bps": round(economic_view, 2) if not pd.isna(economic_view) else np.nan,
    }


# =========================
# PMMS / rate helpers
# =========================
def parse_pmms_upload(uploaded_file) -> Tuple[pd.DataFrame, Optional[str]]:
    try:
        file_name = str(getattr(uploaded_file, "name", "")).lower()

        if file_name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            cols = {str(c).strip().lower(): c for c in df.columns}

            date_col = cols.get("date") or cols.get("week")
            rate_col = cols.get("rate") or cols.get("frm") or cols.get("30 yr") or cols.get("30yr")

            if date_col is None or rate_col is None:
                return pd.DataFrame(), "CSV must contain date/week and rate columns."

            parsed = df[[date_col, rate_col]].copy()
            parsed.columns = ["date", "rate"]
            parsed["date"] = pd.to_datetime(parsed["date"], errors="coerce")
            parsed["rate"] = pd.to_numeric(parsed["rate"], errors="coerce")
            parsed = parsed.dropna(subset=["date", "rate"])
            return parsed, None

        raw = pd.read_excel(uploaded_file, sheet_name=0, header=None)
        if raw.empty:
            return pd.DataFrame(), "Uploaded Excel file is empty."

        header_row = None
        for idx in range(min(len(raw), 25)):
            row_vals = raw.iloc[idx].astype(str).str.strip().str.lower().tolist()
            if "week" in row_vals and "frm" in row_vals:
                header_row = idx
                break

        if header_row is None:
            parsed = raw.iloc[:, [0, 1]].copy()
            parsed.columns = ["date", "rate"]
        else:
            date_col_idx = None
            rate_col_idx = None
            for col_idx in range(raw.shape[1]):
                val = str(raw.iloc[header_row, col_idx]).strip().lower()
                if val == "week":
                    date_col_idx = col_idx
                if val == "frm" and rate_col_idx is None:
                    rate_col_idx = col_idx

            if date_col_idx is None or rate_col_idx is None:
                return pd.DataFrame(), "Could not locate Week and FRM columns in the PMMS workbook."

            parsed = raw.iloc[header_row + 1 :, [date_col_idx, rate_col_idx]].copy()
            parsed.columns = ["date", "rate"]

        parsed["date"] = pd.to_datetime(parsed["date"], errors="coerce")
        parsed["rate"] = pd.to_numeric(parsed["rate"], errors="coerce")
        parsed = parsed.dropna(subset=["date", "rate"]).sort_values("date")

        if parsed.empty:
            return pd.DataFrame(), "Could not detect PMMS dates and rates from the uploaded workbook."

        return parsed, None
    except Exception as exc:
        return pd.DataFrame(), f"Unable to parse PMMS upload: {exc}"


def forecast_pmms(pmms_df: pd.DataFrame, horizon_days: int = 120, lookback_days: int = 365) -> Tuple[float, pd.DataFrame, pd.DataFrame]:
    """
    Uses only recent PMMS history to avoid 1970s-today structural bias.
    Returns forecast, plot dataframe, and the trimmed recent window used.
    """
    df = pmms_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").dropna(subset=["rate"])

    if len(df) < 6:
        return np.nan, df, df

    cutoff = df["date"].max() - pd.Timedelta(days=lookback_days)
    recent = df[df["date"] >= cutoff].copy()

    if len(recent) < 6:
        recent = df.tail(min(len(df), 26)).copy()

    recent["t"] = (recent["date"] - recent["date"].min()).dt.days.astype(float)
    x = recent["t"].to_numpy()
    y = recent["rate"].astype(float).to_numpy()

    slope, intercept = np.polyfit(x, y, 1)
    future_t = float(recent["t"].max() + horizon_days)
    forecast = float((slope * future_t) + intercept)

    future_date = recent["date"].max() + pd.Timedelta(days=horizon_days)
    projected = pd.DataFrame({"date": [future_date], "rate": [forecast], "series": ["Forecast"]})

    hist = recent[["date", "rate"]].copy()
    hist["series"] = "Recent PMMS Used"
    out = pd.concat([hist, projected], ignore_index=True)

    return round(forecast, 4), out, recent


def derive_redelivery_fields(df: pd.DataFrame, current_market_rate: float, forecast_market_rate: float) -> pd.DataFrame:
    out = df.copy()
    out["estimated_current_price"] = out["current_note_rate"].apply(lambda x: estimate_redelivery_price(x, current_market_rate))
    out["estimated_forecast_price_120d"] = out["current_note_rate"].apply(lambda x: estimate_redelivery_price(x, forecast_market_rate))
    out["current_redelivery_gain_bps"] = out["estimated_current_price"].apply(price_to_bps)
    out["forecast_redelivery_gain_bps_120d"] = out["estimated_forecast_price_120d"].apply(price_to_bps)
    return out


# =========================
# Sample data
# =========================
def build_sample_candidate_data() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "loan_id": "L001", "ltv": 72, "cltv": 74, "on_time_payments_12m": 10,
            "partial_pay_count_12m": 0, "broken_promise_count_12m": 0, "avg_days_late_predefault": 2,
            "days_delinquent": 92, "income_type": "salary", "months_on_job": 48,
            "current_note_rate": 3.25, "gas_price_change_pct_90d": 4, "unemployment_change_pct_90d": 0.1,
            "hpi_change_pct_12m": 5, "occupancy_status": "owner occupied"
        },
        {
            "loan_id": "L002", "ltv": 96, "cltv": 101, "on_time_payments_12m": 7,
            "partial_pay_count_12m": 2, "broken_promise_count_12m": 2, "avg_days_late_predefault": 12,
            "days_delinquent": 118, "income_type": "gig", "months_on_job": 8,
            "current_note_rate": 4.75, "gas_price_change_pct_90d": 9, "unemployment_change_pct_90d": 0.4,
            "hpi_change_pct_12m": -2, "occupancy_status": "owner occupied"
        },
        {
            "loan_id": "L003", "ltv": 80, "cltv": 84, "on_time_payments_12m": 9,
            "partial_pay_count_12m": 1, "broken_promise_count_12m": 0, "avg_days_late_predefault": 4,
            "days_delinquent": 90, "income_type": "retired", "months_on_job": 120,
            "current_note_rate": 2.99, "gas_price_change_pct_90d": 3, "unemployment_change_pct_90d": 0.0,
            "hpi_change_pct_12m": 6, "occupancy_status": "owner occupied"
        },
        {
            "loan_id": "L004", "ltv": 68, "cltv": 70, "on_time_payments_12m": 11,
            "partial_pay_count_12m": 0, "broken_promise_count_12m": 0, "avg_days_late_predefault": 1,
            "days_delinquent": 95, "income_type": "salary", "months_on_job": 84,
            "current_note_rate": 7.25, "gas_price_change_pct_90d": 2, "unemployment_change_pct_90d": 0.0,
            "hpi_change_pct_12m": 7, "occupancy_status": "owner occupied"
        },
        {
            "loan_id": "L005", "ltv": 78, "cltv": 82, "on_time_payments_12m": 8,
            "partial_pay_count_12m": 1, "broken_promise_count_12m": 1, "avg_days_late_predefault": 6,
            "days_delinquent": 102, "income_type": "hourly", "months_on_job": 30,
            "current_note_rate": 6.85, "gas_price_change_pct_90d": 5, "unemployment_change_pct_90d": 0.1,
            "hpi_change_pct_12m": 4, "occupancy_status": "owner occupied"
        },
        {
            "loan_id": "L006", "ltv": 92, "cltv": 97, "on_time_payments_12m": 6,
            "partial_pay_count_12m": 2, "broken_promise_count_12m": 2, "avg_days_late_predefault": 14,
            "days_delinquent": 130, "income_type": "self-employed", "months_on_job": 18,
            "current_note_rate": 7.50, "gas_price_change_pct_90d": 8, "unemployment_change_pct_90d": 0.3,
            "hpi_change_pct_12m": 1, "occupancy_status": "owner occupied"
        },
        {
            "loan_id": "L007", "ltv": 65, "cltv": 67, "on_time_payments_12m": 12,
            "partial_pay_count_12m": 0, "broken_promise_count_12m": 0, "avg_days_late_predefault": 0,
            "days_delinquent": 91, "income_type": "fixed income", "months_on_job": 120,
            "current_note_rate": 8.10, "gas_price_change_pct_90d": 1, "unemployment_change_pct_90d": 0.0,
            "hpi_change_pct_12m": 8, "occupancy_status": "owner occupied"
        },
        {
            "loan_id": "L008", "ltv": 99, "cltv": 106, "on_time_payments_12m": 5,
            "partial_pay_count_12m": 3, "broken_promise_count_12m": 3, "avg_days_late_predefault": 18,
            "days_delinquent": 145, "income_type": "gig", "months_on_job": 5,
            "current_note_rate": 3.75, "gas_price_change_pct_90d": 11, "unemployment_change_pct_90d": 0.5,
            "hpi_change_pct_12m": -4, "occupancy_status": "investor"
        },
        {
            "loan_id": "L009", "ltv": 74, "cltv": 76, "on_time_payments_12m": 9,
            "partial_pay_count_12m": 0, "broken_promise_count_12m": 1, "avg_days_late_predefault": 5,
            "days_delinquent": 93, "income_type": "hourly", "months_on_job": 40,
            "current_note_rate": 6.20, "gas_price_change_pct_90d": 3, "unemployment_change_pct_90d": 0.0,
            "hpi_change_pct_12m": 5, "occupancy_status": "owner occupied"
        },
        {
            "loan_id": "L010", "ltv": 88, "cltv": 90, "on_time_payments_12m": 7,
            "partial_pay_count_12m": 1, "broken_promise_count_12m": 1, "avg_days_late_predefault": 8,
            "days_delinquent": 110, "income_type": "salary", "months_on_job": 14,
            "current_note_rate": 5.95, "gas_price_change_pct_90d": 6, "unemployment_change_pct_90d": 0.2,
            "hpi_change_pct_12m": 2, "occupancy_status": "owner occupied"
        },
        {
            "loan_id": "L011", "ltv": 70, "cltv": 72, "on_time_payments_12m": 10,
            "partial_pay_count_12m": 0, "broken_promise_count_12m": 0, "avg_days_late_predefault": 2,
            "days_delinquent": 94, "income_type": "salary", "months_on_job": 60,
            "current_note_rate": 8.45, "gas_price_change_pct_90d": 2, "unemployment_change_pct_90d": 0.0,
            "hpi_change_pct_12m": 9, "occupancy_status": "owner occupied"
        },
        {
            "loan_id": "L012", "ltv": 95, "cltv": 100, "on_time_payments_12m": 6,
            "partial_pay_count_12m": 2, "broken_promise_count_12m": 1, "avg_days_late_predefault": 11,
            "days_delinquent": 121, "income_type": "self-employed", "months_on_job": 10,
            "current_note_rate": 4.10, "gas_price_change_pct_90d": 7, "unemployment_change_pct_90d": 0.2,
            "hpi_change_pct_12m": -1, "occupancy_status": "owner occupied"
        },
    ])


def build_sample_pmms_data() -> pd.DataFrame:
    dates = pd.date_range("2025-04-04", periods=26, freq="W-FRI")
    rates = [6.82, 6.79, 6.76, 6.73, 6.70, 6.67, 6.64, 6.60, 6.57, 6.53, 6.50, 6.47, 6.44,
             6.41, 6.38, 6.35, 6.33, 6.30, 6.27, 6.24, 6.21, 6.18, 6.16, 6.13, 6.10, 6.08]
    return pd.DataFrame({"date": dates, "rate": rates})


# =========================
# Sidebar
# =========================
st.sidebar.header("Configuration")
use_sample = st.sidebar.toggle("Use sample candidate data", value=True)
use_sample_pmms = st.sidebar.toggle("Use sample PMMS trend data", value=True)
lookback_days = st.sidebar.slider("PMMS lookback window (days)", 90, 730, 365, 30)
manual_current_market_rate = st.sidebar.number_input("Override current market rate (optional)", min_value=0.0, max_value=20.0, value=0.0, step=0.05)

st.sidebar.markdown("### EBO Screening Rules")
min_dq = st.sidebar.slider("Minimum delinquency days", 60, 180, 90, 5)
max_dq = st.sidebar.slider("Maximum delinquency days", 90, 365, 150, 5)
min_success_prob = st.sidebar.slider("Minimum expected EBO success", 0.0, 1.0, 0.40, 0.01)

st.sidebar.markdown("### Recommendation Thresholds")
execute_threshold = st.sidebar.slider("Execute now threshold", 0.0, 1.0, 0.65, 0.01)
monitor_threshold = st.sidebar.slider("Monitor threshold", 0.0, 1.0, 0.48, 0.01)

st.sidebar.markdown("### Fairness / Process Guardrails")
st.sidebar.info(
    "EBO selection should never change borrower treatment, outreach cadence, or partial claim evaluation timing. "
    "This tool supports internal execution strategy only."
)


# =========================
# Main layout
# =========================
col1, col2 = st.columns([1.15, 1])

with col1:
    st.subheader("1) Candidate Population Input")
    uploaded_candidates = st.file_uploader("Upload candidate CSV", type=["csv"], key="candidate_upload")

    if use_sample or uploaded_candidates is None:
        candidate_df = build_sample_candidate_data()
    else:
        candidate_df = pd.read_csv(uploaded_candidates)

    st.dataframe(candidate_df, use_container_width=True, height=260)

with col2:
    st.subheader("2) PMMS Trend / Rate Forecast")
    uploaded_pmms = st.file_uploader("Upload PMMS file (CSV or Freddie Mac Excel)", type=["csv", "xlsx", "xls"], key="pmms_upload")

    pmms_parse_error = None
    if use_sample_pmms or uploaded_pmms is None:
        pmms_df = build_sample_pmms_data()
    else:
        pmms_df, pmms_parse_error = parse_pmms_upload(uploaded_pmms)

    if pmms_parse_error:
        st.error(pmms_parse_error)

    forecast_120d, pmms_plot_df, recent_pmms = forecast_pmms(pmms_df, horizon_days=120, lookback_days=lookback_days)

    if manual_current_market_rate > 0:
        current_market_rate = manual_current_market_rate
    else:
        current_market_rate = float(recent_pmms["rate"].iloc[-1]) if not recent_pmms.empty else np.nan

    st.metric("Current Market Rate Used", f"{current_market_rate:.2f}%" if not pd.isna(current_market_rate) else "N/A")
    st.metric("Directional PMMS Forecast (120d)", f"{forecast_120d:.2f}%" if not pd.isna(forecast_120d) else "Insufficient Data")
    if not pmms_plot_df.empty:
        st.line_chart(pmms_plot_df.set_index("date")["rate"])
    st.dataframe(recent_pmms, use_container_width=True, height=260)
    st.caption("The forecast uses only the recent PMMS lookback window rather than the full 1971-present history.")


# =========================
# Data preparation
# =========================
for col, default in {
    "loan_id": "UNKNOWN",
    "ltv": np.nan,
    "cltv": np.nan,
    "on_time_payments_12m": 0,
    "partial_pay_count_12m": 0,
    "broken_promise_count_12m": 0,
    "avg_days_late_predefault": np.nan,
    "days_delinquent": np.nan,
    "income_type": "unknown",
    "months_on_job": np.nan,
    "current_note_rate": np.nan,
    "gas_price_change_pct_90d": np.nan,
    "unemployment_change_pct_90d": np.nan,
    "hpi_change_pct_12m": np.nan,
    "occupancy_status": "unknown",
}.items():
    safe_col(candidate_df, col, default)

candidate_df["forecast_pmms_120d"] = forecast_120d
candidate_df = derive_redelivery_fields(candidate_df, current_market_rate=current_market_rate, forecast_market_rate=forecast_120d)


# =========================
# Run model
# =========================
scored_rows: List[Dict[str, float]] = []
for _, row in candidate_df.iterrows():
    scored_rows.append(score_loan(row, DEFAULT_WEIGHTS))

score_df = pd.concat([candidate_df.reset_index(drop=True), pd.DataFrame(scored_rows)], axis=1)

screened = score_df[
    (score_df["days_delinquent"] >= min_dq)
    & (score_df["days_delinquent"] <= max_dq)
    & (score_df["expected_ebo_success"] >= min_success_prob)
].copy()


def recommend_action(success_prob: float, avg_redelivery_gain_bps: float, forecast_gain_bps_120d: float) -> str:
    econ_component = 0.50 if pd.isna(avg_redelivery_gain_bps) else clamp((avg_redelivery_gain_bps + 300) / 900, 0, 1)
    combined = (success_prob * 0.72) + (econ_component * 0.28)

    if pd.isna(forecast_gain_bps_120d):
        forecast_gain_bps_120d = -999

    if combined >= execute_threshold and avg_redelivery_gain_bps >= 0 and forecast_gain_bps_120d >= 0:
        return "Execute EBO"
    if combined >= monitor_threshold and forecast_gain_bps_120d > -150:
        return "Monitor / Outreach"
    return "Do Not EBO"


screened["recommendation"] = screened.apply(
    lambda r: recommend_action(r["expected_ebo_success"], r["avg_redelivery_gain_bps"], r["forecast_redelivery_gain_bps_120d"]), axis=1
)
screened["regulatory_guardrail"] = "Same borrower treatment and timing required"

score_df["recommendation"] = score_df.apply(
    lambda r: recommend_action(r["expected_ebo_success"], r["avg_redelivery_gain_bps"], r["forecast_redelivery_gain_bps_120d"]), axis=1
)
score_df["regulatory_guardrail"] = "Same borrower treatment and timing required"


# =========================
# Dashboard
# =========================
st.subheader("3) EBO Candidate Dashboard")
a, b, c, d = st.columns(4)
a.metric("Total Loans Reviewed", len(score_df))
b.metric("Eligible Screened Population", len(screened))
c.metric("Avg Re-perform Probability", f"{score_df['reperform_probability'].mean():.1%}")
d.metric("Avg Avg Redelivery Gain/Loss", f"{score_df['avg_redelivery_gain_bps'].mean():.0f} bps")

recommended_view = screened[screened["recommendation"] != "Do Not EBO"].copy()

st.markdown("### Recommended Candidates")
display_cols = [
    "loan_id", "ltv", "cltv", "days_delinquent", "income_type", "occupancy_status",
    "current_note_rate", "forecast_pmms_120d", "estimated_current_price", "estimated_forecast_price_120d",
    "current_redelivery_gain_bps", "forecast_redelivery_gain_bps_120d", "avg_redelivery_gain_bps",
    "reperform_probability", "partial_claim_probability", "expected_ebo_success", "recommendation", "regulatory_guardrail"
]

st.dataframe(
    recommended_view[display_cols].sort_values(["expected_ebo_success", "avg_redelivery_gain_bps"], ascending=[False, False]),
    use_container_width=True,
    height=320,
)

st.markdown("### All Loans Scored")
st.dataframe(
    score_df[display_cols].sort_values(["expected_ebo_success", "avg_redelivery_gain_bps"], ascending=[False, False]),
    use_container_width=True,
    height=320,
)


# =========================
# Explainability
# =========================
st.subheader("4) Loan Explainability")
loan_options = score_df["loan_id"].astype(str).tolist()
selected_loan = st.selectbox("Select loan", loan_options)
loan_row = score_df[score_df["loan_id"].astype(str) == selected_loan].iloc[0]

factor_rows = [
    ("CLTV / Equity Incentive", loan_row["equity_cltv_score"]),
    ("LTV Position", loan_row["equity_ltv_score"]),
    ("Prior Payment History", loan_row["payment_history_score"]),
    ("Pre-default Payment Timing", loan_row["payment_timing_score"]),
    ("Current Delinquency Window", loan_row["dq_score"]),
    ("Income Stability", loan_row["income_stability_score"]),
    ("Rate Incentive to Retain", loan_row["rate_incentive_score"]),
    ("Redelivery Economics", loan_row["redelivery_econ_score"]),
    ("Macro Stress", loan_row["macro_score"]),
    ("Property Trend", loan_row["property_trend_score"]),
    ("Occupancy", loan_row["occupancy_score"]),
]
factor_df = pd.DataFrame(factor_rows, columns=["factor", "score"]).sort_values("score", ascending=False)
st.bar_chart(factor_df.set_index("factor"))

st.markdown(f"""
**Loan ID:** {loan_row['loan_id']}  
**Re-performance Probability:** {loan_row['reperform_probability']:.1%}  
**Partial Claim Probability:** {loan_row['partial_claim_probability']:.1%}  
**Expected EBO Success:** {loan_row['expected_ebo_success']:.1%}  
**Current Estimated Redelivery Gain/Loss:** {loan_row['current_redelivery_gain_bps']:.0f} bps  
**120d Estimated Redelivery Gain/Loss:** {loan_row['forecast_redelivery_gain_bps_120d']:.0f} bps  
**Suggested Action:** {loan_row['recommendation']}  
**Regulatory Guardrail:** {loan_row['regulatory_guardrail']}  
**Why it scored this way:** Strongest drivers were {', '.join(factor_df.head(3)['factor'].tolist())}.  
""")


# =========================
# Data dictionary
# =========================
with st.expander("Prototype data fields to include"):
    st.markdown(
        """
**Core candidate fields**
- `loan_id`
- `ltv`
- `cltv`
- `days_delinquent`
- `current_note_rate`
- `occupancy_status`

**Borrower behavior**
- `on_time_payments_12m`
- `partial_pay_count_12m`
- `broken_promise_count_12m`
- `avg_days_late_predefault`

**Borrower stability**
- `income_type`
- `months_on_job`

**Macro overlays**
- `gas_price_change_pct_90d`
- `unemployment_change_pct_90d`
- `hpi_change_pct_12m`

**Derived by the tool**
- `forecast_pmms_120d`
- `estimated_current_price`
- `estimated_forecast_price_120d`
- `current_redelivery_gain_bps`
- `forecast_redelivery_gain_bps_120d`

**Optional future enhancements**
- DTI / residual income proxy
- escrow stress / payment shock after tax and insurance reset
- call/contact success rate
- hardship reason coding
- prior mod / prior partial claim history
- regional affordability stress index
- promise-to-pay kept ratio
- current forbearance / trial plan status
- true capital markets pricing logic rather than demo price proxies
"""
    )


# =========================
# Sample borrower csv for easy copy/paste
# =========================
with st.expander("Sample borrower CSV"):
    sample_csv = candidate_df[[
        "loan_id", "ltv", "cltv", "on_time_payments_12m", "partial_pay_count_12m",
        "broken_promise_count_12m", "avg_days_late_predefault", "days_delinquent",
        "income_type", "months_on_job", "current_note_rate", "gas_price_change_pct_90d",
        "unemployment_change_pct_90d", "hpi_change_pct_12m", "occupancy_status"
    ]].to_csv(index=False)
    st.code(sample_csv, language="csv")


# =========================
# Download outputs
# =========================
output_buffer = io.BytesIO()
with pd.ExcelWriter(output_buffer, engine="xlsxwriter") as writer:
    score_df.to_excel(writer, index=False, sheet_name="All Loans Scored")
    screened.to_excel(writer, index=False, sheet_name="Recommended EBO")
    recent_pmms.to_excel(writer, index=False, sheet_name="Recent PMMS Used")

st.download_button(
    label="Download Scored Output",
    data=output_buffer.getvalue(),
    file_name="ebo_candidate_scoring_output.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

st.caption(
    "Prototype note: redelivery pricing here is a demo proxy driven by note rate versus market rate. "
    "For production use, replace with actual capital markets pricing and execution logic."
)

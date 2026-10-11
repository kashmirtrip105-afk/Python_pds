
import math
import os
from datetime import datetime
from io import BytesIO

import numpy as np
import joblib
import pandas as pd
import streamlit as st
from scipy import stats

try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except Exception:
    px = None
    go = None
    HAS_PLOTLY = False

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    HAS_REPORTLAB = True
except Exception:
    colors = None
    A4 = None
    getSampleStyleSheet = None
    ParagraphStyle = None
    inch = None
    SimpleDocTemplate = None
    Paragraph = None
    Spacer = None
    Table = None
    TableStyle = None
    HAS_REPORTLAB = False

import config
from modules import database_manager as dbm
from modules import ml_model
from modules import statistical_analysis as statx
from modules.custom_utils import DatabaseError

st.set_page_config(page_title="Indian Water Quality Dashboard", layout="wide")

from math import isnan

def _status_for_metric(metric: str, value: float) -> tuple[str, str]:
    try:
        v = float(value)
    except Exception:
        return ("gray", "Unknown")
    metric = metric.lower()
    if metric == "ph":
        if 6.5 <= v <= 8.5: return ("#1a7f37", "Good")
        if 6.0 <= v <= 9.0: return ("#b7791f", "Moderate")
        return ("#c62828", "Poor")
    if metric == "dissolved oxygen":
        if v >= 7: return ("#1a7f37", "Good")
        if v >= 5: return ("#b7791f", "Moderate")
        return ("#c62828", "Poor")
    if metric == "turbidity":
        if v <= 5: return ("#1a7f37", "Good")
        if v <= 10: return ("#b7791f", "Moderate")
        return ("#c62828", "Poor")
    if metric == "water temperature":
        if 20 <= v <= 30: return ("#1a7f37", "Good")
        if 15 <= v <= 35: return ("#b7791f", "Moderate")
        return ("#c62828", "Poor")
    if metric == "water level":
        if v >= 4: return ("#1a7f37", "Good")
        if v >= 2.5: return ("#b7791f", "Moderate")
        return ("#c62828", "Poor")
    if metric == "bod":
        if v <= 3: return ("#1a7f37", "Good")
        if v <= 6: return ("#b7791f", "Moderate")
        return ("#c62828", "Poor")
    if metric == "tds":
        if v <= 500: return ("#1a7f37", "Good")
        if v <= 1000: return ("#b7791f", "Moderate")
        return ("#c62828", "Poor")
    if metric == "wqi":
        if v >= 80: return ("#1a7f37", "Good")
        if v >= 60: return ("#b7791f", "Moderate")
        return ("#c62828", "Poor")
    return ("gray", "Unknown")

def _metric_card(label: str, value: str, status_color: str, status: str, icon: str):
    st.markdown(
        f"""
        <div style="
            padding: 1rem 1.1rem;
            border: 1px solid rgba(0,0,0,0.08);
            border-radius: 18px;
            box-shadow: 0 6px 18px rgba(0,0,0,0.04);
            background: white;
            min-height: 118px;
        ">
            <div style="font-size: 0.92rem; color: #475569; margin-bottom: .25rem;">{icon} {label}</div>
            <div style="font-size: 1.8rem; font-weight: 700; line-height: 1.1;">{value}</div>
            <div style="margin-top: .45rem; font-size: 0.88rem; font-weight: 600; color: {status_color};">
                ● {status}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def _set_page_css():
    st.markdown("""
    <style>
    .block-container {padding-top: 1.3rem; padding-bottom: 1.5rem;}
    h1, h2, h3 {letter-spacing: -0.02em;}
    .stTabs [data-baseweb="tab-list"] {gap: .45rem;}
    .stTabs [data-baseweb="tab"] {padding: .45rem .7rem; border-radius: 999px;}
    </style>
    """, unsafe_allow_html=True)

RIVER_VALUES = {
    "Tapi": {
        "location": "Surat, Gujarat",
        "station": "Adajan Monitoring Station",
        "ph": 7.4,
        "water_temperature_c": 28.6,
        "water_level_m": 4.2,
        "bod_mg_l": 3.8,
        "tds_mg_l": 420,
        "dissolved_oxygen_mg_l": 7.6,
        "turbidity_ntu": 4.1,
    },
    "Ganga": {
        "location": "Haridwar, Uttarakhand",
        "station": "Har Ki Pauri Monitoring Station",
        "ph": 7.8,
        "water_temperature_c": 24.8,
        "water_level_m": 6.5,
        "bod_mg_l": 2.4,
        "tds_mg_l": 310,
        "dissolved_oxygen_mg_l": 8.9,
        "turbidity_ntu": 2.2,
    },
    "Yamuna": {
        "location": "Prayagraj, Uttar Pradesh",
        "station": "Sangam Monitoring Station",
        "ph": 7.1,
        "water_temperature_c": 27.9,
        "water_level_m": 3.6,
        "bod_mg_l": 5.9,
        "tds_mg_l": 680,
        "dissolved_oxygen_mg_l": 6.8,
        "turbidity_ntu": 8.7,
    },
    "Narmada": {
        "location": "Jabalpur, Madhya Pradesh",
        "station": "Bhedaghat Monitoring Station",
        "ph": 7.9,
        "water_temperature_c": 25.1,
        "water_level_m": 5.9,
        "bod_mg_l": 2.1,
        "tds_mg_l": 260,
        "dissolved_oxygen_mg_l": 9.1,
        "turbidity_ntu": 1.9,
    },
    "Sabarmati": {
        "location": "Ahmedabad, Gujarat",
        "station": "Riverfront Monitoring Station",
        "ph": 7.0,
        "water_temperature_c": 29.4,
        "water_level_m": 2.8,
        "bod_mg_l": 6.8,
        "tds_mg_l": 890,
        "dissolved_oxygen_mg_l": 5.8,
        "turbidity_ntu": 10.2,
    },
    "Godavari": {
        "location": "Nashik, Maharashtra",
        "station": "Ramkund Monitoring Station",
        "ph": 7.5,
        "water_temperature_c": 26.0,
        "water_level_m": 4.7,
        "bod_mg_l": 3.1,
        "tds_mg_l": 380,
        "dissolved_oxygen_mg_l": 8.0,
        "turbidity_ntu": 3.8,
    },
    "Krishna": {
        "location": "Vijayawada, Andhra Pradesh",
        "station": "Prakasam Barrage Monitoring Station",
        "ph": 7.3,
        "water_temperature_c": 27.3,
        "water_level_m": 4.4,
        "bod_mg_l": 4.2,
        "tds_mg_l": 510,
        "dissolved_oxygen_mg_l": 7.4,
        "turbidity_ntu": 5.3,
    },
    "Kaveri": {
        "location": "Mysuru, Karnataka",
        "station": "Srirangapatna Monitoring Station",
        "ph": 7.8,
        "water_temperature_c": 24.6,
        "water_level_m": 6.2,
        "bod_mg_l": 2.7,
        "tds_mg_l": 290,
        "dissolved_oxygen_mg_l": 8.7,
        "turbidity_ntu": 2.4,
    },
    "Brahmaputra": {
        "location": "Guwahati, Assam",
        "station": "Fancy Bazar Monitoring Station",
        "ph": 7.6,
        "water_temperature_c": 23.9,
        "water_level_m": 7.1,
        "bod_mg_l": 2.9,
        "tds_mg_l": 340,
        "dissolved_oxygen_mg_l": 9.2,
        "turbidity_ntu": 1.9,
    },
    "Mahanadi": {
        "location": "Cuttack, Odisha",
        "station": "Mahanadi Barrage Monitoring Station",
        "ph": 7.2,
        "water_temperature_c": 28.1,
        "water_level_m": 5.0,
        "bod_mg_l": 3.4,
        "tds_mg_l": 450,
        "dissolved_oxygen_mg_l": 8.1,
        "turbidity_ntu": 3.4,
    },
}
RIVER_ORDER = list(RIVER_VALUES.keys())

CRUD_COLUMNS = [
    "Record ID",
    "River Name",
    "Location",
    "pH",
    "Dissolved Oxygen",
    "Turbidity",
    "Water Temperature",
    "Water Level",
    "BOD",
    "TDS",
    "WQI Score",
    "Created At",
]


def compute_wqi(ph, do, turbidity, bod, tds, temp, level):
    score = 100
    score -= abs(ph - 7.2) * 8
    score -= max(0, 8.5 - do) * 6
    score -= turbidity * 2.2
    score -= bod * 4.5
    score -= max(0, tds - 300) / 40
    score -= max(0, temp - 28) * 1.5
    score -= max(0, 3.5 - level) * 2
    score = max(0, min(100, round(score, 1)))
    if score >= 80:
        grade = "Excellent"
    elif score >= 65:
        grade = "Good"
    elif score >= 45:
        grade = "Moderate"
    else:
        grade = "Poor"
    return score, grade


def local_variation(base_value, river_name, factor=1.0):
    idx = RIVER_ORDER.index(river_name) + 1
    delta = ((idx % 3) - 1) * factor
    return base_value + delta


def current_row_for_river(river_name):
    base = RIVER_VALUES[river_name]
    ph = round(local_variation(base["ph"], river_name, 0.15), 1)
    do = round(local_variation(base["dissolved_oxygen_mg_l"], river_name, 0.25), 1)
    turb = round(local_variation(base["turbidity_ntu"], river_name, 0.3), 1)
    temp = round(local_variation(base["water_temperature_c"], river_name, 0.2), 1)
    level = round(local_variation(base["water_level_m"], river_name, 0.2), 1)
    bod = round(local_variation(base["bod_mg_l"], river_name, 0.15), 1)
    tds = int(round(local_variation(base["tds_mg_l"], river_name, 12)))
    wqi_score, wqi_grade = compute_wqi(ph, do, turb, bod, tds, temp, level)
    return {
        "River": river_name,
        "Location": base["location"],
        "Station": base["station"],
        "pH": ph,
        "Dissolved Oxygen (mg/L)": do,
        "Turbidity (NTU)": turb,
        "Water Temperature (°C)": temp,
        "Water Level (m)": level,
        "BOD (mg/L)": bod,
        "TDS (mg/L)": tds,
        "WQI Score": wqi_score,
        "WQI Category": wqi_grade,
    }


def _safe_db_read_all():
    try:
        dbm.init_db()
        return dbm.read_all()
    except Exception as exc:
        st.error(f"Failed to load records from database: {exc}")
        return pd.DataFrame(columns=[
            "id", "timestamp", "station", "temperature_c", "ph",
            "dissolved_oxygen_mgl", "conductivity_us_cm", "turbidity_ntu", "contamination_event"
        ])


_DB_TO_UI_COLUMNS = {
    "record_id": "Record ID",
    "river_name": "River Name",
    "location": "Location",
    "ph": "pH",
    "dissolved_oxygen": "Dissolved Oxygen",
    "turbidity": "Turbidity",
    "water_temperature": "Water Temperature",
    "water_level": "Water Level",
    "bod": "BOD",
    "tds": "TDS",
    "wqi_score": "WQI Score",
    "created_at": "Created At",
}
_UI_TO_DB_COLUMNS = {v: k for k, v in _DB_TO_UI_COLUMNS.items()}


def _safe_crud_db_read_all() -> pd.DataFrame:
    """Reads all interactive CRUD records from the real SQLite table
    (river_quality_records) - this is the single source of truth for the
    CRUD tab, the stats tab's river subsets, and the preprocessing tab."""
    try:
        dbm.init_db()
        db_df = dbm.crud_read_all()
    except Exception as exc:
        st.error(f"Failed to load records from database: {exc}")
        db_df = pd.DataFrame(columns=list(_DB_TO_UI_COLUMNS.keys()))
    if db_df.empty:
        return pd.DataFrame(columns=CRUD_COLUMNS)
    return db_df.rename(columns=_DB_TO_UI_COLUMNS).reindex(columns=CRUD_COLUMNS)


def _safe_river_subset(river: str) -> pd.DataFrame:
    combined = _safe_crud_db_read_all()
    combined = combined[combined["River Name"] == river].copy()
    if combined.empty:
        base = current_row_for_river(river)
        rows = []
        rng = np.random.default_rng(abs(hash(river)) % (2**32))
        for i in range(25):
            rows.append({
                "pH": float(np.clip(rng.normal(base["pH"], 0.12), 6.0, 9.0)),
                "Dissolved Oxygen": float(np.clip(rng.normal(base["Dissolved Oxygen (mg/L)"], 0.35), 3.0, 12.0)),
                "Turbidity": float(np.clip(rng.normal(base["Turbidity (NTU)"], 0.45), 0.1, 20.0)),
                "Water Temperature": float(np.clip(rng.normal(base["Water Temperature (°C)"], 0.4), 0.0, 45.0)),
                "Water Level": float(np.clip(rng.normal(base["Water Level (m)"], 0.3), 0.0, 20.0)),
                "BOD": float(np.clip(rng.normal(base["BOD (mg/L)"], 0.25), 0.0, 20.0)),
                "TDS": float(np.clip(rng.normal(base["TDS (mg/L)"], 18), 0.0, 5000.0)),
            })
        combined = pd.DataFrame(rows)
    return combined


def render_dashboard():
    _set_page_css()
    st.title("🇮🇳 Indian Water Quality Monitoring Dashboard")
    st.caption("River-wise water quality overview with Indian monitoring stations, city/state details, and live quality indicators.")
    river_name = st.selectbox("Select River", RIVER_ORDER, index=RIVER_ORDER.index("Tapi"), key="dashboard_river")
    row = current_row_for_river(river_name)
    city, state = [x.strip() for x in row["Location"].split(",")]

    st.info(f"River: {river_name} | Monitoring Station: {row['Station']} | City: {city} | State: {state}")

    cols1 = st.columns(4)
    cards1 = [
        ("pH", row["pH"], *_status_for_metric("ph", row["pH"]), "🧪", f"{row['pH']:.1f}"),
        ("Dissolved Oxygen", row["Dissolved Oxygen (mg/L)"], *_status_for_metric("dissolved oxygen", row["Dissolved Oxygen (mg/L)"]), "💧", f"{row['Dissolved Oxygen (mg/L)']:.1f} mg/L"),
        ("Turbidity", row["Turbidity (NTU)"], *_status_for_metric("turbidity", row["Turbidity (NTU)"]), "🌫", f"{row['Turbidity (NTU)']:.1f} NTU"),
        ("WQI", row["WQI Score"], *_status_for_metric("wqi", row["WQI Score"]), "📊", f"{row['WQI Score']:.1f} ({row['WQI Category']})"),
    ]
    for col, card in zip(cols1, cards1):
        with col:
            label, _, status_color, status, icon, value = card
            _metric_card(label, value, status_color, status, icon)

    st.write("")
    cols2 = st.columns(4)
    cards2 = [
        ("Water Temperature", row["Water Temperature (°C)"], *_status_for_metric("water temperature", row["Water Temperature (°C)"]), "🌡", f"{row['Water Temperature (°C)']:.1f} °C"),
        ("Water Level", row["Water Level (m)"], *_status_for_metric("water level", row["Water Level (m)"]), "🌊", f"{row['Water Level (m)']:.1f} m"),
        ("BOD", row["BOD (mg/L)"], *_status_for_metric("bod", row["BOD (mg/L)"]), "🧫", f"{row['BOD (mg/L)']:.1f} mg/L"),
        ("TDS", row["TDS (mg/L)"], *_status_for_metric("tds", row["TDS (mg/L)"]), "⚪", f"{row['TDS (mg/L)']:.0f} mg/L"),
    ]
    for col, card in zip(cols2, cards2):
        with col:
            label, _, status_color, status, icon, value = card
            _metric_card(label, value, status_color, status, icon)

    overall_color, overall_status = _status_for_metric("wqi", row["WQI Score"])
    st.markdown(
        f"""
        <div style="
            margin-top: 1.25rem;
            padding: 1rem 1.1rem;
            border-radius: 18px;
            background: linear-gradient(90deg, rgba(59,130,246,0.10), rgba(16,185,129,0.08));
            border: 1px solid rgba(0,0,0,0.06);
        ">
            <div style="font-size: 0.95rem; color: #475569;">Overall Water Quality Status</div>
            <div style="font-size: 1.6rem; font-weight: 800; color: {overall_color}; margin-top: .15rem;">{overall_status}</div>
            <div style="font-size: 0.95rem; color: #334155;">River-specific parameters are populated from the selected river profile.</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.write("")
    st.caption("River-specific parameters are populated from the selected river profile and current database state.")



def _crud_validation_error(message: str):
    st.error(message)
    return False

def _crud_status_for_wqi(score: float) -> tuple[str, str]:
    try:
        v = float(score)
    except Exception:
        return ("gray", "Unknown")
    if v >= 80:
        return ("#1a7f37", "Good")
    if v >= 60:
        return ("#b7791f", "Moderate")
    return ("#c62828", "Poor")

def _paginate_dataframe(df: pd.DataFrame, page_size: int, key_prefix: str = "crud"):
    if df.empty:
        return df, 1, 1
    total_pages = max(1, int(np.ceil(len(df) / page_size)))
    page = st.number_input(
        "Page",
        min_value=1,
        max_value=total_pages,
        value=1,
        step=1,
        key=f"{key_prefix}_page",
    )
    start = (page - 1) * page_size
    end = start + page_size
    return df.iloc[start:end].copy(), int(page), total_pages

def _apply_crud_filters(df: pd.DataFrame) -> pd.DataFrame:
    filtered = df.copy()

    search = st.text_input(
        "Search records (river, location, city, station)",
        key="crud_search",
        placeholder="Type to search...",
    ).strip().lower()

    colf1, colf2, colf3, colf4 = st.columns(4)
    river_filter = colf1.selectbox("Filter by River", ["All"] + RIVER_ORDER, key="crud_filter_river")
    city_filter = colf2.selectbox(
        "Filter by City",
        ["All"] + sorted({str(x).split(",")[0].strip() for x in filtered["Location"].dropna().astype(str).unique()}),
        key="crud_filter_city",
    )
    state_filter = colf3.selectbox(
        "Filter by State",
        ["All"] + sorted({str(x).split(",")[1].strip() for x in filtered["Location"].dropna().astype(str).unique() if "," in str(x)}),
        key="crud_filter_state",
    )
    status_filter = colf4.selectbox(
        "Filter by WQI Status",
        ["All", "Good", "Moderate", "Poor"],
        key="crud_filter_status",
    )

    if search:
        mask = pd.Series(False, index=filtered.index)
        for col in ["River Name", "Location"]:
            if col in filtered.columns:
                mask |= filtered[col].astype(str).str.lower().str.contains(search, na=False)
        filtered = filtered[mask].copy()

    if river_filter != "All" and "River Name" in filtered.columns:
        filtered = filtered[filtered["River Name"] == river_filter].copy()

    if city_filter != "All" and "Location" in filtered.columns:
        filtered = filtered[filtered["Location"].astype(str).str.split(",").str[0].str.strip() == city_filter].copy()

    if state_filter != "All" and "Location" in filtered.columns:
        filtered = filtered[filtered["Location"].astype(str).str.contains(state_filter, na=False)].copy()

    if status_filter != "All" and "WQI Score" in filtered.columns:
        filtered = filtered[filtered["WQI Score"].astype(str).str.contains(status_filter, case=False, na=False)].copy()

    return filtered.reset_index(drop=True)

def _to_float(value, default=np.nan):
    try:
        return float(value)
    except Exception:
        return default


def render_crud_tab():

    st.subheader("Data Management & CRUD")
    st.caption("Read, insert, update, delete, search, filter, paginate, and export records.")

    df = _safe_crud_db_read_all()
    if df.empty:
        df = pd.DataFrame(columns=CRUD_COLUMNS)

    if "WQI Score" in df.columns:
        df["WQI Score"] = df["WQI Score"].astype(str)

    filtered_df = _apply_crud_filters(df)

    st.markdown("### Read / View Data")
    page_size = st.selectbox("Rows per page", [10, 15, 20], index=1, key="crud_page_size")
    paged_df, page, total_pages = _paginate_dataframe(filtered_df, page_size, key_prefix="crud")
    st.caption(f"Showing page {page} of {total_pages} ({len(filtered_df)} filtered rows)")
    try:
        st.dataframe(paged_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download CSV",
            data=filtered_df.to_csv(index=False).encode("utf-8"),
            file_name="water_quality_data.csv",
            mime="text/csv",
            key="crud_download_csv",
        )
    except Exception as exc:
        st.error(f"Error while displaying/downloading data: {exc}")

    st.divider()
    st.markdown("### Create / Insert Data")
    with st.form("insert_form", clear_on_submit=True):
        c1, c2 = st.columns(2)
        river = c1.selectbox("River Name", RIVER_ORDER, index=RIVER_ORDER.index("Tapi"), key="crud_insert_river")
        location = c2.text_input("Location", placeholder="Adajan, Surat", key="crud_insert_location")

        c3, c4, c5, c6 = st.columns(4)
        ph = c3.number_input("pH", 0.0, 14.0, 7.4, 0.1, key="crud_insert_ph")
        do = c4.number_input("Dissolved Oxygen (mg/L)", 0.0, 20.0, 7.6, 0.1, key="crud_insert_do")
        turb = c5.number_input("Turbidity (NTU)", 0.0, 100.0, 4.0, 0.1, key="crud_insert_turb")
        temp = c6.number_input("Water Temperature (°C)", 0.0, 60.0, 28.0, 0.1, key="crud_insert_temp")

        c7, c8, c9, c10 = st.columns(4)
        level = c7.number_input("Water Level (m)", 0.0, 50.0, 4.0, 0.1, key="crud_insert_level")
        bod = c8.number_input("BOD (mg/L)", 0.0, 100.0, 3.0, 0.1, key="crud_insert_bod")
        tds = c9.number_input("TDS (mg/L)", 0.0, 5000.0, 420.0, 10.0, key="crud_insert_tds")
        # reserved for future inputs
        st.write("")

        submitted = st.form_submit_button("Insert Record")
        if submitted:
            try:
                if not river.strip():
                    _crud_validation_error("River Name cannot be empty.")
                    st.stop()
                if not location.strip():
                    _crud_validation_error("Location cannot be empty.")
                    st.stop()
                if not (0 <= ph <= 14):
                    _crud_validation_error("pH must be between 0 and 14.")
                    st.stop()
                for label, val in [
                    ("Dissolved Oxygen", do),
                    ("Turbidity", turb),
                    ("Water Temperature", temp),
                    ("Water Level", level),
                    ("BOD", bod),
                    ("TDS", tds),
                ]:
                    if val < 0:
                        _crud_validation_error(f"{label} cannot be negative.")
                        st.stop()

                wqi_score, wqi_category = compute_wqi(ph, do, turb, bod, tds, temp, level)
                db_record = {
                    "river_name": river.strip(),
                    "location": location.strip(),
                    "ph": round(float(ph), 2),
                    "dissolved_oxygen": round(float(do), 2),
                    "turbidity": round(float(turb), 2),
                    "water_temperature": round(float(temp), 2),
                    "water_level": round(float(level), 2),
                    "bod": round(float(bod), 2),
                    "tds": round(float(tds), 2),
                    "wqi_score": f"{wqi_score} ({wqi_category})",
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
                new_id = dbm.crud_create(db_record)
                st.success(f"Record inserted successfully with ID {new_id}.")
                st.rerun()
            except Exception as exc:
                st.error(f"Insert failed: {exc}")

    st.divider()
    st.markdown("### Update & Delete")
    if filtered_df.empty:
        st.info("No matching records found.")
        return

    try:
        numeric_ids = pd.to_numeric(filtered_df["Record ID"], errors="coerce").dropna()
        min_id = int(numeric_ids.min())
        max_id = int(numeric_ids.max())
        record_id = st.number_input("Select Record ID", min_value=min_id, max_value=max_id, value=min_id, step=1, key="crud_record_id")

        record = df[pd.to_numeric(df["Record ID"], errors="coerce") == int(record_id)]
        if record.empty:
            st.warning("Record ID not found.")
            return
        row = record.iloc[0]

        st.dataframe(record, use_container_width=True, hide_index=True)

        with st.form("update_form"):
            u1, u2 = st.columns(2)
            river_u = u1.selectbox(
                "River Name",
                RIVER_ORDER,
                index=RIVER_ORDER.index(str(row["River Name"])) if str(row["River Name"]) in RIVER_ORDER else 0,
                key=f"crud_update_river_{record_id}",
            )
            location_u = u2.text_input("Location", value=str(row["Location"]), key=f"crud_update_location_{record_id}")

            u3, u4, u5, u6 = st.columns(4)
            ph_u = u3.number_input("pH", 0.0, 14.0, float(row.get("pH", 7.2)), 0.1, key=f"crud_update_ph_{record_id}")
            do_u = u4.number_input("Dissolved Oxygen (mg/L)", 0.0, 20.0, float(row.get("Dissolved Oxygen", 7.5)), 0.1, key=f"crud_update_do_{record_id}")
            turb_u = u5.number_input("Turbidity (NTU)", 0.0, 100.0, float(row.get("Turbidity", 3.5)), 0.1, key=f"crud_update_turb_{record_id}")
            temp_u = u6.number_input("Water Temperature (°C)", 0.0, 60.0, float(row.get("Water Temperature", 27.0)), 0.1, key=f"crud_update_temp_{record_id}")

            u7, u8, u9 = st.columns(3)
            level_u = u7.number_input("Water Level (m)", 0.0, 50.0, float(row.get("Water Level", 4.0)), 0.1, key=f"crud_update_level_{record_id}")
            bod_u = u8.number_input("BOD (mg/L)", 0.0, 100.0, float(row.get("BOD", 3.0)), 0.1, key=f"crud_update_bod_{record_id}")
            tds_u = u9.number_input("TDS (mg/L)", 0.0, 5000.0, float(row.get("TDS", 400)), 10.0, key=f"crud_update_tds_{record_id}")

            wqi_score_u, wqi_category_u = compute_wqi(ph_u, do_u, turb_u, bod_u, tds_u, temp_u, level_u)

            c_update, c_delete = st.columns(2)
            update_clicked = c_update.form_submit_button("Update Record")
            delete_clicked = c_delete.form_submit_button("Delete Record")

            if update_clicked:
                try:
                    if not river_u.strip():
                        _crud_validation_error("River Name cannot be empty.")
                        st.stop()
                    if not location_u.strip():
                        _crud_validation_error("Location cannot be empty.")
                        st.stop()
                    if not (0 <= ph_u <= 14):
                        _crud_validation_error("pH must be between 0 and 14.")
                        st.stop()
                    for label, val in [
                        ("Dissolved Oxygen", do_u),
                        ("Turbidity", turb_u),
                        ("Water Temperature", temp_u),
                        ("Water Level", level_u),
                        ("BOD", bod_u),
                        ("TDS", tds_u),
                    ]:
                        if val < 0:
                            _crud_validation_error(f"{label} cannot be negative.")
                            st.stop()

                    fields = {
                        "river_name": river_u.strip(),
                        "location": location_u.strip(),
                        "ph": round(float(ph_u), 2),
                        "dissolved_oxygen": round(float(do_u), 2),
                        "turbidity": round(float(turb_u), 2),
                        "water_temperature": round(float(temp_u), 2),
                        "water_level": round(float(level_u), 2),
                        "bod": round(float(bod_u), 2),
                        "tds": round(float(tds_u), 2),
                        "wqi_score": f"{wqi_score_u} ({wqi_category_u})",
                    }
                    dbm.crud_update(int(record_id), fields)
                    st.success(f"Record ID {int(record_id)} updated successfully.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Update failed: {exc}")

            if delete_clicked:
                try:
                    confirm = st.checkbox("I confirm that I want to permanently delete this record.", key=f"confirm_delete_{record_id}")
                    if not confirm:
                        st.warning("Please confirm deletion before proceeding.")
                        st.stop()
                    dbm.crud_delete(int(record_id))
                    st.success(f"Record ID {int(record_id)} deleted successfully.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Delete failed: {exc}")

    except Exception as exc:
        st.error(f"Error in update/delete section: {exc}")


def _summary_stats(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=cols)
    out["Mean"] = df[cols].mean()
    out["Variance"] = df[cols].var(ddof=1)
    out["Std Deviation"] = df[cols].std(ddof=1)
    out["Median"] = df[cols].median()
    return out.round(3)



# -------- Step 3: Statistical Analysis & Hypothesis Testing helpers --------
STAT_PARAM_SPECS = [
    {"label": "pH", "column": "pH", "icon": "🧪", "safe_range": (6.5, 8.5)},
    {"label": "Dissolved Oxygen", "column": "Dissolved Oxygen", "icon": "💧", "safe_range": (5.0, 20.0)},
    {"label": "Turbidity", "column": "Turbidity", "icon": "🌫️", "safe_range": (0.0, 10.0)},
    {"label": "Water Temperature", "column": "Water Temperature", "icon": "🌡️", "safe_range": (20.0, 30.0)},
    {"label": "Water Level", "column": "Water Level", "icon": "📈", "safe_range": (2.5, 6.5)},
    {"label": "BOD", "column": "BOD", "icon": "🧫", "safe_range": (0.0, 3.0)},
    {"label": "TDS", "column": "TDS", "icon": "⚪", "safe_range": (0.0, 500.0)},
]
STAT_PARAM_OPTIONS = [spec["label"] for spec in STAT_PARAM_SPECS]
STAT_PARAM_LOOKUP = {spec["label"]: spec for spec in STAT_PARAM_SPECS}


def _stats_river_frame(river: str) -> pd.DataFrame:
    """Return a clean river-wise dataframe for all statistical views."""
    df = _safe_river_subset(river).copy()
    if df.empty:
        df = pd.DataFrame()
    df["River Name"] = river
    df["Observation"] = np.arange(1, len(df) + 1)
    for spec in STAT_PARAM_SPECS:
        col = spec["column"]
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.reset_index(drop=True)


def _safe_series(df: pd.DataFrame, column: str) -> pd.Series:
    if df.empty or column not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[column], errors="coerce").dropna().reset_index(drop=True)


def _stat_bundle(series: pd.Series, confidence: float = 0.95) -> dict:
    series = pd.to_numeric(series, errors="coerce").dropna()
    if series.empty:
        return {
            "Sample Size": 0,
            "Mean": np.nan,
            "Median": np.nan,
            "Mode": np.nan,
            "Variance": np.nan,
            "Std Dev": np.nan,
            "Minimum": np.nan,
            "Maximum": np.nan,
            "Range": np.nan,
            "CI Low": np.nan,
            "CI High": np.nan,
        }

    n = int(series.size)
    mean = float(series.mean())
    median = float(series.median())
    mode_vals = series.mode(dropna=True)
    mode = float(mode_vals.iloc[0]) if not mode_vals.empty else np.nan
    variance = float(series.var(ddof=1)) if n > 1 else 0.0
    std_dev = float(series.std(ddof=1)) if n > 1 else 0.0
    minimum = float(series.min())
    maximum = float(series.max())
    value_range = float(maximum - minimum)

    if n > 1 and std_dev > 0 and not math.isnan(std_dev):
        se = std_dev / math.sqrt(n)
        t_critical = float(stats.t.ppf((1 + confidence) / 2.0, df=n - 1))
        margin = t_critical * se
        ci_low = mean - margin
        ci_high = mean + margin
    else:
        ci_low = mean
        ci_high = mean

    return {
        "Sample Size": n,
        "Mean": mean,
        "Median": median,
        "Mode": mode,
        "Variance": variance,
        "Std Dev": std_dev,
        "Minimum": minimum,
        "Maximum": maximum,
        "Range": value_range,
        "CI Low": ci_low,
        "CI High": ci_high,
    }


def _safe_probability(series: pd.Series, safe_low: float, safe_high: float) -> tuple[float, float, float, dict]:
    """Return unsafe probability using a normal approximation + fit diagnostics."""
    series = pd.to_numeric(series, errors="coerce").dropna()
    if series.empty:
        return 0.0, np.nan, np.nan, {}

    mu = float(series.mean())
    sigma = float(series.std(ddof=1)) if len(series) > 1 else 0.0

    fit_info = {}
    try:
        fit_info = statx.fit_normal_distribution(series)
    except Exception:
        fit_info = {}

    if sigma <= 0 or math.isnan(sigma):
        return 0.0, mu, sigma, fit_info

    lower_prob = float(stats.norm.cdf(safe_low, loc=mu, scale=sigma))
    upper_prob = float(1 - stats.norm.cdf(safe_high, loc=mu, scale=sigma))
    unsafe_prob = max(0.0, min(1.0, lower_prob + upper_prob))
    return unsafe_prob, mu, sigma, fit_info


def _plot_distribution(series: pd.Series, parameter: str, river: str, bins: int):
    if not HAS_PLOTLY:
        return None
    fig = px.histogram(
        pd.DataFrame({parameter: series}),
        x=parameter,
        nbins=bins,
        marginal="box",
        histnorm="probability density",
        title=f"Distribution Plot of {parameter} — {river}",
        opacity=0.85,
    )
    fig.update_layout(template="plotly_white", bargap=0.05)
    return fig


def _plot_line(frame: pd.DataFrame, parameter: str, river: str):
    if not HAS_PLOTLY:
        return None
    fig = px.line(
        frame,
        x="Observation",
        y=parameter,
        markers=True,
        title=f"Line Chart of {parameter} Across {river} Readings",
    )
    fig.update_layout(template="plotly_white")
    return fig


def _plot_scatter(frame: pd.DataFrame, x_param: str, y_param: str, river: str):
    if not HAS_PLOTLY:
        return None
    fig = px.scatter(
        frame,
        x=x_param,
        y=y_param,
        title=f"Scatter Plot: {x_param} vs {y_param} — {river}",
    )
    fig.update_layout(template="plotly_white")
    return fig


def _plot_violin(frame_a: pd.DataFrame, frame_b: pd.DataFrame, parameter: str, river_a: str, river_b: str):
    if not HAS_PLOTLY:
        return None
    combo = pd.concat([
        frame_a[[parameter]].assign(River=river_a),
        frame_b[[parameter]].assign(River=river_b),
    ], ignore_index=True)
    fig = px.violin(
        combo,
        x="River",
        y=parameter,
        box=True,
        points="all",
        color="River",
        title=f"Violin Plot of {parameter}: {river_a} vs {river_b}",
    )
    fig.update_layout(template="plotly_white", showlegend=False)
    return fig


def _plot_density(series: pd.Series, parameter: str, river: str):
    if not HAS_PLOTLY:
        return None
    series = pd.to_numeric(series, errors="coerce").dropna()
    if series.empty:
        return None

    if len(series) > 1 and series.std(ddof=1) > 0:
        xs = np.linspace(series.min(), series.max(), 250)
        kde = stats.gaussian_kde(series)
        ys = kde(xs)
    else:
        xs = np.array(series)
        ys = np.ones_like(xs, dtype=float)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Density"))
    fig.update_layout(
        title=f"Density Plot (KDE) of {parameter} — {river}",
        xaxis_title=parameter,
        yaxis_title="Density",
        template="plotly_white",
    )
    return fig


def _plot_boxplot(frame_a: pd.DataFrame, frame_b: pd.DataFrame, parameter: str, river_a: str, river_b: str):
    if not HAS_PLOTLY:
        return None
    combo = pd.concat([
        frame_a[[parameter]].assign(River=river_a),
        frame_b[[parameter]].assign(River=river_b),
    ], ignore_index=True)
    fig = px.box(
        combo,
        x="River",
        y=parameter,
        points="all",
        color="River",
        title=f"Boxplot Comparison for {parameter}",
    )
    fig.update_layout(template="plotly_white", showlegend=False)
    return fig


def _plot_histogram(series: pd.Series, parameter: str, river: str, bins: int):
    if not HAS_PLOTLY:
        return None
    fig = px.histogram(
        pd.DataFrame({parameter: series}),
        x=parameter,
        nbins=bins,
        title=f"Histogram of {parameter} — {river}",
        opacity=0.85,
    )
    fig.update_layout(template="plotly_white", bargap=0.05)
    return fig


def _build_report_pdf(river: str, parameter: str, summary: dict, conclusion: str, ttest_result: dict | None = None) -> bytes:
    if not HAS_REPORTLAB:
        return b""

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story = []

    title_style = styles["Title"]
    body_style = styles["BodyText"]
    body_style.leading = 14

    story.append(Paragraph("Indian Water Quality Monitoring System", title_style))
    story.append(Spacer(1, 0.18 * inch))
    story.append(Paragraph(f"<b>River:</b> {river}", body_style))
    story.append(Paragraph(f"<b>Parameter:</b> {parameter}", body_style))
    story.append(Paragraph(f"<b>Generated On:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", body_style))
    story.append(Spacer(1, 0.16 * inch))

    summary_rows = [["Metric", "Value"]]
    for key in ["Sample Size", "Mean", "Median", "Mode", "Variance", "Std Dev", "Minimum", "Maximum", "Range", "CI Low", "CI High"]:
        val = summary.get(key, "")
        if isinstance(val, float):
            val = f"{val:.4f}"
        summary_rows.append([key, str(val)])

    table = Table(summary_rows, hAlign="LEFT", colWidths=[2.3 * inch, 3.9 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1d4ed8")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.HexColor("#eef2ff")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(table)
    story.append(Spacer(1, 0.18 * inch))

    story.append(Paragraph(f"<b>Conclusion:</b> {conclusion}", body_style))
    story.append(Spacer(1, 0.12 * inch))

    if ttest_result:
        story.append(Paragraph("<b>Hypothesis Test Result</b>", styles["Heading2"]))
        t_rows = [["Field", "Value"]]
        for key in ["T Statistic", "P Value", "Confidence Level", "Decision"]:
            val = ttest_result.get(key, "")
            if isinstance(val, float):
                val = f"{val:.4f}"
            t_rows.append([key, str(val)])
        t_table = Table(t_rows, hAlign="LEFT", colWidths=[2.3 * inch, 3.9 * inch])
        t_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f766e")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.HexColor("#ecfeff")]),
        ]))
        story.append(t_table)
        story.append(Spacer(1, 0.12 * inch))
        story.append(Paragraph(f"<b>Interpretation:</b> {ttest_result.get('Interpretation', '')}", body_style))

    doc.build(story)
    buf.seek(0)
    return buf.getvalue()


def _export_summary_csv(summary_df: pd.DataFrame, ttest_result: dict | None = None) -> bytes:
    if ttest_result:
        ttest_df = pd.DataFrame([ttest_result])
        combined = pd.concat([summary_df, ttest_df], axis=0, ignore_index=True, sort=False)
        return combined.to_csv(index=False).encode("utf-8")
    return summary_df.to_csv(index=False).encode("utf-8")


def render_stats_tab():
    st.subheader("Statistical Analysis & Hypothesis Testing")
    st.caption("Explore river-wise summaries, probability, charts, confidence intervals, and hypothesis testing.")

    try:
        top_left, top_mid, top_right = st.columns([1.1, 1.1, 1.0])
        river = top_left.selectbox(
            "Select River",
            RIVER_ORDER,
            index=RIVER_ORDER.index("Tapi"),
            key="stats_river_main",
        )
        parameter = top_mid.selectbox(
            "Select Parameter",
            STAT_PARAM_OPTIONS,
            index=0,
            key="stats_parameter_main",
        )
        compare_river = top_right.selectbox(
            "Compare With",
            [r for r in RIVER_ORDER if r != river],
            index=0,
            key="stats_compare_river_main",
        )
    except Exception as exc:
        st.error(f"Unable to prepare statistical controls: {exc}")
        return

    river_df = _stats_river_frame(river)
    compare_df = _stats_river_frame(compare_river)
    param_spec = STAT_PARAM_LOOKUP[parameter]
    param_col = param_spec["column"]
    safe_low, safe_high = param_spec["safe_range"]

    series = _safe_series(river_df, param_col)
    compare_series = _safe_series(compare_df, param_col)

    if series.empty:
        st.warning("No numeric data is available for the selected river and parameter.")
        return

    bundle = _stat_bundle(series)
    status_color, status = _status_for_metric(parameter.lower(), bundle["Mean"])
    fit_info = {}
    try:
        fit_info = statx.fit_normal_distribution(series)
    except Exception:
        fit_info = {}

    summary_df = pd.DataFrame([{
        "River": river,
        "Parameter": parameter,
        "Sample Size": bundle["Sample Size"],
        "Mean": round(bundle["Mean"], 4) if pd.notna(bundle["Mean"]) else np.nan,
        "Median": round(bundle["Median"], 4) if pd.notna(bundle["Median"]) else np.nan,
        "Mode": round(bundle["Mode"], 4) if pd.notna(bundle["Mode"]) else np.nan,
        "Variance": round(bundle["Variance"], 4) if pd.notna(bundle["Variance"]) else np.nan,
        "Std Dev": round(bundle["Std Dev"], 4) if pd.notna(bundle["Std Dev"]) else np.nan,
        "Minimum": round(bundle["Minimum"], 4) if pd.notna(bundle["Minimum"]) else np.nan,
        "Maximum": round(bundle["Maximum"], 4) if pd.notna(bundle["Maximum"]) else np.nan,
        "Range": round(bundle["Range"], 4) if pd.notna(bundle["Range"]) else np.nan,
        "CI Low": round(bundle["CI Low"], 4) if pd.notna(bundle["CI Low"]) else np.nan,
        "CI High": round(bundle["CI High"], 4) if pd.notna(bundle["CI High"]) else np.nan,
        "Status": status,
    }])

    summary_tabs = st.tabs([
        "Summary Statistics",
        "Visualizations",
        "Probability & Sampling",
        "Hypothesis Testing",
        "Export Results",
    ])

    with summary_tabs[0]:
        st.markdown(f"### Summary Statistics for {river} — {parameter}")
        city_state = current_row_for_river(river)
        try:
            city, state = [x.strip() for x in city_state["Location"].split(",")]
        except Exception:
            city, state = city_state.get("Location", "Unknown"), ""
        st.info(
            f"River: {river} | Monitoring Station: {city_state['Station']} | City: {city} | State: {state}"
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Mean", f"{bundle['Mean']:.3f}")
        c2.metric("Median", f"{bundle['Median']:.3f}")
        c3.metric("Mode", f"{bundle['Mode']:.3f}" if pd.notna(bundle["Mode"]) else "N/A")

        r1, r2, r3 = st.columns(3)
        r1.metric("Variance", f"{bundle['Variance']:.3f}")
        r2.metric("Std Dev", f"{bundle['Std Dev']:.3f}")
        r3.metric("Sample Size", f"{int(bundle['Sample Size'])}")

        r4, r5, r6 = st.columns(3)
        r4.metric("Minimum", f"{bundle['Minimum']:.3f}")
        r5.metric("Maximum", f"{bundle['Maximum']:.3f}")
        r6.metric("Range", f"{bundle['Range']:.3f}")

        r7, r8, r9 = st.columns(3)
        r7.metric("95% CI Low", f"{bundle['CI Low']:.3f}")
        r8.metric("95% CI High", f"{bundle['CI High']:.3f}")
        r9.metric("Water Quality Status", status)

        st.divider()
        if pd.notna(bundle["CI Low"]) and pd.notna(bundle["CI High"]):
            if status == "Good":
                st.success(
                    f"We are 95% confident that the average {parameter} of the {river} River lies between "
                    f"{bundle['CI Low']:.3f} and {bundle['CI High']:.3f}."
                )
            elif status == "Moderate":
                st.warning(
                    f"We are 95% confident that the average {parameter} of the {river} River lies between "
                    f"{bundle['CI Low']:.3f} and {bundle['CI High']:.3f}."
                )
            else:
                st.error(
                    f"We are 95% confident that the average {parameter} of the {river} River lies between "
                    f"{bundle['CI Low']:.3f} and {bundle['CI High']:.3f}."
                )

        if fit_info:
            fit_df = pd.DataFrame([fit_info])
            st.markdown("#### Distribution Fit Diagnostics")
            st.dataframe(fit_df, use_container_width=True, hide_index=True)

        st.markdown("#### Detailed Statistics Table")
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        last_test = st.session_state.get("stats_last_test")
        with st.expander("Statistical Conclusion Panel", expanded=True):
            conclusion_text = (
                f"The selected river is {river}. The parameter analyzed is {parameter}. "
                f"The mean is {bundle['Mean']:.3f} and the standard deviation is {bundle['Std Dev']:.3f}. "
                f"The 95% confidence interval is [{bundle['CI Low']:.3f}, {bundle['CI High']:.3f}]. "
            )
            if last_test:
                conclusion_text += last_test.get("Interpretation", "A hypothesis test was completed.")
            else:
                conclusion_text += "Run the hypothesis test tab to generate a comparison between two rivers."
            st.info(conclusion_text)

    with summary_tabs[1]:
        st.markdown(f"### Interactive Visualizations — {river} ({parameter})")
        try:
            viz_tabs = st.tabs([
                "Distribution",
                "Line Chart",
                "Scatter Plot",
                "Violin Plot",
                "Density Plot",
                "Boxplot",
                "Histogram",
            ])

            viz_frame = river_df.copy()
            viz_frame["Observation"] = np.arange(1, len(viz_frame) + 1)

            with viz_tabs[0]:
                bins = st.slider("Number of bins", 5, 50, 20, key="stats_dist_bins")
                if HAS_PLOTLY:
                    fig = _plot_distribution(series, parameter, river, bins)
                    if fig is not None:
                        fig.add_vrect(x0=safe_low, x1=safe_high, fillcolor="green", opacity=0.08, line_width=0)
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Plotly is not available in this environment.")

            with viz_tabs[1]:
                if HAS_PLOTLY:
                    fig = _plot_line(viz_frame, param_col, river)
                    if fig is not None:
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Plotly is not available in this environment.")

            with viz_tabs[2]:
                scatter_left, scatter_right = st.columns(2)
                x_param = scatter_left.selectbox(
                    "X Parameter",
                    STAT_PARAM_OPTIONS,
                    index=0,
                    key="stats_scatter_x",
                )
                y_param = scatter_right.selectbox(
                    "Y Parameter",
                    [p for p in STAT_PARAM_OPTIONS if p != x_param],
                    index=0,
                    key="stats_scatter_y",
                )
                x_col = STAT_PARAM_LOOKUP[x_param]["column"]
                y_col = STAT_PARAM_LOOKUP[y_param]["column"]
                scatter_df = viz_frame[[x_col, y_col]].rename(columns={x_col: x_param, y_col: y_param})
                if HAS_PLOTLY:
                    fig = _plot_scatter(scatter_df, x_param, y_param, river)
                    if fig is not None:
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Plotly is not available in this environment.")

            with viz_tabs[3]:
                if HAS_PLOTLY:
                    compare_violin_df_a = _stats_river_frame(river)
                    compare_violin_df_b = _stats_river_frame(compare_river)
                    violin_fig = _plot_violin(compare_violin_df_a, compare_violin_df_b, param_col, river, compare_river)
                    if violin_fig is not None:
                        st.plotly_chart(violin_fig, use_container_width=True)
                else:
                    st.warning("Plotly is not available in this environment.")

            with viz_tabs[4]:
                if HAS_PLOTLY:
                    density_fig = _plot_density(series, parameter, river)
                    if density_fig is not None:
                        st.plotly_chart(density_fig, use_container_width=True)
                    else:
                        st.info("Not enough data to build a density plot.")
                else:
                    st.warning("Plotly is not available in this environment.")

            with viz_tabs[5]:
                if HAS_PLOTLY:
                    box_fig = _plot_boxplot(river_df, compare_df, param_col, river, compare_river)
                    if box_fig is not None:
                        st.plotly_chart(box_fig, use_container_width=True)
                else:
                    st.warning("Plotly is not available in this environment.")

            with viz_tabs[6]:
                bins = st.slider("Histogram bins", 5, 50, 20, key="stats_hist_bins")
                if HAS_PLOTLY:
                    hist_fig = _plot_histogram(series, parameter, river, bins)
                    if hist_fig is not None:
                        st.plotly_chart(hist_fig, use_container_width=True)
                else:
                    st.warning("Plotly is not available in this environment.")
        except Exception as exc:
            st.error(f"Interactive visualizations failed: {exc}")

    with summary_tabs[2]:
        st.markdown("### Probability & Sampling")
        if len(series) < 6:
            # st.slider requires min_value < max_value strictly - with 5 or
            # fewer readings available there's no meaningful range to offer,
            # so just use every available reading as the sample.
            st.info(f"Only {len(series)} reading(s) available for {river} - using all of them as the sample.")
            sample_size = len(series)
        else:
            sample_size = st.slider(
                "Random sample size",
                min_value=5,
                max_value=len(series),
                value=min(15, len(series)),
                step=1,
                key="stats_sample_size_v2",
            )
        sample_size = min(sample_size, len(series))
        sample = series.sample(sample_size, random_state=42)

        unsafe_prob, mu, sigma, fit_details = _safe_probability(sample, safe_low, safe_high)
        st.write(f"Using a random sample of **{sample_size}** readings from **{river}**.")
        st.write(f"Safe range for **{parameter}**: **{safe_low} to {safe_high}**")

        p1, p2, p3 = st.columns(3)
        p1.metric("Sample Mean", f"{mu:.3f}" if pd.notna(mu) else "N/A")
        p2.metric("Sample Std Dev", f"{sigma:.3f}" if pd.notna(sigma) else "N/A")
        p3.metric("Unsafe Probability", f"{unsafe_prob:.2%}")

        if unsafe_prob < 0.05:
            st.success(f"The probability that a new {parameter} reading falls outside the safe range is low.")
        elif unsafe_prob < 0.20:
            st.warning(f"The probability that a new {parameter} reading falls outside the safe range is moderate.")
        else:
            st.error(f"The probability that a new {parameter} reading falls outside the safe range is high.")

        st.markdown("#### Sample Used for Estimation")
        st.dataframe(pd.DataFrame({parameter: sample}), use_container_width=True, hide_index=True)

        if fit_details:
            st.markdown("#### Normal Distribution Fit")
            st.dataframe(pd.DataFrame([fit_details]), use_container_width=True, hide_index=True)

        st.markdown(
            f"We estimate the probability of an unsafe {parameter} reading using a normal distribution built from the selected sample."
        )

    with summary_tabs[3]:
        st.markdown("### Independent T-Test")
        test_left, test_right = st.columns(2)
        river_a = test_left.selectbox(
            "River A",
            RIVER_ORDER,
            index=RIVER_ORDER.index(river),
            key="stats_river_a",
        )
        river_b_options = [r for r in RIVER_ORDER if r != river_a]
        river_b_default = river_b_options.index(compare_river) if compare_river in river_b_options else 0
        river_b = test_right.selectbox(
            "River B",
            river_b_options,
            index=river_b_default,
            key="stats_river_b",
        )
        ttest_param = st.selectbox(
            "Metric for T-test",
            STAT_PARAM_OPTIONS,
            index=STAT_PARAM_OPTIONS.index(parameter),
            key="stats_ttest_param",
        )
        confidence_level = st.slider(
            "Confidence Level (%)",
            min_value=90,
            max_value=99,
            value=95,
            step=1,
            key="stats_confidence_level",
        )

        if st.button("Run Independent T-Test", key="stats_run_ttest"):
            try:
                df_a = _stats_river_frame(river_a)
                df_b = _stats_river_frame(river_b)
                col = STAT_PARAM_LOOKUP[ttest_param]["column"]
                series_a = _safe_series(df_a, col)
                series_b = _safe_series(df_b, col)

                if series_a.empty or series_b.empty:
                    st.warning("Not enough data to run the t-test.")
                else:
                    t_stat, p_val = stats.ttest_ind(series_a, series_b, equal_var=False, nan_policy="omit")
                    alpha = 1 - (confidence_level / 100.0)
                    decision = "Reject the null hypothesis" if p_val < alpha else "Fail to reject the null hypothesis"
                    interpretation = (
                        f"Since p-value = {p_val:.4f} is {'below' if p_val < alpha else 'above'} the alpha level of {alpha:.2f}, "
                        f"{'there is' if p_val < alpha else 'there is no'} statistically significant difference in {ttest_param} "
                        f"between the {river_a} and {river_b} rivers."
                    )

                    result = {
                        "River A": river_a,
                        "River B": river_b,
                        "Metric": ttest_param,
                        "T Statistic": float(t_stat),
                        "P Value": float(p_val),
                        "Confidence Level": int(confidence_level),
                        "Decision": decision,
                        "Interpretation": interpretation,
                    }
                    st.session_state["stats_last_test"] = result

                    q1, q2, q3 = st.columns(3)
                    q1.metric("T Statistic", f"{t_stat:.4f}")
                    q2.metric("P Value", f"{p_val:.4f}")
                    q3.metric("Confidence Level", f"{confidence_level}%")

                    result_df = pd.DataFrame([result])
                    st.dataframe(result_df, use_container_width=True, hide_index=True)

                    if p_val < alpha:
                        st.success(interpretation)
                    else:
                        st.info(interpretation)
            except Exception as exc:
                st.error(f"T-test failed: {exc}")

        last_test = st.session_state.get("stats_last_test")
        if last_test:
            st.markdown("#### Latest Hypothesis Test Summary")
            st.dataframe(pd.DataFrame([last_test]), use_container_width=True, hide_index=True)

    with summary_tabs[4]:
        st.markdown("### Export Statistical Results")
        last_test = st.session_state.get("stats_last_test")
        export_df = summary_df.copy()
        if last_test:
            export_test_df = pd.DataFrame([last_test])
        else:
            export_test_df = None

        csv_bytes = _export_summary_csv(export_df, last_test)

        st.download_button(
            label="Download Statistical Summary CSV",
            data=csv_bytes,
            file_name=f"{river}_{parameter}_statistical_summary.csv".replace(" ", "_"),
            mime="text/csv",
            key="stats_download_csv",
        )

        if HAS_REPORTLAB:
            try:
                conclusion = (
                    f"The selected river is {river}. The parameter analyzed is {parameter}. "
                    f"The mean is {bundle['Mean']:.3f} and the standard deviation is {bundle['Std Dev']:.3f}. "
                    f"The 95% confidence interval is [{bundle['CI Low']:.3f}, {bundle['CI High']:.3f}]. "
                )
                if last_test:
                    conclusion += last_test.get("Interpretation", "")
                else:
                    conclusion += "A hypothesis test has not been run yet."
                pdf_bytes = _build_report_pdf(river, parameter, bundle, conclusion, last_test)
                st.download_button(
                    label="Download Statistical Report PDF",
                    data=pdf_bytes,
                    file_name=f"{river}_{parameter}_statistical_report.pdf".replace(" ", "_"),
                    mime="application/pdf",
                    key="stats_download_pdf",
                )
            except Exception as exc:
                st.warning(f"PDF export could not be created: {exc}")
        else:
            st.warning("PDF export is unavailable because the PDF library is not installed.")

        st.markdown("#### Export Preview")
        st.dataframe(export_df, use_container_width=True, hide_index=True)
        if export_test_df is not None:
            st.dataframe(export_test_df, use_container_width=True, hide_index=True)
        else:
            st.info("Run a hypothesis test first to include test results in the export.")
# -------- End Step 3 helpers --------
def _render_stat_card(label, value, delta=None):
    st.markdown(
        f"""
        <div style="
            padding: 1rem 1.05rem;
            border-radius: 18px;
            border: 1px solid rgba(148,163,184,.25);
            background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
            box-shadow: 0 6px 18px rgba(15,23,42,.05);
            min-height: 112px;
        ">
            <div style="font-size:.88rem;color:#64748b;font-weight:600;margin-bottom:.3rem;">{label}</div>
            <div style="font-size:1.65rem;font-weight:800;color:#0f172a;line-height:1.1;">{value}</div>
            <div style="margin-top:.45rem;font-size:.82rem;color:#64748b;">{delta or ''}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _safe_numeric_series(df, col):
    return pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(dtype=float)


def _build_cleaning_report(before_df, after_df, outlier_counts, missing_before, missing_after):
    rows_before = len(before_df)
    rows_after = len(after_df)
    return pd.DataFrame([
        ["Rows Before Cleaning", rows_before],
        ["Rows After Cleaning", rows_after],
        ["Rows Removed", max(rows_before - rows_after, 0)],
        ["Missing Values Before", int(missing_before)],
        ["Missing Values After", int(missing_after)],
        ["Duplicate Rows Removed", int(max(before_df.duplicated().sum(), 0))],
        ["Outliers Removed", int(sum(outlier_counts.values()))],
        ["Cleaning Status", "Completed" if rows_after <= rows_before else "Check Data"],
    ], columns=["Metric", "Value"])


def render_preprocessing_tab():
    st.subheader("Preprocessing & Visualization")
    st.caption("Clean data with NumPy-based outlier filtering, then explore river-wise trends and correlations.")

    try:
        df = _safe_crud_db_read_all()

        if df.empty:
            st.info("No dataset available yet. Add records in the CRUD tab first.")
            return

        st.markdown("### Dataset Preview")
        preview1, preview2, preview3, preview4 = st.columns(4)
        preview1.metric("Rows", len(df))
        preview2.metric("Columns", len(df.columns))
        preview3.metric("Missing Values", int(df.isna().sum().sum()))
        preview4.metric("Duplicate Rows", int(df.duplicated().sum()))
        with st.expander("Show first rows"):
            st.dataframe(df.head(10), use_container_width=True, hide_index=True)
        with st.expander("Show last rows"):
            st.dataframe(df.tail(10), use_container_width=True, hide_index=True)

        numeric_map = {
            "pH": "pH",
            "Dissolved Oxygen (mg/L)": "Dissolved Oxygen",
            "Turbidity (NTU)": "Turbidity",
            "Water Temperature (°C)": "Water Temperature",
            "Water Level (m)": "Water Level",
            "BOD (mg/L)": "BOD",
            "TDS (mg/L)": "TDS",
        }

        st.markdown("### Data Cleaning")
        cols = st.multiselect(
            "Select columns to clean",
            list(numeric_map.keys()),
            default=["pH", "Dissolved Oxygen (mg/L)", "Turbidity (NTU)"],
            key="preprocess_cols",
        )

        before_rows = len(df)
        missing_before = int(df.isna().sum().sum())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Rows Before Cleaning", before_rows)

        if st.button("Run Data Cleaning", key="preprocess_run_cleaning"):
            try:
                cleaned = df.copy()
                cleaned = cleaned.drop_duplicates().reset_index(drop=True)
                selected_cols = [numeric_map[c] for c in cols if numeric_map[c] in cleaned.columns]

                outlier_counts = {}
                for col in selected_cols:
                    cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
                    cleaned[col] = cleaned[col].fillna(cleaned[col].median())
                    arr = cleaned[col].to_numpy(dtype=float)
                    if arr.size == 0:
                        outlier_counts[col] = 0
                        continue

                    mean = np.nanmean(arr)
                    std = np.nanstd(arr)
                    if std and not np.isnan(std) and std > 0:
                        z = np.abs((arr - mean) / std)
                        outlier_mask = z > 3
                        outlier_counts[col] = int(np.sum(outlier_mask))
                        cleaned = cleaned.loc[~outlier_mask].copy()
                    else:
                        outlier_counts[col] = 0

                cleaned = cleaned.dropna(subset=selected_cols).reset_index(drop=True)
                after_rows = len(cleaned)
                missing_after = int(cleaned.isna().sum().sum())

                c2.metric("Total Rows After Cleaning", after_rows)
                c3.metric("Rows Removed", max(before_rows - after_rows, 0))
                c4.metric("Missing Values Handled", max(missing_before - missing_after, 0))

                st.session_state["cleaned_preview_df"] = cleaned.copy()
                st.session_state["cleaning_report_df"] = _build_cleaning_report(
                    df, cleaned, outlier_counts, missing_before, missing_after
                )
                st.session_state["cleaned_outlier_counts"] = outlier_counts
                st.session_state["cleaned_before_df"] = df.copy()
                st.session_state["cleaned_after_df"] = cleaned.copy()

                st.success("Data cleaning completed successfully.")

                st.markdown("### Cleaning Report")
                report_df = st.session_state["cleaning_report_df"]
                st.dataframe(report_df, use_container_width=True, hide_index=True)

                csv_bytes = cleaned.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Cleaned Dataset (CSV)",
                    data=csv_bytes,
                    file_name="cleaned_water_quality_dataset.csv",
                    mime="text/csv",
                    key="download_cleaned_csv",
                )
                try:
                    xlsx_buf = BytesIO()
                    with pd.ExcelWriter(xlsx_buf, engine="openpyxl") as writer:
                        cleaned.to_excel(writer, index=False, sheet_name="CleanedData")
                    st.download_button(
                        "Download Cleaned Dataset (Excel)",
                        data=xlsx_buf.getvalue(),
                        file_name="cleaned_water_quality_dataset.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="download_cleaned_xlsx",
                    )
                except Exception as exc:
                    st.warning(f"Excel export unavailable: {exc}")

                st.markdown("### Outlier Highlighting")
                outlier_counts = st.session_state.get("cleaned_outlier_counts", {})
                if outlier_counts:
                    o1, o2, o3, o4 = st.columns(4)
                    for col, value, box in zip(
                        list(outlier_counts.keys())[:4],
                        list(outlier_counts.values())[:4],
                        [o1, o2, o3, o4],
                    ):
                        box.metric(col, value)
                outlier_records = []
                for col in selected_cols:
                    if col in df.columns:
                        series = pd.to_numeric(df[col], errors="coerce")
                        med = np.nanmedian(series.to_numpy(dtype=float))
                        mad = np.nanmedian(np.abs(series.to_numpy(dtype=float) - med))
                        if mad == 0 or np.isnan(mad):
                            continue
                        robust_z = 0.6745 * (series - med) / mad
                        outlier_records.append(df.loc[robust_z.abs() > 3.5])
                if outlier_records:
                    outlier_df = pd.concat(outlier_records).drop_duplicates()
                    st.dataframe(outlier_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No obvious outliers detected in the selected columns.")

                st.markdown("### Before vs After Comparison")
                bc1, bc2 = st.columns(2)
                with bc1:
                    st.dataframe(df.head(10), use_container_width=True, hide_index=True)
                with bc2:
                    st.dataframe(cleaned.head(10), use_container_width=True, hide_index=True)

                try:
                    compare = pd.DataFrame({
                        "Before": [len(df), int(df.isna().sum().sum()), int(df.duplicated().sum())],
                        "After": [len(cleaned), int(cleaned.isna().sum().sum()), int(cleaned.duplicated().sum())]
                    }, index=["Rows", "Missing Values", "Duplicates"])
                    fig = None
                    if HAS_PLOTLY:
                        fig = px.bar(compare.reset_index(), x="index", y=["Before", "After"], barmode="group",
                                     title="Before vs After Cleaning Comparison", labels={"index": "Metric", "value": "Count"})
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.dataframe(compare, use_container_width=True)
                except Exception as exc:
                    st.warning(f"Comparison chart unavailable: {exc}")

                st.markdown("### Dataset After Cleaning")
                st.dataframe(cleaned, use_container_width=True, hide_index=True)
            except Exception as exc:
                st.error(f"Data cleaning failed: {exc}")
        else:
            cleaned = st.session_state.get("cleaned_preview_df", df.copy())
            c2.metric("Total Rows After Cleaning", len(cleaned))
            c3.metric("Rows Removed", max(before_rows - len(cleaned), 0))
            c4.metric("Missing Values Handled", max(missing_before - int(cleaned.isna().sum().sum()), 0))
            if "cleaning_report_df" in st.session_state:
                st.markdown("### Cleaning Report")
                st.dataframe(st.session_state["cleaning_report_df"], use_container_width=True, hide_index=True)
            st.download_button(
                "Download Cleaned Dataset (CSV)",
                data=cleaned.to_csv(index=False).encode("utf-8"),
                file_name="cleaned_water_quality_dataset.csv",
                mime="text/csv",
                key="download_cleaned_csv_idle",
            )

        st.divider()
        st.markdown("### Time-Series Analysis")
        river = st.selectbox("Select River for Trends", RIVER_ORDER, index=RIVER_ORDER.index("Tapi"), key="prep_river")
        river_df = _safe_river_subset(river).copy()

        try:
            if river_df.empty:
                st.info("No river records available for plotting.")
            else:
                plot_df = pd.DataFrame({
                    "pH": pd.to_numeric(river_df["pH"], errors="coerce").reset_index(drop=True),
                    "Dissolved Oxygen": pd.to_numeric(river_df["Dissolved Oxygen"], errors="coerce").reset_index(drop=True),
                }).dropna()
                if HAS_PLOTLY and not plot_df.empty:
                    plot_df["Sample"] = np.arange(1, len(plot_df) + 1)
                    fig = px.line(plot_df, x="Sample", y=["pH", "Dissolved Oxygen"], markers=True,
                                  title=f"pH and Dissolved Oxygen Trend - {river}")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.line_chart(plot_df)
        except Exception as exc:
            st.error(f"Time-series chart failed: {exc}")

        st.divider()
        st.markdown("### Correlation Heatmap")
        try:
            corr_cols = ["pH", "Dissolved Oxygen", "Turbidity"]
            corr_df = river_df.copy()
            for col in corr_cols:
                corr_df[col] = pd.to_numeric(corr_df[col], errors="coerce")
            corr = corr_df[corr_cols].corr()
            st.dataframe(corr.round(3), use_container_width=True)
            if HAS_PLOTLY:
                fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu", zmin=-1, zmax=1)
                fig.update_layout(title="Correlation Heatmap")
                st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:
            st.error(f"Correlation heatmap failed: {exc}")
    except Exception as exc:
        st.error(f"Preprocessing & visualization tab failed: {exc}")
def _load_ml_artifacts():
    """Try loading trained artifacts and expose a safe fallback."""
    artifacts = {}
    try:
        # Try a few common names/locations used in the project.
        candidate_paths = [
            os.path.join(config.BASE_DIR, "data", "processed", "water_quality_classifier.pkl"),
            os.path.join(config.BASE_DIR, "data", "processed", "best_model.pkl"),
            os.path.join(config.BASE_DIR, "data", "processed", "best_ph_model.pkl"),
        ]
        existing = [p for p in candidate_paths if os.path.exists(p)]
        if not existing:
            return None, None, "Model file not found."

        model_path = existing[0]
        artifacts["model"] = joblib.load(model_path)
        # optional scaler
        scaler_candidates = [
            model_path.replace(".pkl", "_scaler.pkl"),
            os.path.join(config.BASE_DIR, "data", "processed", "scaler.pkl"),
        ]
        for sp in scaler_candidates:
            if os.path.exists(sp):
                try:
                    artifacts["scaler"] = joblib.load(sp)
                    break
                except Exception:
                    pass

        return artifacts.get("model"), artifacts.get("scaler"), None
    except Exception as exc:
        return None, None, str(exc)


def _predict_with_artifacts(model, scaler, features):
    """Predict using the loaded artifacts, with safe fallback to project inference code.

    `features` must already be ordered to match the model's training feature
    order (see ml_model.CLASSIFICATION_FEATURES) - scalers/estimators fitted
    on plain numpy arrays have no column-name matching to fall back on, so a
    wrong column order silently produces a wrong (but not-erroring) result.
    """
    X = pd.DataFrame([features])
    try:
        if scaler is not None:
            try:
                X_scaled = scaler.transform(X)
                pred = model.predict(X_scaled)[0]
                proba = None
                if hasattr(model, "predict_proba"):
                    proba = float(np.max(model.predict_proba(X_scaled)[0]))
                return pred, proba
            except Exception:
                pass

        pred = model.predict(X)[0]
        proba = None
        if hasattr(model, "predict_proba"):
            proba = float(np.max(model.predict_proba(X)[0]))
        return pred, proba
    except Exception:
        return None, None


def render_ml_tab():
    st.subheader("Machine Learning & Quality Predictor")
    st.caption("Use the trained model to predict water potability / contamination risk.")

    # Keep the interface aligned to your CIPAT Step 4 requirement.
    c1, c2 = st.columns(2)
    ph = c1.slider("pH", 0.0, 14.0, 7.2, 0.1, key="ml_ph")
    do = c2.slider("Dissolved Oxygen (mg/L)", 0.0, 20.0, 7.5, 0.1, key="ml_do")

    c3, c4 = st.columns(2)
    turb = c3.slider("Turbidity (NTU)", 0.0, 100.0, 4.0, 0.1, key="ml_turb")
    bod = c4.slider("BOD (mg/L)", 0.0, 100.0, 3.0, 0.1, key="ml_bod")

    c5, c6 = st.columns(2)
    temp = c5.number_input("Water Temperature (°C)", 0.0, 60.0, 28.0, 0.1, key="ml_temp")
    tds = c6.number_input("TDS (mg/L)", 0.0, 5000.0, 420.0, 10.0, key="ml_tds")

    c7, c8 = st.columns(2)
    level = c7.number_input("Water Level (m)", 0.0, 50.0, 4.2, 0.1, key="ml_level")
    hour = c8.number_input("Hour of Day", 0, 23, 12, 1, key="ml_hour")

    # Column order MUST match modules.ml_model.CLASSIFICATION_FEATURES, since
    # the saved StandardScaler/model were fit on plain numpy arrays with no
    # column-name checking - a mismatched order gives a silently wrong result.
    features = {col: {
        "ph": ph, "temperature_c": temp, "dissolved_oxygen_mgl": do,
        "conductivity_us_cm": tds, "turbidity_ntu": turb, "hour": hour,
    }[col] for col in ml_model.CLASSIFICATION_FEATURES}

    if st.button("Predict Water Potability"):
        try:
            model, scaler, load_err = _load_ml_artifacts()
            if model is None:
                st.warning(
                    "ML model file was not found. "
                    "Please run the full pipeline (main.py --full) or option 7 "
                    "so a trained .pkl model exists in data/processed."
                )
                with st.expander("Model Insights"):
                    st.write("Expected artifact: water_quality_classifier.pkl")
                    st.write(f"Load status: {load_err}")
                    st.write("Current input features:")
                    st.json(features)
                return

            pred, proba = _predict_with_artifacts(model, scaler, features)

            if pred is None:
                st.warning(
                    "Prediction could not be completed with the loaded model artifact - "
                    "the input could not be scored. Try retraining via main.py --full."
                )
                return

            # The classifier's target is `contamination_event`: 1 = contamination
            # detected (unsafe), 0 = normal reading (safe/potable).
            is_contaminated = int(pred) == 1

            if is_contaminated:
                st.error("Contaminated / Unsafe")
            else:
                st.success("Safe for Consumption / Potable")

            if proba is not None:
                st.metric("Model Confidence", f"{proba:.2%}")

            with st.expander("Model Insights"):
                st.write("Loaded model type:", type(model).__name__)
                st.write("Input features used for prediction:")
                st.json(features)
                if hasattr(model, "feature_importances_"):
                    fi = pd.DataFrame({
                        "Feature": ["ph", "temperature_c", "dissolved_oxygen_mgl", "conductivity_us_cm", "turbidity_ntu", "hour"],
                        "Importance": model.feature_importances_[:6] if len(model.feature_importances_) >= 6 else model.feature_importances_,
                    })
                    st.dataframe(fi, use_container_width=True, hide_index=True)
                elif hasattr(model, "coef_"):
                    st.write("Model coefficients are available for inspection.")
                else:
                    st.write("Feature importance is not exposed by this model type.")

        except Exception as exc:
            st.warning(f"Prediction could not be completed: {exc}")



def _render_top_nav_banner():
    st.markdown(
        """
        <div style="padding: 0.65rem 0 0.15rem 0; margin-bottom: 0.55rem;">
            <div style="display:flex; flex-wrap: wrap; gap: 0.55rem; align-items:center;">
                <span style="padding:0.35rem 0.8rem; border-radius:999px; background:#eef2ff; color:#1e3a8a; border:1px solid #c7d2fe; font-weight:600;">Indian Water Quality Dashboard</span>
                <span style="padding:0.35rem 0.8rem; border-radius:999px; background:#f8fafc; color:#334155; border:1px solid #e2e8f0; font-weight:600;">Data Management & CRUD</span>
                <span style="padding:0.35rem 0.8rem; border-radius:999px; background:#f8fafc; color:#334155; border:1px solid #e2e8f0; font-weight:600;">Statistical Analysis & Hypothesis Testing</span>
                <span style="padding:0.35rem 0.8rem; border-radius:999px; background:#f8fafc; color:#334155; border:1px solid #e2e8f0; font-weight:600;">Preprocessing & Visualization</span>
                <span style="padding:0.35rem 0.8rem; border-radius:999px; background:#f8fafc; color:#334155; border:1px solid #e2e8f0; font-weight:600;">Machine Learning & Quality Predictor</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def main():
    _render_top_nav_banner()
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Indian Water Quality Dashboard",
        "Data Management & CRUD",
        "Statistical Analysis & Hypothesis Testing",
        "Preprocessing & Visualization",
        "Machine Learning & Quality Predictor",
    ])
    with tab1:
        render_dashboard()
    with tab2:
        render_crud_tab()
    with tab3:
        render_stats_tab()
    with tab4:
        render_preprocessing_tab()
    with tab5:
        render_ml_tab()


if __name__ == "__main__":
    main()

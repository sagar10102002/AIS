# streamlit_ais_app.py
# Professional Streamlit UI for AIS dataset + vessel classifier & speed predictor

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import io
import base64
import plotly.express as px
import pydeck as pdk
import streamlit.components.v1 as components
from datetime import datetime

# ----------------------- Helpers / Caching ---------------------------------

@st.cache_data(show_spinner=False)
def load_csv(path):
    return pd.read_csv(path)

@st.cache_resource(show_spinner=False)
def load_model(source):
    """
    Load a model from a file path, file-like object, or raw bytes.
    Supports joblib and pickle formats.
    """
    import pickle

    try:
        if isinstance(source, (str, os.PathLike)) and os.path.exists(source):
            return joblib.load(source)  # File path
        elif hasattr(source, "read"):
            model_bytes = source.read()  # Uploaded file-like object
            return joblib.load(io.BytesIO(model_bytes))
        elif isinstance(source, (bytes, bytearray)):
            return joblib.load(io.BytesIO(source))  # Raw bytes
        else:
            raise ValueError("Unsupported model source type.")
    except Exception as e:
        # Try pickle as fallback
        try:
            if hasattr(source, "read"):
                model_bytes = source.read()
                return pickle.loads(model_bytes)
            elif isinstance(source, (bytes, bytearray)):
                return pickle.loads(source)
            elif isinstance(source, (str, os.PathLike)) and os.path.exists(source):
                with open(source, "rb") as f:
                    return pickle.load(f)
            else:
                raise e
        except Exception as e2:
            raise RuntimeError(f"Model loading failed: {e2}")

def to_csv_download(df, name="export.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f"data:file/csv;base64,{b64}"
    return href

# ----------------------- App UI & Layout ----------------------------------

st.set_page_config(page_title="AIS Explorer — Tech UI", layout="wide", page_icon="🚢")

# Custom minimal 'tech' CSS
st.markdown(
    """
    <style>
    .stApp { background: (#ffffff); color: #cbd5e1; }
    .sidebar .sidebar-content { background: #020617; }
    .card { background: rgba(255,255,255,0.02); padding:12px; border-radius:12px; box-shadow: 0 6px 18px rgba(2,6,23,0.6); }
    .small-muted { color:#94a3b8; font-size:12px }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
col1, col2 = st.columns([0.7, 0.3])
with col1:
    st.markdown("# 🚢 AIS Explorer")
    st.markdown("<div class='small-muted'><h4>Interactive dashboard to explore AIS data, run vessel-type and speed predictions, visualize tracks and export results.</h4></div>", unsafe_allow_html=True)
with col2:
    st.image("logo.png", width=200)

st.markdown("---")

# Sidebar: file loading + options
with st.sidebar:
    st.header("Data & Models")
    st.markdown("Upload CSV or keep defaults (server files).")

    uploaded_csv = st.file_uploader("Upload AIS CSV", type=["csv"], key="ais_csv")
    default_csv_paths = [
        "/mnt/data/AIS_100000_data.csv",
        "/mnt/data/processed_ais.csv",
        "/mnt/data/ais.csv",
    ]

    csv_path = None
    if uploaded_csv is not None:
        csv_path = uploaded_csv
    else:
        for p in default_csv_paths:
            if os.path.exists(p):
                csv_path = p
                break

    st.markdown("---")
    st.markdown("**Models**")
    uploaded_vessel_model = st.file_uploader("Upload vessel_classifier.pkl", type=["pkl","joblib"], key="vessel_model")
    uploaded_speed_model = st.file_uploader("Upload speed_predictor.pkl", type=["pkl","joblib"], key="speed_model")

    # try default paths
    default_vessel_path = "/mnt/data/vessel_classifier.pkl"
    default_speed_path = "/mnt/data/speed_predictor.pkl"

    vessel_model_path = uploaded_vessel_model if uploaded_vessel_model is not None else (default_vessel_path if os.path.exists(default_vessel_path) else None)
    speed_model_path = uploaded_speed_model if uploaded_speed_model is not None else (default_speed_path if os.path.exists(default_speed_path) else None)

    st.markdown("---")
    st.markdown("**Map file (his_map.html)**")
    uploaded_map = st.file_uploader("Upload his_map.html", type=["html"], key="his_map")
    default_map_path = "/mnt/data/his_map.html"
    map_path = uploaded_map if uploaded_map is not None else (default_map_path if os.path.exists(default_map_path) else None)

    st.markdown("---")
    st.caption("Tip: if model files are missing, upload via the file uploaders above.")

# Load data (main area)
if csv_path is None:
    st.error("No AIS CSV found. Please upload a file in the sidebar or place `AIS_100000_data.csv` in the working directory.")
    st.stop()

try:
    if isinstance(csv_path, str):
        df = load_csv(csv_path)
    else:
        df = pd.read_csv(csv_path)  # uploaded file-like object
except Exception as e:
    st.error(f"Failed to load CSV: {e}")
    st.stop()

st.sidebar.markdown(f"**Rows:** {len(df):,}  —  **Columns:** {df.shape[1]}")

# Ensure latitude & longitude exist
if not any(col.lower() in ["lat","latitude"] for col in df.columns) or not any(col.lower() in ["lon","lng","longitude"] for col in df.columns):
    st.warning("CSV doesn't contain clear latitude/longitude columns. UI will still show data table. Rename columns to 'lat' and 'lon' or 'latitude'/'longitude' for mapping features.")

# KPIs
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("Total AIS points", f"{len(df):,}")
with k2:
    unique_mmsi = df['MMSI'].nunique() if 'MMSI' in df.columns else df.iloc[:,0].nunique()
    st.metric("Unique vessels", f"{unique_mmsi:,}")
with k3:
    st.metric("Columns", f"{df.shape[1]}")
with k4:
    st.metric("Loaded on", datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC'))

st.markdown("---")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Explore Data", "Map & Tracks", "Predict & Export", "Embed History Map"])

with tab1:
    st.subheader("Explore Data")
    c1, c2 = st.columns([2,1])
    with c1:
        st.dataframe(df)
    with c2:
        st.markdown("#### Quick Plots")
        numeric = df.select_dtypes(include=[np.number])
        if numeric.shape[1] >= 1:
            col_plot = st.selectbox("Numeric column", options=numeric.columns)
            fig = px.histogram(df, x=col_plot, nbins=40, marginal="box", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numeric columns detected for quick plots.")

with tab2:
    st.subheader("Map & Tracks")
    lat_cols = [c for c in df.columns if c.lower() in ('lat','latitude')]
    lon_cols = [c for c in df.columns if c.lower() in ('lon','lng','longitude')]
    if lat_cols and lon_cols:
        lat_col = lat_cols[0]
        lon_col = lon_cols[0]
        st.markdown("**Interactive deck.gl map**")
        midpoint = (float(df[lat_col].median()), float(df[lon_col].median()))
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=df.dropna(subset=[lat_col, lon_col]),
            get_position=[lon_col, lat_col],
            get_radius=100,
            pickable=True,
            opacity=0.6,
        )
        view_state = pdk.ViewState(latitude=midpoint[0], longitude=midpoint[1], zoom=3, pitch=0)
        r = pdk.Deck(layers=[layer], initial_view_state=view_state, map_style='mapbox://styles/mapbox/dark-v10')
        st.pydeck_chart(r)

        if 'MMSI' in df.columns:
            mmsi = st.selectbox("MMSI", options=sorted(df['MMSI'].unique().tolist()), index=0)
            vessel_df = df[df['MMSI'] == mmsi].sort_values(by=df.columns[0])
            if len(vessel_df) >= 2:
                st.markdown(f"Recent points for MMSI **{mmsi}** — {len(vessel_df)} points")
                st.map(vessel_df[[lat_col, lon_col]].dropna())
            else:
                st.info("Not enough points for a track preview.")
        else:
            st.info("Column 'MMSI' not found in dataset — map shows raw points.")
    else:
        st.warning("Latitude/Longitude columns not detected. Please upload a CSV with correct naming.")

with tab3:
    st.subheader("Predict & Export")
    colA, colB = st.columns(2)

    with colA:
        st.markdown("**Vessel Type Model**")
        if vessel_model_path is not None:
            try:
                vessel_model = load_model(vessel_model_path)
                st.success("Vessel model loaded")
                run_vessel = st.button("Run vessel-type prediction")
            except Exception as e:
                st.error(f"Failed to load vessel model: {e}")
                vessel_model = None
                run_vessel = False
        else:
            st.info("No vessel model found — upload one in the sidebar.")
            vessel_model = None
            run_vessel = False

    with colB:
        st.markdown("**Speed Model**")
        if speed_model_path is not None:
            try:
                speed_model = load_model(speed_model_path)
                st.success("Speed model loaded")
                run_speed = st.button("Run speed prediction")
            except Exception as e:
                st.error(f"Failed to load speed model: {e}")
                speed_model = None
                run_speed = False
        else:
            st.info("No speed model found — upload one in the sidebar.")
            speed_model = None
            run_speed = False

    if run_vessel and vessel_model is not None:
        st.info("Running vessel-type predictions...")
        try:
            features = df.select_dtypes(include=[np.number]).fillna(0)
            preds = vessel_model.predict(features)
            df['pred_vessel_type'] = preds
            st.success("Vessel-type predictions added.")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    if run_speed and speed_model is not None:
        st.info("Running speed predictions...")
        try:
            features = df.select_dtypes(include=[np.number]).fillna(0)
            preds = speed_model.predict(features)
            df['pred_speed'] = preds
            st.success("Speed predictions added.")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    st.markdown("**Preview & Export**")
    st.dataframe(df.head(200))
    csv_href = to_csv_download(df)
    st.markdown(f"[Download predictions as CSV]({csv_href})")

with tab4:
    st.subheader("Embedded History Map (his_map.html)")
    if map_path is not None:
        try:
            if hasattr(map_path, 'read'):
                html_bytes = map_path.read()
                html = html_bytes.decode('utf-8') if isinstance(html_bytes, bytes) else html_bytes
            else:
                with open(map_path, 'r', encoding='utf8') as f:
                    html = f.read()
            components.html(html, height=700, scrolling=True)
        except Exception as e:
            st.error(f"Failed to embed his_map.html: {e}")
    else:
        st.info("No his_map.html found.")

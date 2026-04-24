import base64
import json
import math
import os
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

# =========================
# CONFIG
# =========================
LOCAL_JSON = Path(__file__).parent / "latest_telemetry.json"
MAX_HISTORIAL = 300
REFRESH_SECONDS = 1

SMOOTH_WINDOW = 5
ASCENS_CONFIRM_POINTS = 4
ASCENS_THRESHOLD = 0.8
ASCENS_GAIN_MIN = 3.0

FASE_WINDOW = 10          # finestra d'avaluació de velocitat (punts)
FASE_V_UP = 0.8           # llindar velocitat ascens (m/s)
FASE_V_DOWN = -0.8        # llindar velocitat descens (m/s)
FASE_LAND_V_ABS = 0.25    # velocitat màxima per considerar aterrat (m/s)
FASE_LAND_ALTURA_MAX = 5.0

FASE_CONFIRM_PUNTS = 3    # confirmacions per a transicions normals
FASE_CONFIRM_ATERRAT = 6  # confirmacions per declarar aterratge (molt conservador)
FASE_GAP_SEGONS = 8       # segons sense dades noves → mantén última fase coneguda

MAP_HEIGHT = 650
MAP_ZOOM = 18
MAP_MOVE_THRESHOLD_METERS = 15.0
MAP_FORCE_REFRESH_SECONDS = 30

TAULA_COLUMNS = [
    "temps_txt",
    "temps",
    "lat",
    "lon",
    "alt",
    "alt_press",
    "alt_suav",
    "altura_guanyada",
    "altura_maxima_total",
    "vel",
    "vel_calc",
    "vel_lineal_calc",
    "temp",
    "press",
    "retard_s",
    "camX",
    "camY",
    "pc_rebut_ts",
]

PLOTLY_CONFIG = {
    "displayModeBar": False,
    "scrollZoom": False,
    "responsive": True,
}

# =========================
# UI BASE
# =========================
st.set_page_config(page_title="Estació de terra", layout="wide")

ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"
SATPI_LOGO = ASSETS_DIR / "satpi_logo.png"
INSTITUT_LOGO = ASSETS_DIR / "institut_logo.png"


def imatge_a_base64(path: Path):
    try:
        return base64.b64encode(path.read_bytes()).decode("utf-8")
    except Exception:
        return None

st.markdown(
    """
    <style>
    :root {
        --bg-main: #040b18;
        --bg-soft: #091529;
        --card-bg: rgba(10, 23, 43, 0.82);
        --card-border: rgba(119, 170, 255, 0.16);
        --text-main: #f8fbff;
        --text-soft: #aac0dc;
        --shadow: 0 12px 32px rgba(0,0,0,0.28);
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(40,85,160,0.18), transparent 32%),
            radial-gradient(circle at top right, rgba(0,170,220,0.10), transparent 28%),
            linear-gradient(180deg, #030814 0%, #07111f 55%, #040a14 100%);
    }

    .block-container {
        padding-top: 0.8rem;
        padding-bottom: 1.3rem;
        max-width: none;
    }

    /* HEADER */
    .header-shell {
        margin-bottom: 1.1rem;
    }

    .header-logo-box {
        height: 122px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 22px;
        background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.015));
        border: 1px solid rgba(255,255,255,0.05);
        box-shadow: var(--shadow);
        backdrop-filter: blur(10px);
    }

    .header-logo-img {
        display: block;
        width: auto;
        height: auto;
        object-fit: contain;
        filter: drop-shadow(0 3px 10px rgba(0,0,0,0.22));
    }

    .institut-logo {
        max-width: 220px;
        max-height: 76px;
    }

    .satpi-logo {
        max-width: 92px;
        max-height: 92px;
    }

    .top-header {
        min-height: 122px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        padding: 20px 30px;
        border-radius: 24px;
        background:
            linear-gradient(135deg, rgba(14,29,52,0.96), rgba(4,29,54,0.96)),
            linear-gradient(90deg, rgba(86,182,255,0.10), rgba(124,227,200,0.06));
        border: 1px solid rgba(130,180,255,0.14);
        box-shadow: var(--shadow);
        position: relative;
        overflow: hidden;
    }

    .top-header::before {
        content: "";
        position: absolute;
        inset: 0;
        background:
            radial-gradient(circle at 18% 50%, rgba(86,182,255,0.12), transparent 26%),
            radial-gradient(circle at 82% 35%, rgba(124,227,200,0.08), transparent 22%);
        pointer-events: none;
    }

    .top-header-title {
        position: relative;
        font-size: 2.55rem;
        font-weight: 800;
        line-height: 1.02;
        letter-spacing: -0.03em;
        color: var(--text-main);
        margin-bottom: 10px;
    }

    .top-header-subtitle {
        position: relative;
        font-size: 1.02rem;
        color: var(--text-soft);
        letter-spacing: 0.01em;
    }

    .info-card {
        background: #0f1724;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 18px 20px;
        margin-bottom: 14px;
        color: #f8fafc;
    }

    .info-card h3 {
        margin-top: 0;
        margin-bottom: 14px;
        font-size: 1.35rem;
        color: #ffffff;
    }

    .info-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 8px 20px;
    }

    .info-item {
        padding: 8px 0;
        border-bottom: 1px solid rgba(255,255,255,0.05);
        font-size: 1rem;
        line-height: 1.35;
        color: #e5e7eb;
    }

    .info-item b {
        color: #ffffff;
        font-weight: 700;
    }

    .map-wrap {
        background: #0f1724;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 10px;
    }

    @media (max-width: 1100px) {
        .top-header-title { font-size: 2rem; }
        .top-header-subtitle { font-size: 0.95rem; }
        .institut-logo { max-width: 180px; max-height: 62px; }
        .satpi-logo { max-width: 82px; max-height: 82px; }
    }

    @media (max-width: 900px) {
        .top-header-title { font-size: 1.55rem; }
        .header-logo-box, .top-header { min-height: 100px; height: 100px; }
    }

    /*  FASE CARD  */
    .fase-card {
        border-radius: 20px;
        padding: 26px 28px;
        margin-bottom: 0;
        display: flex;
        flex-direction: column;
        gap: 10px;
        min-height: 220px;
        position: relative;
        overflow: hidden;
    }
    .fase-card::before {
        content: "";
        position: absolute;
        inset: 0;
        pointer-events: none;
        border-radius: 20px;
    }
    .fase-icon { font-size: 2.8rem; line-height: 1; }
    .fase-nom {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        line-height: 1.05;
    }
    .fase-desc {
        font-size: 1.05rem;
        opacity: 0.82;
        line-height: 1.5;
        margin-top: 4px;
    }
    .fase-retard {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        font-size: 0.88rem;
        font-weight: 600;
        border-radius: 999px;
        padding: 5px 14px;
        margin-top: 6px;
        align-self: flex-start;
        letter-spacing: 0.02em;
    }
    .retard-dot {
        width: 8px; height: 8px;
        border-radius: 50%;
        display: inline-block;
        flex-shrink: 0;
    }

    /*  MOVIMENT CARD  */
    .mov-card {
        background: rgba(10,23,43,0.82);
        border: 1.5px solid rgba(56,189,248,0.18);
        border-radius: 20px;
        padding: 22px 24px;
        display: flex;
        flex-direction: column;
        gap: 14px;
        min-height: 220px;
    }
    .mov-titol {
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.13em;
        color: #64748b;
        text-transform: uppercase;
        margin-bottom: 2px;
    }
    .mov-principal {
        font-size: 1.55rem;
        font-weight: 800;
        color: #f8fbff;
        line-height: 1.2;
    }
    .mov-principal span.accent { color: #38bdf8; }
    .mov-principal span.up     { color: #34d399; }
    .mov-principal span.down   { color: #fb923c; }
    .mov-principal span.stable { color: #94a3b8; }
    .mov-fila {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 10px 14px;
        border-radius: 12px;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.06);
    }
    .mov-fila-icon { font-size: 1.35rem; flex-shrink: 0; }
    .mov-fila-text { font-size: 0.97rem; color: #cbd5e1; line-height: 1.35; }
    .mov-fila-text b { color: #f1f5f9; font-weight: 700; }
    .mov-fila-vel {
        margin-left: auto;
        font-size: 1.1rem;
        font-weight: 800;
        color: #f8fbff;
        flex-shrink: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def renderitzar_header():
    st.markdown('<div class="header-shell">', unsafe_allow_html=True)

    col_left, col_center, col_right = st.columns([1.25, 4.9, 1.25], gap="medium")

    with col_left:
        b64 = imatge_a_base64(INSTITUT_LOGO) if INSTITUT_LOGO.exists() else None
        if b64:
            st.markdown(
                f"""
                <div class="header-logo-box">
                    <img src="data:image/png;base64,{b64}"
                         class="header-logo-img institut-logo">
                </div>
                """,
                unsafe_allow_html=True,
            )

    with col_center:
        st.markdown(
            """
            <div class="top-header">
                <div class="top-header-title">Estació de terra SATPI26</div>
                <div class="top-header-subtitle">
                    Institut Bernat el Ferrer · CanSat · Telemetria en temps real
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_right:
        b64 = imatge_a_base64(SATPI_LOGO) if SATPI_LOGO.exists() else None
        if b64:
            st.markdown(
                f"""
                <div class="header-logo-box">
                    <img src="data:image/png;base64,{b64}"
                         class="header-logo-img satpi-logo">
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)


renderitzar_header()

# =========================
# STATE
# =========================
def init_state():
    if "init" in st.session_state:
        return

    st.session_state.init = True
    st.session_state.historial = deque(maxlen=MAX_HISTORIAL)
    st.session_state.altura_base = None
    st.session_state.ha_descendit = False
    st.session_state.last_valid_gps = None
    st.session_state.last_json_temps = None
    st.session_state.last_json_pc_rebut_ts = None
    st.session_state.last_json_mtime = None

    st.session_state.map_html_cached = ""
    st.session_state.map_last_render_time = 0.0
    st.session_state.map_lat_render = None
    st.session_state.map_lon_render = None
    st.session_state.launch_temps = None
    # Màquina d'estats de fase
    st.session_state.fase_confirmada = "Esperant enlairament"
    st.session_state.fase_candidata = None
    st.session_state.fase_candidata_n = 0
    st.session_state.last_data_wall_time = 0.0


def reset_missio():
    st.session_state.historial = deque(maxlen=MAX_HISTORIAL)
    st.session_state.altura_base = None
    st.session_state.ha_descendit = False
    st.session_state.last_valid_gps = None
    st.session_state.last_json_temps = None
    st.session_state.last_json_pc_rebut_ts = None
    st.session_state.last_json_mtime = None
    st.session_state.map_html_cached = ""
    st.session_state.map_last_render_time = 0.0
    st.session_state.map_lat_render = None
    st.session_state.map_lon_render = None
    st.session_state.launch_temps = None
    # Màquina d'estats de fase
    st.session_state.fase_confirmada = "Esperant enlairament"
    st.session_state.fase_candidata = None
    st.session_state.fase_candidata_n = 0
    st.session_state.last_data_wall_time = 0.0


init_state()

# =========================
# TEMPS / RETARD
# =========================
def calcular_retard_segons(pc_rebut_ts):
    try:
        pc_rebut_ts = float(pc_rebut_ts)
    except Exception:
        return None
    return max(0.0, time.time() - pc_rebut_ts)


# =========================
# LECTURA JSON LOCAL
# =========================
def llegir_json_local(path):
    path = Path(path)
    if not path.exists():
        return None

    try:
        # Lectura robusta (fitxer pot estar-se actualitzant)
        raw = None
        for _ in range(3):
            try:
                raw = path.read_text(encoding="utf-8")
                data = json.loads(raw)
                break
            except json.JSONDecodeError:
                time.sleep(0.03)
        else:
            return None

        if not data or "temps" not in data:
            return None

        return {
            "lat": float(data["lat"]),
            "lon": float(data["lon"]),
            "alt": float(data["alt"]),
            "vel": float(data["vel"]),
            "temp": float(data["temp"]),
            "press": float(data["press"]),
            "alt_press": float(data["alt_press"]),
            "temps_txt": str(data.get("temps_txt", "")),
            "temps": float(data["temps"]),
            "camX": str(data.get("camX", "center")),
            "camY": str(data.get("camY", "center")),
            "pc_rebut_ts": float(data.get("pc_rebut_ts")) if data.get("pc_rebut_ts") is not None else None,
        }

    except Exception:
        return None


def processar_lectura_json(path):
    path = Path(path)
    data = llegir_json_local(path)
    if data is None:
        return

    try:
        mtime = path.stat().st_mtime
    except Exception:
        mtime = None

    # Evita reprocessar exactament el mateix fitxer (però assegura temps real)
    if (
        st.session_state.last_json_mtime is not None
        and mtime is not None
        and mtime == st.session_state.last_json_mtime
    ):
        return

    if st.session_state.historial and data["temps"] < st.session_state.historial[-1]["temps"]:
        reset_missio()

    retard = calcular_retard_segons(data.get("pc_rebut_ts"))
    data["retard_s"] = float(retard) if retard is not None else 0.0

    st.session_state.last_json_temps = data["temps"]
    st.session_state.last_json_pc_rebut_ts = data.get("pc_rebut_ts")
    st.session_state.last_json_mtime = mtime
    st.session_state.last_data_wall_time = time.time()  # per a detecció de gaps

    if coords_valides(data["lat"], data["lon"]):
        st.session_state.last_valid_gps = {
            "lat": data["lat"],
            "lon": data["lon"],
            "temps": data["temps"],
        }

    st.session_state.historial.append(data)


# =========================
# GPS / DISTÀNCIA
# =========================
def coords_valides(lat, lon):
    try:
        lat = float(lat)
        lon = float(lon)
    except Exception:
        return False
    return np.isfinite(lat) and np.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180


def metres_per_grau(lat):
    return 111320.0, 111320.0 * math.cos(math.radians(lat))


def distancia_metres(lat1, lon1, lat2, lon2):
    if not (coords_valides(lat1, lon1) and coords_valides(lat2, lon2)):
        return 0.0

    m_lat = 111320.0
    m_lon = 111320.0 * math.cos(math.radians((lat1 + lat2) / 2.0))
    dx = (lon2 - lon1) * m_lon
    dy = (lat2 - lat1) * m_lat
    return float(math.hypot(dx, dy))


# =========================
# CÀLCULS
# =========================
def calcular_velocitat_lineal_df(df):
    if len(df) < 2:
        return pd.Series(0.0, index=df.index)

    dt = df["temps"].diff()
    valid_actual = df["lat"].between(-90, 90) & df["lon"].between(-180, 180)
    valid_anterior = valid_actual.shift(fill_value=False)
    valid = valid_actual & valid_anterior & (dt > 0)

    m_lat = 111320.0
    m_lon = 111320.0 * np.cos(np.radians(df["lat"]))
    dx = df["lon"].diff() * m_lon
    dy = df["lat"].diff() * m_lat

    vel = pd.Series(np.hypot(dx, dy) / dt, index=df.index)
    return vel.where(valid, 0.0).replace([np.inf, -np.inf], 0).fillna(0.0)


def afegir_variables_altura(df):
    df = df.copy()

    df["alt_suav"] = df["alt"].rolling(window=SMOOTH_WINDOW, min_periods=1).mean()
    df["vel_calc"] = df["alt_suav"].diff() / df["temps"].diff()
    df["vel_calc"] = df["vel_calc"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["vel_lineal_calc"] = calcular_velocitat_lineal_df(df)

    if st.session_state.altura_base is None and len(df) >= ASCENS_CONFIRM_POINTS + 1:
        ultimes_vels = df["vel_calc"].tail(ASCENS_CONFIRM_POINTS)
        guany_finestra = float(
            df["alt_suav"].iloc[-1] - df["alt_suav"].iloc[-ASCENS_CONFIRM_POINTS - 1]
        )

        if (ultimes_vels > ASCENS_THRESHOLD).all() and guany_finestra >= ASCENS_GAIN_MIN:
            idx_ref = max(0, len(df) - ASCENS_CONFIRM_POINTS - 1)
            st.session_state.altura_base = float(df.iloc[idx_ref]["alt_suav"])
            if st.session_state.launch_temps is None:
                st.session_state.launch_temps = float(df.iloc[idx_ref]["temps"])

    if st.session_state.altura_base is None:
        df["altura_guanyada"] = 0.0
    else:
        df["altura_guanyada"] = (df["alt_suav"] - st.session_state.altura_base).clip(lower=0)

    df["altura_maxima_total"] = df["alt"].cummax()
    return df, st.session_state.altura_base


def calcular_velocitat_vertical(df):
    """Velocitat vertical calculada sobre una finestra de punts (mediana).
    Molt més robusta a outliers i gaps puntuals que usar només 2 punts."""
    if len(df) < 2:
        return 0.0

    n = min(6, len(df))
    recent = df.tail(n)
    dt = recent["temps"].diff()
    v = (recent["alt_suav"].diff() / dt).replace([np.inf, -np.inf], np.nan).dropna()

    if len(v) == 0:
        return 0.0
    return float(v.median())


def calcular_temps_aprox_aterratge(df, altura_guanyada, fase):
    if fase != "Descens" or altura_guanyada <= 0 or len(df) < 5:
        return None

    vels = df["vel_calc"].tail(8)
    vels_negatives = vels[vels < -0.25]

    if len(vels_negatives) < 2:
        return None

    velocitat_descens = abs(float(vels_negatives.mean()))
    if velocitat_descens <= 0:
        return None

    return altura_guanyada / velocitat_descens


def format_temps_aprox(segons):
    if segons is None:
        return "-"

    total = max(0, int(round(segons)))
    minuts, segons_restants = divmod(total, 60)
    hores, minuts_restants = divmod(minuts, 60)

    if hores > 0:
        return f"{hores}h {minuts_restants:02d}m"
    if minuts > 0:
        return f"{minuts}m {segons_restants:02d}s"
    return f"{segons_restants}s"


def obtenir_fase_intelligent(df):
    if len(df) < 2:
        return "Esperant enlairament"

    # Velocitat vertical suavitzada (robusta a soroll i a dt variable)
    dt = df["temps"].diff()
    v_alt = df["alt_suav"].diff() / dt
    v_alt = v_alt.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    w = min(FASE_WINDOW, len(df))
    v_mean = float(v_alt.tail(w).mean())
    v_abs_mean = float(v_alt.tail(w).abs().mean())

    altura_guanyada = float(df.iloc[-1].get("altura_guanyada", 0.0))
    vel_lineal = float(df.iloc[-1].get("vel_lineal_calc", 0.0))

    # 1) Encara no tenim un enlairament clar
    if st.session_state.altura_base is None:
        if altura_guanyada >= ASCENS_GAIN_MIN and v_mean >= (FASE_V_UP * 0.6):
            return "Ascens"
        return "Esperant enlairament"

    # 2) Ascens / Descens (amb histeresi)
    if v_mean >= FASE_V_UP:
        return "Ascens"

    if v_mean <= FASE_V_DOWN:
        st.session_state.ha_descendit = True
        return "Descens"

    # 3) Després de descens, criteri d'aterratge més estricte
    if st.session_state.ha_descendit:
        # gairebé sense moviment vertical i sense moviment lineal
        if altura_guanyada <= FASE_LAND_ALTURA_MAX and v_abs_mean <= FASE_LAND_V_ABS and vel_lineal <= 0.5:
            return "Aterrat"
        return "Vol actiu"

    # 4) En vol però sense pujar/baixar clarament
    return "Vol actiu"


def moviment_estable():
    return {
        "mov_x": "X estable",
        "mov_y": "Y estable",
        "mov_z": "Altitud estable",
        "vel_lineal": 0.0,
        "direccio": "sense moviment",
    }


def calcular_moviment_i_velocitat_lineal(df):
    if len(df) < 2:
        return moviment_estable()

    a = df.iloc[-1]
    b = df.iloc[-2]
    dt = a["temps"] - b["temps"]

    if dt <= 0:
        return moviment_estable()

    delta_lon = a["lon"] - b["lon"]
    delta_lat = a["lat"] - b["lat"]
    delta_alt = a["alt_suav"] - b["alt_suav"]

    th_gps = 0.00001
    th_alt = 0.3

    mov_x = "X: cap a l'est" if delta_lon > th_gps else "X: cap a l'oest" if delta_lon < -th_gps else "X estable"
    mov_y = "Y: cap al nord" if delta_lat > th_gps else "Y: cap al sud" if delta_lat < -th_gps else "Y estable"
    mov_z = "Z: pujant" if delta_alt > th_alt else "Z: baixant" if delta_alt < -th_alt else "Altitud estable"

    if coords_valides(a["lat"], a["lon"]) and coords_valides(b["lat"], b["lon"]):
        m_lat, m_lon = metres_per_grau(a["lat"])
        dx_m = delta_lon * m_lon
        dy_m = delta_lat * m_lat
        vel_lineal = float(a["vel_lineal_calc"])
    else:
        dx_m = dy_m = vel_lineal = 0.0

    comp = []
    if dy_m > 0.3:
        comp.append("nord")
    elif dy_m < -0.3:
        comp.append("sud")

    if dx_m > 0.3:
        comp.append("est")
    elif dx_m < -0.3:
        comp.append("oest")

    return {
        "mov_x": mov_x,
        "mov_y": mov_y,
        "mov_z": mov_z,
        "vel_lineal": vel_lineal,
        "direccio": "-".join(comp) if comp else "sense moviment",
    }


# =========================
# GRÀFIQUES
# =========================
def mini_grafic(df, y, title):
    fig = px.line(df, x="temps", y=y, title=title)
    fig.update_layout(
        height=260,
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
        transition={"duration": 0},
        uirevision=f"graf-{y}",
    )
    return fig


# =========================
# MAPA
# =========================
def generar_html_mapa_leaflet(lat, lon, zoom=18, height=650):
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8"/>
        <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
        <style>
            html, body {{
                margin: 0;
                padding: 0;
                background: transparent;
            }}

            #map {{
                width: 100%;
                height: {height}px;
                border-radius: 14px;
                overflow: hidden;
                background: #d9d9d9;
            }}

            .pulse-marker {{
                width: 16px;
                height: 16px;
                background: #ff3b30;
                border: 3px solid rgba(255,255,255,0.95);
                border-radius: 50%;
            }}
        </style>
    </head>
    <body>
        <div id="map"></div>

        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        <script>
            const lat = {lat};
            const lon = {lon};
            const zoom = {zoom};

            const map = L.map("map", {{
                zoomControl: true,
                attributionControl: true,
                preferCanvas: true,
                zoomAnimation: false,
                fadeAnimation: false,
                markerZoomAnimation: false
            }}).setView([lat, lon], zoom);

            L.tileLayer("https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png", {{
                maxZoom: 19,
                maxNativeZoom: 19,
                detectRetina: false,
                attribution: "&copy; OpenStreetMap contributors"
            }}).addTo(map);

            const redIcon = L.divIcon({{
                className: "",
                html: '<div class="pulse-marker"></div>',
                iconSize: [22, 22],
                iconAnchor: [11, 11]
            }});

            L.marker([lat, lon], {{ icon: redIcon }}).addTo(map);

            L.circle([lat, lon], {{
                color: "#ff3b30",
                weight: 2,
                fillColor: "#ff3b30",
                fillOpacity: 0.10,
                radius: 8
            }}).addTo(map);

            setTimeout(() => {{
                map.invalidateSize();
            }}, 150);
        </script>
    </body>
    </html>
    """


def renderitzar_mapa():
    gps = st.session_state.last_valid_gps

    if gps is None:
        st.warning("No hi ha coordenades GPS vàlides per mostrar el mapa.")
        return

    lat = float(gps["lat"])
    lon = float(gps["lon"])

    now = time.time()

    if st.session_state.map_lat_render is None or st.session_state.map_lon_render is None:
        dist_m = 999999.0
    else:
        dist_m = distancia_metres(
            st.session_state.map_lat_render,
            st.session_state.map_lon_render,
            lat,
            lon,
        )

    elapsed = now - st.session_state.map_last_render_time

    cal_actualitzar = (
        st.session_state.map_html_cached == ""
        or dist_m >= MAP_MOVE_THRESHOLD_METERS
        or elapsed >= MAP_FORCE_REFRESH_SECONDS
    )

    if cal_actualitzar:
        st.session_state.map_lat_render = lat
        st.session_state.map_lon_render = lon
        st.session_state.map_last_render_time = now
        st.session_state.map_html_cached = generar_html_mapa_leaflet(
            lat=lat,
            lon=lon,
            zoom=MAP_ZOOM,
            height=MAP_HEIGHT,
        )

    st.markdown('<div class="map-wrap">', unsafe_allow_html=True)
    components.html(st.session_state.map_html_cached, height=MAP_HEIGHT, scrolling=False)
    st.markdown("</div>", unsafe_allow_html=True)


# =========================
# HTML CARDS: FASE + MOVIMENT
# =========================
def _html_card_fase(fase: str, retard_s: float) -> str:
    _FASES = {
        "Esperant enlairament": {
            "icon": "",
            "bg":     "linear-gradient(135deg, #0f172a 0%, #1e293b 100%)",
            "border": "rgba(100,116,139,0.55)",
            "glow":   "rgba(100,116,139,0.15)",
            "color":  "#94a3b8",
            "desc":   "Sistemes actius. Esperant el moment de l'enlairament.",
        },
        "Ascens": {
            "icon": "",
            "bg":     "linear-gradient(135deg, #052e16 0%, #064e3b 100%)",
            "border": "rgba(52,211,153,0.7)",
            "glow":   "rgba(52,211,153,0.18)",
            "color":  "#34d399",
            "desc":   "El cohet s'està enlairant. Altitud en augment constant.",
        },
        "Vol actiu": {
            "icon": "",
            "bg":     "linear-gradient(135deg, #0c1a35 0%, #0c2a4a 100%)",
            "border": "rgba(56,189,248,0.6)",
            "glow":   "rgba(56,189,248,0.15)",
            "color":  "#38bdf8",
            "desc":   "En vol. Monitoritzant posició, alçada i condicions.",
        },
        "Descens": {
            "icon": "🪂",
            "bg":     "linear-gradient(135deg, #2c0a00 0%, #431407 100%)",
            "border": "rgba(249,115,22,0.7)",
            "glow":   "rgba(249,115,22,0.18)",
            "color":  "#fb923c",
            "desc":   "Descens actiu. Calculant temps d'aterratge.",
        },
        "Aterrat": {
            "icon": "",
            "bg":     "linear-gradient(135deg, #0f172a 0%, #1c1917 100%)",
            "border": "rgba(168,162,158,0.55)",
            "glow":   "rgba(168,162,158,0.12)",
            "color":  "#a8a29e",
            "desc":   "Missió completada. Cohet recuperat a terra.",
        },
    }

    cfg = _FASES.get(fase, _FASES["Vol actiu"])

    if retard_s <= 3:
        r_dot_color = "#34d399"
        r_bg        = "rgba(52,211,153,0.12)"
        r_border    = "rgba(52,211,153,0.35)"
        r_text      = f"Temps real · retard {retard_s:.0f} s"
    elif retard_s <= 10:
        r_dot_color = "#fbbf24"
        r_bg        = "rgba(251,191,36,0.12)"
        r_border    = "rgba(251,191,36,0.35)"
        r_text      = f"Petit retard · {retard_s:.0f} s"
    else:
        r_dot_color = "#f87171"
        r_bg        = "rgba(248,113,113,0.12)"
        r_border    = "rgba(248,113,113,0.35)"
        r_text      = f"Retard important · {retard_s:.0f} s"

    return f"""
<div class="fase-card" style="
    background: {cfg['bg']};
    border: 1.5px solid {cfg['border']};
    box-shadow: 0 0 32px {cfg['glow']}, 0 8px 24px rgba(0,0,0,0.35);
    color: {cfg['color']};
">
    <div style="font-size:0.72rem;font-weight:700;letter-spacing:0.13em;opacity:0.7;text-transform:uppercase;margin-bottom:2px;">
        ETAPA DE LA MISSIÓ
    </div>
    <div class="fase-icon">{cfg['icon']}</div>
    <div class="fase-nom">{fase}</div>
    <div class="fase-desc">{cfg['desc']}</div>
    <div class="fase-retard" style="background:{r_bg};border:1px solid {r_border};color:{r_dot_color};">
        <span class="retard-dot" style="background:{r_dot_color};
            box-shadow:0 0 6px {r_dot_color};"></span>
        {r_text}
    </div>
</div>
"""


def _html_card_moviment(
    moviment: dict,
    vel_vertical: float,
    vel_lineal: float,
    direccio: str,
    temps_aterratge_txt: str,
    fase: str,
) -> str:
    #  Inline style constants 
    S_CARD  = (
        "background:rgba(10,23,43,0.82);""border:1.5px solid rgba(56,189,248,0.22);""border-radius:20px;""padding:22px 24px;""display:flex;""flex-direction:column;""gap:14px;""min-height:220px;""box-shadow:0 8px 24px rgba(0,0,0,0.3);"
    )
    S_LABEL = (
        "font-size:0.72rem;""font-weight:700;""letter-spacing:0.13em;""color:#64748b;""text-transform:uppercase;""margin-bottom:2px;"
    )
    S_PRINCIPAL = (
        "font-size:1.55rem;""font-weight:800;""color:#f8fbff;""line-height:1.2;"
    )
    S_FILA = (
        "display:flex;""align-items:center;""gap:12px;""padding:10px 14px;""border-radius:12px;""background:rgba(255,255,255,0.04);""border:1px solid rgba(255,255,255,0.07);"
    )
    S_ICON = "font-size:1.35rem;flex-shrink:0;"
    S_TEXT = "font-size:0.97rem;color:#cbd5e1;line-height:1.35;flex:1;"
    S_VEL  = "font-size:1.05rem;font-weight:800;color:#f1f5f9;flex-shrink:0;"

    #  Descripció vertical 
    if vel_vertical > 0.5:
        vert_icon = ""
        vert_color = "#34d399"
        vert_word = "PUJANT"
        vert_desc = "pujant"
    elif vel_vertical < -0.5:
        vert_icon = ""
        vert_color = "#fb923c"
        vert_word = "BAIXANT"
        vert_desc = "baixant"
    else:
        vert_icon = ""
        vert_color = "#94a3b8"
        vert_word = "ALTITUD ESTABLE"
        vert_desc = "estable"

    vert_principal = (
        f"El cohet està "
        f"<span style='color:{vert_color};font-weight:900;'>{vert_word}</span>"
        f" a <span style='color:#38bdf8;font-weight:900;'>{abs(vel_vertical):.1f} m/s</span>"
        if vert_word != "ALTITUD ESTABLE"
        else f"El cohet està <span style='color:#94a3b8;font-weight:900;'>ALTITUD ESTABLE</span>"
    )

    #  Descripció horitzontal 
    dir_clean = direccio.replace("-", "-").upper() if direccio != "sense moviment" else "SENSE MOVIMENT"
    if vel_lineal > 0.3:
        horit_icon = ""
        horit_label = f"Cap al <strong style='color:#f1f5f9;'>{dir_clean}</strong>"
    else:
        horit_icon = ""
        horit_label = "<strong style='color:#94a3b8;'>Sense desplaçament horitzontal</strong>"

    #  Velocitat total 3D 
    vel_3d = (vel_vertical ** 2 + vel_lineal ** 2) ** 0.5

    #  Fila aterratge 
    land_fila = ""
    if fase == "Descens" and temps_aterratge_txt != "-":
        land_fila = (
            f'<div style="{S_FILA}">'
            f'<span style="{S_ICON}"></span>'
            f'<span style="{S_TEXT}">Temps aprox. d\'aterratge</span>'
            f'<span style="{S_VEL}color:#fb923c;">{temps_aterratge_txt}</span>'
            f'</div>'
        )

    return (
        f'<div style="{S_CARD}">'

        f'<div style="{S_LABEL}">MOVIMENT</div>'

        f'<div style="{S_PRINCIPAL}">{vert_principal}</div>'

        f'<div style="{S_FILA}">'
        f'  <span style="{S_ICON}">{vert_icon}</span>'
        f'  <span style="{S_TEXT}"><strong style="color:#f1f5f9;">Velocitat vertical</strong> · {vert_desc}</span>'
        f'  <span style="{S_VEL}color:{vert_color};">{vel_vertical:+.2f} m/s</span>'
        f'</div>'

        f'<div style="{S_FILA}">'
        f'  <span style="{S_ICON}">{horit_icon}</span>'
        f'  <span style="{S_TEXT}">{horit_label}</span>'
        f'  <span style="{S_VEL}color:#38bdf8;">{vel_lineal:.2f} m/s</span>'
        f'</div>'

        f'<div style="{S_FILA}">'
        f'  <span style="{S_ICON}"></span>'
        f'  <span style="{S_TEXT}"><strong style="color:#f1f5f9;">Velocitat total</strong> (3D)</span>'
        f'  <span style="{S_VEL}color:#e2e8f0;">{vel_3d:.2f} m/s</span>'
        f'</div>'

        f'{land_fila}'

        f'</div>'
    )


# =========================
# RENDER
# =========================
def renderitzar_bloc_gps_i_mapa(
    dada,
    fase,
    altura_guanyada,
    altura_maxima_total,
    vel_lineal,
    direccio_lineal,
    temps_aterratge_txt,
    altura_base,
):
    st.subheader("Posició GPS en temps real")

    gps = st.session_state.last_valid_gps
    col_info, col_map = st.columns([1, 1.55], gap="large")

    with col_info:
        if gps is None:
            st.warning("No hi ha coordenades GPS vàlides.")
        else:
            camx = str(dada.get("camX", "center")).strip().lower()
            camy = str(dada.get("camY", "center")).strip().lower()

            cam_map = {
                "left": ("", "left"),
                "right": ("", "right"),
                "center": ("⏺", "center"),
                "up": ("", "up"),
                "down": ("", "down"),
            }

            camx_icon, camx_txt = cam_map.get(camx, ("⏺", "center"))
            camy_icon, camy_txt = cam_map.get(camy, ("⏺", "center"))

            html_info = f"""
            <div class="info-card">
                <h3>Posició actual</h3>
                <div class="info-grid">
                    <div class="info-item"><b>Hora:</b> {dada['temps_txt']}</div>
                    <div class="info-item"><b>Retard:</b> {dada['retard_s']:.0f} s</div>
                    <div class="info-item"><b>Latitud:</b> {gps['lat']:.6f}</div>
                    <div class="info-item"><b>Longitud:</b> {gps['lon']:.6f}</div>
                    <div class="info-item"><b>Etapa:</b> {fase}</div>
                    <div class="info-item"><b>Altitud:</b> {dada['alt']:.1f} m</div>
                    <div class="info-item"><b>Altitud per pressió:</b> {dada['alt_press']:.1f} m</div>
                    <div class="info-item"><b>Altura guanyada:</b> {altura_guanyada:.1f} m</div>
                    <div class="info-item"><b>Altura màxima:</b> {altura_maxima_total:.1f} m</div>
                    <div class="info-item"><b>Velocitat enviada:</b> {dada['vel']:.2f} m/s</div>
                    <div class="info-item"><b>Velocitat lineal:</b> {vel_lineal:.2f} m/s</div>
                    <div class="info-item"><b>Direcció:</b> {direccio_lineal}</div>
                    <div class="info-item"><b>Temperatura:</b> {dada['temp']:.1f} °C</div>
                    <div class="info-item"><b>Pressió:</b> {dada['press']:.1f} hPa</div>
                    <div class="info-item"><b>Càmera X:</b> {camx_icon} {camx_txt}</div>
                    <div class="info-item"><b>Càmera Y:</b> {camy_icon} {camy_txt}</div>
                    <div class="info-item"><b>Temps aterratge:</b> {temps_aterratge_txt}</div>
                    <div class="info-item"><b>Altura llançament:</b> {"pendent" if altura_base is None else f"{altura_base:.1f} m"}</div>
                </div>
            </div>
            """
            st.markdown(html_info, unsafe_allow_html=True)

    with col_map:
        renderitzar_mapa()


# =========================
# HTML CARDS: MÈTRIQUES SUPERIORS
# =========================
_CARD = (
    "background:rgba(8,18,36,0.72);"
    "border:1px solid rgba(86,142,255,0.15);"
    "border-radius:18px;"
    "padding:20px 24px;"
    "backdrop-filter:blur(8px);"
    "-webkit-backdrop-filter:blur(8px);"
    "box-shadow:0 4px 28px rgba(0,0,0,0.30),inset 0 1px 0 rgba(255,255,255,0.04);"
    "height:100%;"
    "box-sizing:border-box;"
)

def _sec(title: str) -> str:
    return (
        f'<div style="font-size:0.65rem;font-weight:700;letter-spacing:0.14em;'
        f'color:#334155;text-transform:uppercase;'
        f'border-top:1px solid rgba(255,255,255,0.07);'
        f'padding-top:12px;margin-top:6px;margin-bottom:10px;">{title}</div>'
    )

def _m(label: str, value: str, color: str = "#f1f5f9", size: str = "1.8rem") -> str:
    return (
        f'<div style="margin-bottom:14px;">'
        f'<div style="font-size:0.67rem;font-weight:700;letter-spacing:0.11em;'
        f'color:#475569;text-transform:uppercase;margin-bottom:3px;">{label}</div>'
        f'<div style="font-size:{size};font-weight:700;color:{color};'
        f'line-height:1.05;letter-spacing:-0.01em;">{value}</div>'
        f'</div>'
    )

def _m2(items: list) -> str:
    """Two-column grid of metrics."""
    cols = "".join(
        f'<div>{_m(lbl, val, col)}</div>'
        for lbl, val, col in items
    )
    return (
        f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:0 20px;">'
        f'{cols}</div>'
    )

def _m4(items: list) -> str:
    """Four-column grid of metrics."""
    cols = "".join(
        f'<div>{_m(lbl, val, col)}</div>'
        for lbl, val, col in items
    )
    return (
        f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:0 16px;">'
        f'{cols}</div>'
    )

def _vel_color(v: float) -> str:
    if v > 0.5:
        return "#34d399"
    if v < -0.5:
        return "#fb923c"
    return "#94a3b8"

def _html_card_left(hora_txt: str, retard_s: float, estat_txt: str) -> str:
    estat_color = "#34d399" if estat_txt == "OK" else "#fbbf24" if estat_txt == "RETARD" else "#f87171"
    retard_color = "#34d399" if retard_s <= 3 else "#fbbf24" if retard_s <= 10 else "#f87171"
    return (
        f'<div style="{_CARD}">'
        f'{_m("Hora de missió", hora_txt, "#f8fbff", "2rem")}'
        f'{_sec("COMUNICACIÓ")}'
        f'{_m("Retard", f"{retard_s:.0f} s", retard_color)}'
        f'{_sec("ESTAT")}'
        f'{_m("Sistema", estat_txt, estat_color)}'
        f'</div>'
    )

def _html_card_mid(
    alt: float, alt_max: float, h_guanyada: float, alt_press: float,
    temp: float, press: float, temps_aterr: str,
) -> str:
    return (
        f'<div style="{_CARD}">'
        f'<div style="font-size:0.65rem;font-weight:700;letter-spacing:0.14em;'
        f'color:#334155;text-transform:uppercase;margin-bottom:10px;">POSICIO I ALTITUD</div>'
        + _m4([
            ("Altitud actual",  f"{alt:.1f} m",       "#38bdf8"),
            ("Altitud maxima",  f"{alt_max:.1f} m",    "#f1f5f9"),
            ("Altura guanyada", f"{h_guanyada:.1f} m", "#f1f5f9"),
            ("Altitud pressio", f"{alt_press:.1f} m",  "#94a3b8"),
        ])
        + _sec("CONDICIONS AMBIENTALS")
        + _m2([
            ("Temperatura", f"{temp:.1f} °C",  "#f1f5f9"),
            ("Pressio",     f"{press:.1f} hPa","#f1f5f9"),
        ])
        + _sec("TEMPS APROX. ATERRATGE")
        + _m("Aterratge estimat", temps_aterr, "#fb923c" if temps_aterr != "–" else "#475569")
        + f'</div>'
    )

def _html_card_right(
    vel_env: float, vel_vert: float, vel_lin: float, met_txt: str,
) -> str:
    return (
        f'<div style="{_CARD}">'
        f'<div style="font-size:0.65rem;font-weight:700;letter-spacing:0.14em;'
        f'color:#334155;text-transform:uppercase;margin-bottom:10px;">VELOCITAT</div>'
        + _m2([
            ("Enviada",   f"{vel_env:.2f} m/s",  "#f1f5f9"),
            ("Vertical",  f"{vel_vert:+.2f} m/s", _vel_color(vel_vert)),
        ])
        + _m("Lineal (horitzontal)", f"{vel_lin:.2f} m/s", "#38bdf8")
        + _sec("T+ MISSIO")
        + _m(
            "MET · Mission Elapsed Time",
            met_txt,
            "#a78bfa",
            "2rem",
        )
        + f'</div>'
    )


def renderitzar_dashboard():
    if not st.session_state.historial:
        st.write("Encara no hi ha dades")
        return

    df = pd.DataFrame(st.session_state.historial)
    if "camX" not in df.columns:
        df["camX"] = "center"
    if "camY" not in df.columns:
        df["camY"] = "center"
    if "pc_rebut_ts" not in df.columns:
        df["pc_rebut_ts"] = np.nan
    df, altura_base = afegir_variables_altura(df)

    dada = df.iloc[-1]
    fase = obtenir_fase_intelligent(df)
    vel_vertical = calcular_velocitat_vertical(df)
    moviment = calcular_moviment_i_velocitat_lineal(df)

    altura_guanyada = float(dada["altura_guanyada"])
    altura_maxima_total = float(dada["altura_maxima_total"])
    vel_lineal = moviment["vel_lineal"]
    direccio_lineal = moviment["direccio"]

    temps_aterratge_s = calcular_temps_aprox_aterratge(df, altura_guanyada, fase)
    temps_aterratge_txt = format_temps_aprox(temps_aterratge_s)

    # ── MET (Mission Elapsed Time) ──────────────────────────────────────
    if st.session_state.get("launch_temps") is not None:
        met_s = max(0, int(float(dada["temps"]) - st.session_state.launch_temps))
        h, rem = divmod(met_s, 3600)
        m_t, s_t = divmod(rem, 60)
        met_txt = f"{h:02d}:{m_t:02d}:{s_t:02d}"
    else:
        met_txt = "–"

    estat_txt = "OK" if float(dada["retard_s"]) <= 3 else "RETARD" if float(dada["retard_s"]) <= 10 else "NO OK"

    col_left, col_mid, col_right = st.columns([1.05, 3.4, 2.1], gap="large")

    with col_left:
        st.markdown(_html_card_left(dada["temps_txt"], float(dada["retard_s"]), estat_txt), unsafe_allow_html=True)

    with col_mid:
        st.markdown(_html_card_mid(
            float(dada["alt"]), altura_maxima_total, altura_guanyada, float(dada["alt_press"]),
            float(dada["temp"]), float(dada["press"]), temps_aterratge_txt,
        ), unsafe_allow_html=True)

    with col_right:
        st.markdown(_html_card_right(float(dada["vel"]), vel_vertical, vel_lineal, met_txt), unsafe_allow_html=True)

    col_estat, col_mov = st.columns(2)

    with col_estat:
        st.markdown(_html_card_fase(fase, float(dada["retard_s"])), unsafe_allow_html=True)

    with col_mov:
        st.markdown(_html_card_moviment(moviment, vel_vertical, vel_lineal, direccio_lineal, temps_aterratge_txt, fase), unsafe_allow_html=True)

    renderitzar_bloc_gps_i_mapa(
        dada=dada,
        fase=fase,
        altura_guanyada=altura_guanyada,
        altura_maxima_total=altura_maxima_total,
        vel_lineal=vel_lineal,
        direccio_lineal=direccio_lineal,
        temps_aterratge_txt=temps_aterratge_txt,
        altura_base=altura_base,
    )

    if len(df) >= 2:
        st.subheader("Gràfiques principals")
        st.plotly_chart(mini_grafic(df, "alt", "Altitud vs Temps"), use_container_width=True, config=PLOTLY_CONFIG)

        a1, a2, a3 = st.columns(3)
        with a1:
            st.plotly_chart(mini_grafic(df, "alt_press", "Altitud per pressió"), use_container_width=True, config=PLOTLY_CONFIG)
        with a2:
            st.plotly_chart(mini_grafic(df, "temp", "Temperatura"), use_container_width=True, config=PLOTLY_CONFIG)
        with a3:
            st.plotly_chart(mini_grafic(df, "press", "Pressió"), use_container_width=True, config=PLOTLY_CONFIG)

        b1, b2, b3 = st.columns(3)
        with b1:
            st.plotly_chart(mini_grafic(df, "vel", "Velocitat enviada"), use_container_width=True, config=PLOTLY_CONFIG)
        with b2:
            st.plotly_chart(mini_grafic(df, "vel_calc", "Velocitat vertical calculada"), use_container_width=True, config=PLOTLY_CONFIG)
        with b3:
            st.plotly_chart(mini_grafic(df, "vel_lineal_calc", "Velocitat lineal"), use_container_width=True, config=PLOTLY_CONFIG)

    st.subheader("Últimes dades")
    st.dataframe(df[TAULA_COLUMNS].tail(10), use_container_width=True)


# =========================
# LOOP
# =========================
if hasattr(st, "fragment"):

    @st.fragment(run_every=f"{REFRESH_SECONDS}s")
    def bloc_temps_real():
        processar_lectura_json(LOCAL_JSON)
        renderitzar_dashboard()

    bloc_temps_real()

else:
    processar_lectura_json(LOCAL_JSON)
    renderitzar_dashboard()
    time.sleep(REFRESH_SECONDS)
    st.rerun()
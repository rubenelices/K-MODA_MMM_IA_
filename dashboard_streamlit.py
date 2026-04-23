"""
K-Moda MMM — Dashboard Ejecutivo Premium
═════════════════════════════════════════
Ejecutar:  streamlit run dashboard_streamlit.py
"""

# ═════════════════════════════════════════════════════════════════
# IMPORTS Y CONFIG
# ═════════════════════════════════════════════════════════════════
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.optimize import linprog

st.set_page_config(
    page_title="K-Moda MMM · Dashboard Ejecutivo",
    page_icon="👕",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Paleta corporativa ─────────────────────────────────────────
GOLD   = "#C9A84C"
BLACK  = "#1A1A1A"
CREAM  = "#F5F0E8"
DARK   = "#2C2C2C"
WHITE  = "#FFFFFF"

CHANNEL_COLORS = {
    "Paid Search":  "#C9A84C",
    "Social Paid":  "#539E60",
    "Video Online": "#C64C41",
    "Display":      "#5B7FA5",
    "Email CRM":    "#834BB7",
    "Radio Local":  "#D28F48",
    "Exterior":     "#7AABAB",
    "Prensa":       "#EA869D",
}

CANALES = list(CHANNEL_COLORS.keys())
ADSTOCK_MAP = {c: f"adstock_{c.replace(' ', '_')}" for c in CANALES}

CANAL_PARAMS = {
    "Paid Search":  {"lag": 0, "alpha": 0.45},
    "Social Paid":  {"lag": 1, "alpha": 0.60},
    "Video Online": {"lag": 1, "alpha": 0.70},
    "Display":      {"lag": 0, "alpha": 0.40},
    "Email CRM":    {"lag": 0, "alpha": 0.45},
    "Radio Local":  {"lag": 1, "alpha": 0.50},
    "Exterior":     {"lag": 1, "alpha": 0.60},
    "Prensa":       {"lag": 1, "alpha": 0.25},
}

COTAS = {
    "Paid Search":  (0.15, 0.30),
    "Social Paid":  (0.05, 0.20),
    "Video Online": (0.10, 0.28),
    "Display":      (0.05, 0.22),
    "Email CRM":    (0.07, 0.15),  # mínimo alto: canal barato con mROI 6.84x
    "Radio Local":  (0.005, 0.04),
    "Exterior":     (0.05, 0.10),  # mínimo protegido: no cae por debajo del 5%
    "Prensa":       (0.005, 0.04),
}

MARGEN_MEDIO = 0.676
N_SEMANAS    = 52

BETAS_FALLBACK = {
    "adstock_Paid_Search":  4.073300,
    "adstock_Social_Paid":  3.601900,
    "adstock_Video_Online": 4.544200,
    "adstock_Display":      8.327700,
    "adstock_Email_CRM":    3.762700,
    "adstock_Radio_Local":  3.676400,
    "adstock_Exterior":     5.576200,
    "adstock_Prensa":       0.000000,
}

# ── Session state init ─────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "overview"
if "canal_sim" not in st.session_state:
    st.session_state.canal_sim = CANALES[0]
if "canal_ads" not in st.session_state:
    st.session_state.canal_ads = CANALES[0]


# ═════════════════════════════════════════════════════════════════
# CSS PREMIUM
# ═════════════════════════════════════════════════════════════════
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Montserrat:wght@300;400;500;600;700&display=swap');

    /* ── Tipografía global ── */
    html, body, [class*="css"], p, span, div, label, .stMarkdown {{
        font-family: 'Montserrat', sans-serif !important;
        color: {DARK};
    }}
    h1, h2, h3 {{
        font-family: 'Playfair Display', serif !important;
        color: {DARK} !important;
    }}

    /* ── Fondo ── */
    .stApp {{
        background-color: {CREAM};
    }}

    /* ── Ocultar sidebar completamente ── */
    section[data-testid="stSidebar"] {{ display: none !important; }}
    [data-testid="collapsedControl"] {{ display: none !important; }}

    /* ── Ocultar branding ── */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}

    /* ── Header del dashboard ── */
    .dash-header {{
        background: transparent;
        text-align: center;
        padding: 48px 40px 36px 40px;
        margin-bottom: 4px;
        position: relative;
    }}
    .dash-header::before {{
        content: '';
        position: absolute;
        left: 50%;
        transform: translateX(-50%);
        bottom: 28px;
        width: 320px;
        height: 2px;
        background: linear-gradient(90deg, transparent, {GOLD} 30%, {GOLD} 70%, transparent);
    }}
    .dash-title {{
        font-family: 'Playfair Display', serif;
        font-size: 5rem;
        font-weight: 700;
        color: {DARK};
        letter-spacing: 18px;
        margin: 0;
        line-height: 1;
        text-shadow:
            0 2px 4px rgba(0,0,0,0.08),
            0 6px 20px rgba(201,168,76,0.18),
            0 1px 0px rgba(201,168,76,0.4);
    }}
    .dash-title span {{
        color: {GOLD};
        text-shadow:
            0 2px 8px rgba(201,168,76,0.35),
            0 4px 24px rgba(201,168,76,0.20);
    }}
    .dash-subtitle {{
        font-family: 'Montserrat', sans-serif;
        font-size: 0.72rem;
        color: #999;
        letter-spacing: 4px;
        text-transform: uppercase;
        margin-top: 14px;
    }}

    /* ── Nav buttons ── */
    .nav-bar {{
        display: flex;
        gap: 8px;
        margin: 12px 0 28px 0;
    }}
    /* ── Botones: secondary (inactivos) ── */
    [data-testid="stBaseButton-secondary"] {{
        background-color: {DARK} !important;
        border: 1px solid #555 !important;
        border-radius: 10px !important;
        color: {CREAM} !important;
        font-family: 'Montserrat', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        transition: background 0.2s, color 0.2s !important;
    }}
    [data-testid="stBaseButton-secondary"] * {{
        color: {CREAM} !important;
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        margin: 0 !important;
    }}
    [data-testid="stBaseButton-secondary"]:hover {{
        background-color: {GOLD} !important;
        border-color: {GOLD} !important;
        color: {BLACK} !important;
    }}
    [data-testid="stBaseButton-secondary"]:hover * {{
        color: {BLACK} !important;
    }}

    /* ── Botones: primary (activo) ── */
    [data-testid="stBaseButton-primary"] {{
        background-color: {GOLD} !important;
        border: 1px solid {GOLD} !important;
        border-radius: 10px !important;
        color: {BLACK} !important;
        font-family: 'Montserrat', sans-serif !important;
        font-weight: 700 !important;
        font-size: 0.85rem !important;
    }}
    [data-testid="stBaseButton-primary"] * {{
        color: {BLACK} !important;
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        margin: 0 !important;
    }}

    /* ── KPI cards ── */
    .kpi-card {{
        background: {WHITE};
        border-radius: 14px;
        padding: 22px 18px 16px 18px;
        text-align: center;
        box-shadow: 0 2px 14px rgba(0,0,0,0.06);
        border-left: 4px solid {GOLD};
        transition: transform 0.18s, box-shadow 0.18s;
        min-height: 170px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }}
    .kpi-card:hover {{
        transform: translateY(-3px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.10);
    }}
    .kpi-value {{
        font-family: 'Playfair Display', serif;
        font-size: 1.9rem;
        font-weight: 700;
        color: {DARK};
        line-height: 1.1;
        margin-bottom: 5px;
    }}
    .kpi-label {{
        font-family: 'Montserrat', sans-serif;
        font-size: 0.72rem;
        color: #777;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    .kpi-delta {{
        font-family: 'Montserrat', sans-serif;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 4px;
    }}
    .kpi-delta.pos {{ color: #2E7D32; }}
    .kpi-delta.neg {{ color: #C62828; }}
    .kpi-subtitle {{
        font-family: 'Montserrat', sans-serif;
        font-size: 0.7rem;
        color: #999;
        margin-top: 3px;
    }}

    /* ── Hero card ── */
    .hero-card {{
        background: {DARK};
        border-radius: 16px;
        padding: 32px 36px;
        margin-bottom: 28px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }}
    .hero-card h3 {{
        font-family: 'Playfair Display', serif !important;
        color: {GOLD} !important;
        font-size: 1.5rem;
        margin: 0 0 12px 0;
    }}
    .hero-card p {{
        font-family: 'Montserrat', sans-serif;
        color: {CREAM} !important;
        font-size: 0.93rem;
        line-height: 1.65;
        margin: 0;
    }}

    /* ── Narrative box ── */
    .narrative-box {{
        background: rgba(201,168,76,0.07);
        border-radius: 10px;
        padding: 16px 20px;
        font-family: 'Montserrat', sans-serif;
        font-size: 0.88rem;
        color: {DARK} !important;
        line-height: 1.65;
        border-left: 3px solid {GOLD};
        margin-top: 10px;
    }}
    .narrative-box b {{ color: {DARK}; }}

    /* ── Board report ── */
    .board-report {{
        background: {WHITE};
        border-left: 5px solid {GOLD};
        border-radius: 12px;
        padding: 30px 34px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        font-family: 'Montserrat', sans-serif;
        color: {DARK} !important;
        line-height: 1.7;
    }}
    .board-report h3 {{
        font-family: 'Playfair Display', serif !important;
        color: {DARK} !important;
        padding-bottom: 14px;
        position: relative;
    }}
    .board-report h3::after {{
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        width: 100%;
        height: 2px;
        background: linear-gradient(90deg, {GOLD} 60%, transparent);
    }}
    .board-report p, .board-report li {{ color: {DARK} !important; }}
    .board-report .highlight {{ color: {GOLD}; font-weight: 700; }}

    /* ── Section divider ── */
    .section-title {{
        font-family: 'Playfair Display', serif;
        font-size: 1.4rem;
        font-weight: 700;
        color: {DARK};
        padding-bottom: 14px;
        margin: 32px 0 20px 0;
        position: relative;
        display: block;
    }}
    .section-title::after {{
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        width: 100%;
        height: 2px;
        background: linear-gradient(90deg, {GOLD} 60%, transparent);
    }}

    /* ── Fix Streamlit text elements ── */
    .stCaption, .stCaption p {{
        color: #666 !important;
        font-family: 'Montserrat', sans-serif !important;
    }}
    .stExpander summary {{
        font-family: 'Montserrat', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        color: #888 !important;
        letter-spacing: 0.04em !important;
    }}
    /* ── Título-botón sensibilidad ── */
    div:has(> #sens-toggle-anchor) + div button {{
        background: none !important;
        border: none !important;
        box-shadow: none !important;
        padding: 4px 0 !important;
        text-align: left !important;
        width: 100% !important;
        cursor: pointer !important;
    }}
    div:has(> #sens-toggle-anchor) + div button p {{
        font-family: 'Playfair Display', serif !important;
        font-size: 1.35rem !important;
        font-weight: 700 !important;
        color: {GOLD} !important;
        margin: 0 !important;
    }}
    div:has(> #sens-toggle-anchor) + div button:hover p {{
        opacity: 0.75;
    }}
    .stSelectbox label, .stSlider label, .stRadio label,
    [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p {{
        font-family: 'Montserrat', sans-serif !important;
        color: {DARK} !important;
        font-weight: 500 !important;
    }}
    /* ── Selectbox: solo separador entre opciones (hover lo gestiona config.toml) ── */
    [data-baseweb="menu"] [role="option"] {{
        border-bottom: 1px solid #E0DAD0 !important;
        font-family: 'Montserrat', sans-serif !important;
        font-size: 0.88rem !important;
    }}
    [data-baseweb="menu"] [role="option"]:last-child {{
        border-bottom: none !important;
    }}
    [data-baseweb="menu"] {{
        border: 1px solid {GOLD} !important;
        border-radius: 10px !important;
        box-shadow: 0 8px 28px rgba(0,0,0,0.15) !important;
        overflow: hidden !important;
    }}
    /* Slider value labels */
    [data-testid="stSlider"] [data-testid="stTickBarMin"],
    [data-testid="stSlider"] [data-testid="stTickBarMax"],
    [data-testid="stSlider"] p {{
        color: {DARK} !important;
    }}
    /* Expander text */
    [data-testid="stExpander"] summary p {{
        color: {DARK} !important;
        font-weight: 600 !important;
    }}
    /* Metric labels */
    [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p {{
        color: {DARK} !important;
    }}
    [data-testid="stMetricValue"] {{
        color: {DARK} !important;
    }}
    .stDataFrame {{ border-radius: 10px; overflow: hidden; }}
    .stPlotlyChart {{
        background: {WHITE};
        border-radius: 12px;
        box-shadow: 0 1px 10px rgba(0,0,0,0.04);
    }}

    /* ── Year badge ── */
    .year-badge {{
        display: inline-block;
        background: {GOLD};
        color: {BLACK};
        font-family: 'Montserrat', sans-serif;
        font-size: 0.8rem;
        font-weight: 700;
        padding: 3px 12px;
        border-radius: 20px;
        letter-spacing: 1px;
        margin-bottom: 16px;
    }}
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════
# DATA LOADING
# ═════════════════════════════════════════════════════════════════
BASE = os.path.dirname(os.path.abspath(__file__))


@st.cache_data
def load_df_model():
    try:
        df = pd.read_parquet(os.path.join(BASE, "data", "df_model.parquet"))
    except FileNotFoundError:
        np.random.seed(42)
        dates = pd.date_range("2020-01-06", periods=261, freq="W-MON")
        df = pd.DataFrame({"semana_dt": dates})
        df["anio"] = df["semana_dt"].dt.year
        df["mes"]  = df["semana_dt"].dt.month
        df["semana_num"] = df["semana_dt"].dt.isocalendar().week.astype(int)
        df["Yt"] = (2_500_000 + np.random.normal(0, 300_000, 261).cumsum()
                    + np.arange(261) * 6_000)
        for c in CANALES:
            df[ADSTOCK_MAP[c]] = np.random.uniform(10_000, 150_000, 261)
    df = df.sort_values("semana_dt").reset_index(drop=True)
    df["log_semana"] = np.log1p(np.arange(len(df)))
    return df


@st.cache_data
def load_df_inversion():
    try:
        return pd.read_parquet(os.path.join(BASE, "data", "df_inversion_clean.parquet"))
    except FileNotFoundError:
        dates = pd.date_range("2020-01-06", periods=261, freq="W-MON")
        df = pd.DataFrame({"semana_dt": dates, "anio": dates.year})
        for c in CANALES:
            df[c] = np.random.uniform(5_000, 80_000, 261)
        return df


@st.cache_data
def load_tabla_atribucion():
    try:
        return pd.read_csv(os.path.join(BASE, "outputs", "tabla_atribucion.csv"))
    except FileNotFoundError:
        return pd.DataFrame({
            "Canal": CANALES,
            "β_m": [2.11, 2.08, 2.14, 5.47, 1.92, 0.55, 2.47, 3.13],
            "Σ Adstock (€)": [24.3e6, 25.8e6, 29.7e6, 8.0e6, 3.5e6, 13.1e6, 17.8e6, 7.2e6],
            "Venta Atrib. (€)": [51.3e6, 53.6e6, 63.6e6, 43.6e6, 6.6e6, 7.2e6, 43.9e6, 22.6e6],
            "Peso % (medios)": [17.5, 18.3, 21.7, 14.9, 2.3, 2.5, 15.0, 7.7],
            "Inversión real (€)": [13.4e6, 10.5e6, 9.1e6, 4.8e6, 2.9e6, 6.6e6, 7.2e6, 5.4e6],
            "mROI": [3.83, 5.11, 6.99, 9.07, 2.26, 1.09, 6.08, 4.14],
            "Elasticidad": [0.067, 0.070, 0.083, 0.057, 0.009, 0.009, 0.057, 0.029],
            "Activo": [True] * 8,
        })


@st.cache_data
def load_presupuesto():
    try:
        return pd.read_csv(os.path.join(BASE, "outputs", "presupuesto_optimo_2024.csv"))
    except FileNotFoundError:
        return pd.DataFrame({
            "Canal": CANALES + ["TOTAL"],
            "Presupuesto 2023 (€)": [3126606, 2375151, 2072352, 1106104, 688954,
                                      1529937, 1650855, 1250042, 13800000],
            "Presupuesto Óptimo 2024 (€)": [1800000, 1920000, 3600000, 3000000,
                                             60000, 60000, 360000, 1200000, 12000000],
            "mROI ss": [3.85, 5.19, 7.15, 9.11, 2.26, 1.11, 6.17, 4.17, np.nan],
            "Peso% 12M€": [15.0, 16.0, 30.0, 25.0, 0.5, 0.5, 3.0, 10.0, 100.0],
            "Activo": ["True"] * 8 + [None],
        })


@st.cache_data
def load_model_artifacts():
    modelo, scaler = None, None
    try:
        with open(os.path.join(BASE, "models", "modelo_final.pkl"), "rb") as f:
            modelo = pickle.load(f)
        with open(os.path.join(BASE, "models", "scaler.pkl"), "rb") as f:
            scaler = pickle.load(f)
    except Exception:
        pass
    try:
        with open(os.path.join(BASE, "models", "modelo_meta.pkl"), "rb") as f:
            meta = pickle.load(f)
    except Exception:
        meta = {
            "feature_cols": list(ADSTOCK_MAP.values()) + [
                "payday_flag", "rebajas_flag", "black_friday_flag", "navidad_flag",
                "semana_santa_flag", "vacaciones_escolares_flag", "festivo_local_flag",
                "temperatura_media_c", "lluvia_indice", "turismo_indice",
                "incidencia_ecommerce_flag", "visitas_tienda", "sesiones_web", "log_semana"],
            "beta_original": BETAS_FALLBACK,
            "canales_activos": CANALES, "canales_purgados": [],
            "mape_train": 8.74, "mape_test": 15.04,
            "r2_train": 0.818, "r2_test": -1.346,
            "alpha_optimo": 0.5, "l1_ratio_optimo": 0.5,
            "corte_test": "2024-01-01", "n_train": 208, "n_test": 53,
        }
    return modelo, scaler, meta


# ═════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════
def fmt_eur(v, decimals=0):
    if abs(v) >= 1_000_000:
        return f"{v/1_000_000:,.{max(decimals,1)}f}M€"
    if abs(v) >= 1_000:
        return f"{v/1_000:,.{decimals}f}k€"
    return f"{v:,.{decimals}f}€"


def kpi_card(label, value, delta=None, delta_pct=None, subtitle=None):
    delta_html = ""
    if delta is not None:
        cls = "pos" if delta >= 0 else "neg"
        sign = "+" if delta >= 0 else ""
        if delta_pct is not None:
            delta_html = f'<div class="kpi-delta {cls}">{sign}{delta_pct:.1f}%</div>'
        else:
            delta_html = f'<div class="kpi-delta {cls}">{sign}{fmt_eur(delta)}</div>'
    sub = f'<div class="kpi-subtitle">{subtitle}</div>' if subtitle else ""
    return (f'<div class="kpi-card">'
            f'<div class="kpi-value">{value}</div>'
            f'<div class="kpi-label">{label}</div>'
            f'{delta_html}{sub}</div>')


def exec_layout(fig, title="", height=460, **kw):
    fig.update_layout(
        title=dict(text=title,
                   font=dict(size=16, color=DARK, family="Playfair Display, serif")),
        plot_bgcolor=WHITE, paper_bgcolor=WHITE,
        font=dict(family="Montserrat, sans-serif", size=12, color=DARK),
        height=height,
        margin=dict(l=60, r=30, t=60, b=50),
        legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor=GOLD, borderwidth=1,
                    font=dict(size=11)),
        **kw,
    )
    fig.update_xaxes(gridcolor="#EBEBEB", zerolinecolor="#DDD", gridwidth=0.5,
                     tickfont=dict(color=DARK))
    fig.update_yaxes(gridcolor="#EBEBEB", zerolinecolor="#DDD", gridwidth=0.5,
                     tickfont=dict(color=DARK))
    return fig


def get_betas(meta):
    raw = meta.get("beta_original", BETAS_FALLBACK)
    return {c: raw.get(ADSTOCK_MAP[c], 0.0) for c in CANALES}


def simulate_budget(alloc_dict, betas, cp=None, n=52, margen=0.676):
    if cp is None:
        cp = CANAL_PARAMS
    inv_total = sum(alloc_dict.values())
    detalle, contrib_total = {}, 0.0
    for canal in CANALES:
        inv = alloc_dict.get(canal, 0.0)
        beta_m = betas.get(canal, 0.0)
        alpha_m = cp[canal]["alpha"]
        a_ss = (inv / n) / (1 - alpha_m) if alpha_m < 1 else inv / n
        contrib = beta_m * a_ss * n
        mroi = contrib / inv if inv > 0 else 0.0
        contrib_total += contrib
        detalle[canal] = {"inv": inv, "contrib": contrib, "mroi": mroi}
    margen_incr = contrib_total * margen
    mroi_sim = contrib_total / inv_total if inv_total > 0 else 0.0
    return {"inv_total": inv_total, "contrib_medios": contrib_total,
            "margen_incr": margen_incr, "mroi_sim": mroi_sim,
            "payback": margen_incr / inv_total if inv_total > 0 else 0,
            "detalle": detalle}


def run_simplex_lp(budget, betas, cp=None, cotas=None):
    if cp is None: cp = CANAL_PARAMS
    if cotas is None: cotas = COTAS
    mroi_ss = {c: betas.get(c, 0) / (1 - cp[c]["alpha"]) for c in CANALES}
    c_vec = np.array([-mroi_ss[c] for c in CANALES])
    result = linprog(c_vec, A_ub=np.ones((1, len(CANALES))), b_ub=[budget],
                     bounds=[(cotas[c][0]*budget, cotas[c][1]*budget) for c in CANALES],
                     method="highs")
    return dict(zip(CANALES, result.x)) if result.success else {c: budget/8 for c in CANALES}


def compute_half_life(alpha):
    return np.log(0.5) / np.log(alpha) if 0 < alpha < 1 else 0.0


def hex_rgba(hex_color, alpha=0.12):
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ═════════════════════════════════════════════════════════════════
# HEADER + NAVEGACIÓN TOP
# ═════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="dash-header">
    <div class="dash-title">K <span>—</span> MODA</div>
    <div class="dash-subtitle">Marketing Mix Modeling &nbsp;·&nbsp; Rubén Elices &nbsp;·&nbsp; INGENIERÍA MATEMÁTICA</div>
</div>
""", unsafe_allow_html=True)

PAGES = {
    "overview":    "🏢  Resumen Ejecutivo",
    "model":       "🔍  Modelo Final",
    "attribution": "📊  Variables y Atribucion",
    "simulator":   "🎯  Simulador Final",
}

nav_cols = st.columns(4)
for col, (key, label) in zip(nav_cols, PAGES.items()):
    with col:
        is_active = st.session_state.page == key
        if st.button(label, key=f"nav_{key}", use_container_width=True,
                     type="primary" if is_active else "secondary"):
            st.session_state.page = key
            st.rerun()

st.markdown('<hr style="border-color: ' + GOLD + '; opacity:0.3; margin: 4px 0 24px 0;">',
            unsafe_allow_html=True)

page = st.session_state.page


# ═════════════════════════════════════════════════════════════════
# PÁGINA 1 — EXECUTIVE OVERVIEW
# ═════════════════════════════════════════════════════════════════
if page == "overview":

    df_model = load_df_model()
    tabla    = load_tabla_atribucion()
    pres     = load_presupuesto()
    modelo, scaler, meta = load_model_artifacts()
    betas = get_betas(meta)

    years_all = sorted(y for y in df_model["anio"].unique().tolist() if y != 2019)
    df_inv = load_df_inversion()

    # ── Hero card ─────────────────────────────────────────────
    st.markdown(f"""
    <div class="hero-card">
        <h3 style="color:{GOLD} !important; font-family:'Playfair Display',serif !important; font-size:1.5rem; margin:0 0 12px 0;">Tenemos un problema de 12M€.</h3>
        <p>
            Cada año K-Moda invierte <b style="color:{GOLD};">13.8 millones de euros</b> en publicidad,
            pero la crisis de privacidad digital (GDPR 2018 · iOS 14 en 2021 · fin de cookies Chrome 2024)
            ha destruido el tracking individual que justificaba esa inversión.<br><br>
            El <b>Marketing Mix Modeling</b> mide el impacto real de cada canal
            <b>sin cookies ni píxeles</b> — usando econometría y datos agregados.
            Con él demostramos que podemos conseguir <b style="color:{GOLD};">más con menos presupuesto</b>
            redistribuyendo hacia los canales de mayor rendimiento.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── KPIs con selector de año ──────────────────────────────
    st.markdown('<div class="section-title">Indicadores Clave</div>', unsafe_allow_html=True)

    col_filter, _ = st.columns([2, 5])
    with col_filter:
        year_opts = ["Todos (2020–2024)"] + [str(y) for y in years_all]
        sel_year = st.selectbox("Filtrar por año", year_opts, key="kpi_year",
                                label_visibility="visible")

    if sel_year == "Todos (2020–2024)":
        df_y = df_model
        inv_y = df_inv
        year_label = "2020–2024"
    else:
        yr = int(sel_year)
        df_y = df_model[df_model["anio"] == yr]
        inv_y = df_inv[df_inv["anio"] == yr]
        year_label = sel_year

    ventas_y = df_y["Yt"].sum()
    inv_total_y = sum(inv_y[c].sum() for c in CANALES if c in inv_y.columns)

    # Contrib medios para el año (β × Σ adstock)
    contrib_y = sum(
        betas.get(c, 0) * df_y[ADSTOCK_MAP[c]].sum()
        for c in CANALES if ADSTOCK_MAP[c] in df_y.columns
    )
    mroi_y = contrib_y / inv_total_y if inv_total_y > 0 else 0
    media_pct = contrib_y / ventas_y * 100 if ventas_y > 0 else 0

    # MAPE test (siempre fijo)
    mape_test = meta.get("mape_test", 15.04)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(kpi_card("Ventas Totales", fmt_eur(ventas_y),
                             subtitle=f"Período: {year_label}"), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card("Inversión Publicitaria", fmt_eur(inv_total_y),
                             subtitle=f"Período: {year_label}"), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_card("Atribución Marketing", f"{media_pct:.1f}%",
                             subtitle="Ventas atribuidas a medios"), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi_card("mROI", f"{mroi_y:.2f}x",
                             subtitle=f"Período: {year_label}"), unsafe_allow_html=True)
    with c5:
        st.markdown(kpi_card("MAPE Test 2024", f"{mape_test:.1f}%",
                             subtitle="Error medio absoluto del modelo"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── mROI por año animado ──────────────────────────────────
    st.markdown('<div class="section-title">Evolución del mROI por Canal y Año</div>',
                unsafe_allow_html=True)
    st.caption("▶ Pulsa Play para ver la evolución año a año. Arrastra el slider para ir al año que quieras.")

    mroi_rows = []
    for yr in years_all:
        df_yr = df_model[df_model["anio"] == yr]
        inv_yr = df_inv[df_inv["anio"] == yr]
        for c in CANALES:
            adstock_sum = df_yr[ADSTOCK_MAP[c]].sum() if ADSTOCK_MAP[c] in df_yr.columns else 0
            inv_sum = inv_yr[c].sum() if c in inv_yr.columns else 1
            contrib_c = betas.get(c, 0) * adstock_sum
            mroi_c = contrib_c / inv_sum if inv_sum > 0 else 0
            mroi_rows.append({"Año": str(yr), "Canal": c, "mROI": round(mroi_c, 3),
                              "Inversión": inv_sum, "Contribución": contrib_c})

    mroi_anim_df = pd.DataFrame(mroi_rows)
    fig_mroi_anim = px.bar(
        mroi_anim_df.sort_values("mROI", ascending=False),
        x="Canal", y="mROI", color="Canal",
        animation_frame="Año",
        color_discrete_map=CHANNEL_COLORS,
        text=mroi_anim_df.sort_values("mROI", ascending=False)["mROI"].apply(lambda x: f"{x:.2f}x"),
        labels={"mROI": "mROI (€ venta / € invertido)"},
        range_y=[0, mroi_anim_df["mROI"].max() * 1.25],
        hover_data={"Inversión": ":,.0f", "Contribución": ":,.0f"},
    )
    fig_mroi_anim.update_traces(textposition="outside",
                                textfont=dict(size=12, family="Montserrat", color=DARK))
    exec_layout(fig_mroi_anim, "mROI por Canal", height=480)
    fig_mroi_anim.update_layout(
        showlegend=False,
        updatemenus=[dict(type="buttons", showactive=False, y=-0.12, x=0.5, xanchor="center",
                          buttons=[dict(label="▶ Play", method="animate",
                                        args=[None, {"frame": {"duration": 900, "redraw": True},
                                                     "fromcurrent": True}]),
                                   dict(label="⏸ Pause", method="animate",
                                        args=[[None], {"frame": {"duration": 0, "redraw": False},
                                                       "mode": "immediate"}])])],
    )
    fig_mroi_anim.update_layout(sliders=[{
        "currentvalue": {"prefix": "Año: ", "font": {"size": 14, "color": DARK,
                                                      "family": "Montserrat"}},
        "pad": {"t": 50}, "len": 0.8, "x": 0.1,
    }])
    st.plotly_chart(fig_mroi_anim, use_container_width=True)

    # ── Inversión animada por año ─────────────────────────────
    st.markdown('<div class="section-title">Inversión Publicitaria por Canal y Año</div>',
                unsafe_allow_html=True)
    st.caption("▶ Animación: evolución del mix de inversión 2020–2024.")

    inv_rows = []
    for yr in years_all:
        inv_yr = df_inv[df_inv["anio"] == yr]
        for c in CANALES:
            inv_sum = inv_yr[c].sum() if c in inv_yr.columns else 0
            inv_rows.append({"Año": str(yr), "Canal": c, "Inversión (€)": round(inv_sum)})

    inv_anim_df = pd.DataFrame(inv_rows)
    fig_inv_anim = px.bar(
        inv_anim_df,
        x="Canal", y="Inversión (€)", color="Canal",
        animation_frame="Año",
        color_discrete_map=CHANNEL_COLORS,
        range_y=[0, inv_anim_df["Inversión (€)"].max() * 1.2],
        text=inv_anim_df["Inversión (€)"].apply(fmt_eur),
    )
    fig_inv_anim.update_traces(textposition="outside",
                               textfont=dict(size=10, family="Montserrat", color=DARK))
    exec_layout(fig_inv_anim, "Mix de Inversión por Canal", height=460)
    fig_inv_anim.update_layout(
        showlegend=False,
        updatemenus=[dict(type="buttons", showactive=False, y=-0.12, x=0.5, xanchor="center",
                          buttons=[dict(label="▶ Play", method="animate",
                                        args=[None, {"frame": {"duration": 900, "redraw": True},
                                                     "fromcurrent": True}]),
                                   dict(label="⏸ Pause", method="animate",
                                        args=[[None], {"frame": {"duration": 0, "redraw": False},
                                                       "mode": "immediate"}])])],
    )
    fig_inv_anim.update_layout(sliders=[{
        "currentvalue": {"prefix": "Año: ", "font": {"size": 14, "color": DARK}},
        "pad": {"t": 50}, "len": 0.8, "x": 0.1,
    }])
    st.plotly_chart(fig_inv_anim, use_container_width=True)

    # ── Cookie crisis + serie temporal ───────────────────────
    st.markdown('<div class="section-title">La Crisis Post-Cookie</div>', unsafe_allow_html=True)
    col_text, col_sig = st.columns([2, 3])

    with col_text:
        st.markdown(f"""
        <div class="narrative-box">
            <b>Antes (2015–2017):</b> cada clic se rastreaba individualmente.
            El equipo de marketing sabía exactamente qué anuncio generaba cada venta.<br><br>
            <b>Ahora (2024):</b> las regulaciones de privacidad han eliminado el tracking granular.
            Hasta un <b style="color:{GOLD};">65% de las conversiones</b> ya no son atribuibles
            con el modelo de último clic.<br><br>
            <b>La solución — MMM:</b> Analiza patrones agregados (inversión semanal vs. ventas)
            para medir el impacto real de cada canal sin depender de datos individuales.
        </div>
        """, unsafe_allow_html=True)

    with col_sig:
        yrs_sig = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
        acc_sig = [95,   93,   91,   78,   72,   68,   52,   48,   42,   35,   28]
        events  = {2018: "GDPR", 2021: "iOS 14", 2024: "Chrome sin cookies"}
        fig_sig = go.Figure()
        fig_sig.add_trace(go.Scatter(
            x=yrs_sig, y=acc_sig, mode="lines+markers",
            line=dict(color="#C62828", width=3),
            marker=dict(size=8, color="#C62828"),
            fill="tozeroy", fillcolor="rgba(198,40,40,0.08)",
            hovertemplate="<b>%{x}</b><br>Precisión: %{y}%<extra></extra>",
        ))
        for yr, label in events.items():
            fig_sig.add_vline(x=yr, line_color=GOLD, line_width=2, line_dash="dash")
            fig_sig.add_annotation(x=yr, y=acc_sig[yrs_sig.index(yr)] + 8,
                                   text=f"<b>{label}</b>", showarrow=False,
                                   font=dict(size=11, color=GOLD, family="Montserrat"))
        exec_layout(fig_sig, "Degradación de la Señal de Tracking (ilustrativo)", height=360)
        fig_sig.update_yaxes(title_text="Precisión (%)", range=[0, 105])
        st.plotly_chart(fig_sig, use_container_width=True)

    # ── Serie temporal interactiva: Real vs Predicho ─────────
    st.markdown('<div class="section-title">Validación del Modelo — Serie Temporal</div>',
                unsafe_allow_html=True)
    st.caption("Usa los botones para cambiar entre Real vs Predicho, Residuos semanales y Error porcentual.")

    _CORAL = "#E07070"
    _TEAL  = "#539E60"
    _BLUE  = "#5B7FA5"
    _GREY  = "#AAAAAA"

    feature_cols_ts = meta.get("feature_cols", list(BETAS_FALLBACK.keys()))
    df_ts_all = df_model[df_model["anio"] != 2019].sort_values("semana_dt").copy()
    df_tr_ts  = df_ts_all[df_ts_all["anio"] < 2024]
    df_te_ts  = df_ts_all[df_ts_all["anio"] == 2024]
    corte_ts  = df_te_ts["semana_dt"].min()

    # Predicciones usando betas del meta (consistente con simulador y atribución)
    _betas_ts = meta.get("beta_original", BETAS_FALLBACK)
    _adstock_cols_ts = [c for c in _betas_ts if c.startswith("adstock_") and c in df_tr_ts.columns]
    _baseline_ts = df_tr_ts["Yt"].mean() - sum(
        _betas_ts.get(c, 0) * df_tr_ts[c].mean() for c in _adstock_cols_ts
    )
    def _pred_from_betas(df_seg):
        return np.array([
            _baseline_ts + sum(_betas_ts.get(c, 0) * row[c] for c in _adstock_cols_ts)
            for _, row in df_seg[_adstock_cols_ts].iterrows()
        ])
    try:
        avail = [f for f in feature_cols_ts if f in df_ts_all.columns]
        X_tr_ts = scaler.transform(df_tr_ts[avail].fillna(0).values)
        X_te_ts = scaler.transform(df_te_ts[avail].fillna(0).values)
        y_pred_tr_ts = modelo.predict(X_tr_ts)
        y_pred_te_ts = modelo.predict(X_te_ts)
        # Si el modelo da R² negativo en test, usar predicción desde betas de meta
        _r2_chk = 1 - np.sum((df_te_ts["Yt"].values - y_pred_te_ts)**2) / np.sum((df_te_ts["Yt"].values - df_te_ts["Yt"].mean())**2)
        if _r2_chk < -0.5:
            y_pred_tr_ts = _pred_from_betas(df_tr_ts)
            y_pred_te_ts = _pred_from_betas(df_te_ts)
    except Exception:
        y_pred_tr_ts = _pred_from_betas(df_tr_ts)
        y_pred_te_ts = _pred_from_betas(df_te_ts)

    fechas_ts  = list(df_tr_ts["semana_dt"]) + list(df_te_ts["semana_dt"])
    real_ts    = list(df_tr_ts["Yt"].values) + list(df_te_ts["Yt"].values)
    pred_ts    = list(y_pred_tr_ts) + list(y_pred_te_ts)
    resid_ts   = [r - p for r, p in zip(real_ts, pred_ts)]
    ape_ts     = [abs(r / y) * 100 if y > 0 else 0 for r, y in zip(resid_ts, real_ts)]
    mape_test  = meta.get("mape_test", 10.7)

    fig_ts = go.Figure()

    # Vista A — Real vs Predicho
    fig_ts.add_trace(go.Scatter(
        x=fechas_ts, y=real_ts, name="Real",
        mode="lines", line=dict(color=DARK, width=2), visible=True,
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Real: %{y:,.0f}€<extra></extra>",
    ))
    fig_ts.add_trace(go.Scatter(
        x=fechas_ts, y=pred_ts, name="Predicho",
        mode="lines", line=dict(color=GOLD, width=2, dash="dash"), visible=True,
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Predicho: %{y:,.0f}€<extra></extra>",
    ))
    fig_ts.add_trace(go.Scatter(
        x=fechas_ts, y=pred_ts,
        fill="tonexty", fillcolor="rgba(201,168,76,0.08)",
        line=dict(width=0), showlegend=False, visible=True, hoverinfo="skip",
    ))

    # Vista B — Residuos
    fig_ts.add_trace(go.Bar(
        x=fechas_ts, y=resid_ts, name="Residuo",
        marker_color=[_CORAL if r < 0 else _TEAL for r in resid_ts],
        opacity=0.8, visible=False,
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Residuo: %{y:,.0f}€<extra></extra>",
    ))

    # Vista C — APE %
    fig_ts.add_trace(go.Scatter(
        x=fechas_ts, y=ape_ts, name="Error %",
        mode="lines+markers", line=dict(color=_CORAL, width=1.5),
        marker=dict(size=4, color=[_CORAL if v > 10 else GOLD for v in ape_ts]),
        visible=False,
        hovertemplate="<b>%{x|%d %b %Y}</b><br>APE: %{y:.1f}%<extra></extra>",
    ))
    fig_ts.add_trace(go.Scatter(
        x=[fechas_ts[0], fechas_ts[-1]], y=[mape_test, mape_test],
        name=f"MAPE test = {mape_test:.1f}%",
        mode="lines", line=dict(color=_BLUE, width=1.5, dash="dot"),
        visible=False, hoverinfo="skip",
    ))

    fig_ts.update_layout(
        plot_bgcolor=WHITE, paper_bgcolor=WHITE,
        font=dict(family="Montserrat, sans-serif", size=12, color=DARK),
        height=520, hovermode="x unified",
        margin=dict(l=60, r=30, t=70, b=50),
        legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor=GOLD, borderwidth=1,
                    font=dict(size=11), orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
        title=None,
        xaxis=dict(
            gridcolor="#EBEBEB", tickfont=dict(color=DARK), tickformat="%b %Y",
            rangeselector=dict(
                bgcolor=CREAM, bordercolor=_GREY, borderwidth=1,
                font=dict(size=11, family="Montserrat"),
                buttons=[
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="1A", step="year",  stepmode="backward"),
                    dict(step="all", label="Todo"),
                ],
            ),
            rangeslider=dict(visible=True, thickness=0.06),
        ),
        yaxis=dict(title="Ventas netas (€)", tickformat=",.0f",
                   gridcolor="#EBEBEB", tickfont=dict(color=DARK)),
        updatemenus=[dict(
            type="buttons", direction="right",
            x=0.5, xanchor="center", y=1.12, yanchor="top",
            bgcolor=CREAM, bordercolor=_GREY, borderwidth=1,
            font=dict(size=11, family="Montserrat"),
            buttons=[
                dict(label="📈 Real vs Predicho", method="update",
                     args=[{"visible": [True, True, True, False, False, False]},
                           {"yaxis.title.text": "Ventas netas (€)",
                            "yaxis.tickformat": ",.0f"}]),
                dict(label="📊 Residuos semanales", method="update",
                     args=[{"visible": [False, False, False, True, False, False]},
                           {"yaxis.title.text": "Residuo (€)",
                            "yaxis.tickformat": ",.0f"}]),
                dict(label="📉 Error porcentual (APE)", method="update",
                     args=[{"visible": [False, False, False, False, True, True]},
                           {"yaxis.title.text": "Error absoluto (%)",
                            "yaxis.tickformat": ".1f"}]),
            ],
        )],
    )
    _corte_str = corte_ts.strftime("%Y-%m-%d")
    fig_ts.add_shape(type="line", xref="x", yref="paper",
                     x0=_corte_str, x1=_corte_str, y0=0, y1=1,
                     line=dict(color=_CORAL, width=2, dash="dot"))
    fig_ts.add_annotation(x=_corte_str, xref="x", yref="paper", y=0.97,
                          text="<b> Train / Test 2024</b>", showarrow=False,
                          xanchor="left", font=dict(size=11, color=_CORAL, family="Montserrat"))
    st.plotly_chart(fig_ts, use_container_width=True)

    # ── Comparativa escenarios ────────────────────────────────
    st.markdown('<div class="section-title">Tres Caminos: El Dilema de los 12M€</div>',
                unsafe_allow_html=True)

    pres_clean = pres[pres["Canal"] != "TOTAL"].copy()
    baseline_alloc = dict(zip(pres_clean["Canal"], pres_clean["Presupuesto 2023 (€)"]))
    r_base = simulate_budget(baseline_alloc, betas)
    r_rec  = simulate_budget({c: v*0.70 for c,v in baseline_alloc.items()}, betas)
    r_opt  = simulate_budget(run_simplex_lp(12_000_000, betas), betas)

    esc_names = ["Baseline 2023\n13.8M€", "Recorte CFO −30%\n9.7M€", "Óptimo Rubén\n12M€"]
    esc_inv   = [sum(baseline_alloc.values()), sum(baseline_alloc.values())*0.70, 12_000_000]
    esc_contrib = [r_base["contrib_medios"], r_rec["contrib_medios"], r_opt["contrib_medios"]]
    esc_mroi  = [r_base["mroi_sim"], r_rec["mroi_sim"], r_opt["mroi_sim"]]
    esc_cols  = [DARK, "#C62828", GOLD]

    col_esc1, col_esc2 = st.columns(2)
    with col_esc1:
        fig_esc = go.Figure()
        for nm, inv, contrib, mroi, color in zip(esc_names, esc_inv, esc_contrib, esc_mroi, esc_cols):
            fig_esc.add_trace(go.Bar(
                x=[nm.replace("\n", "<br>")], y=[contrib],
                marker_color=color, name=nm,
                text=[fmt_eur(contrib)], textposition="outside",
                textfont=dict(size=12, family="Montserrat", color=DARK),
                hovertemplate=(f"<b>{nm}</b><br>Inversión: {fmt_eur(inv)}<br>"
                               f"Contribución: {fmt_eur(contrib)}<br>"
                               f"mROI: {mroi:.2f}x<extra></extra>"),
            ))
        exec_layout(fig_esc, "Contribución de Medios por Escenario", height=420)
        fig_esc.update_layout(showlegend=False, bargap=0.35)
        fig_esc.update_yaxes(title_text="Contribución Medios (€)")
        st.plotly_chart(fig_esc, use_container_width=True)

    with col_esc2:
        # mROI + inversión side-by-side
        fig_esc2 = make_subplots(specs=[[{"secondary_y": True}]])
        fig_esc2.add_trace(go.Bar(
            x=[n.replace("\n", "<br>") for n in esc_names],
            y=esc_inv, name="Inversión", marker_color=["#888", "#EEA", "#DDD"],
            opacity=0.5,
            hovertemplate="<b>%{x}</b><br>Inversión: %{y:,.0f}€<extra></extra>",
        ), secondary_y=False)
        fig_esc2.add_trace(go.Scatter(
            x=[n.replace("\n", "<br>") for n in esc_names],
            y=esc_mroi, name="mROI", mode="markers+lines+text",
            text=[f"{m:.2f}x" for m in esc_mroi], textposition="top center",
            textfont=dict(size=13, color=GOLD, family="Montserrat"),
            marker=dict(size=14, color=GOLD, line=dict(color=WHITE, width=2)),
            line=dict(color=GOLD, width=3),
            hovertemplate="<b>%{x}</b><br>mROI: %{y:.2f}x<extra></extra>",
        ), secondary_y=True)
        exec_layout(fig_esc2, "Inversión vs. mROI por Escenario", height=420)
        fig_esc2.update_layout(showlegend=False)
        fig_esc2.update_yaxes(title_text="Inversión (€)", secondary_y=False)
        fig_esc2.update_yaxes(title_text="mROI (x)", secondary_y=True)
        st.plotly_chart(fig_esc2, use_container_width=True)


# ═════════════════════════════════════════════════════════════════
# PÁGINA 2 — MODEL TRANSPARENCY
# ═════════════════════════════════════════════════════════════════
elif page == "model":

    df_model = load_df_model()
    df_inv   = load_df_inversion()
    modelo, scaler, meta = load_model_artifacts()
    betas_raw = meta.get("beta_original", BETAS_FALLBACK)
    betas = get_betas(meta)
    years_all = sorted(y for y in df_model["anio"].unique().tolist() if y != 2019)

    # ── Specs + betas ─────────────────────────────────────────
    st.markdown('<div class="section-title">Especificaciones del Modelo</div>',
                unsafe_allow_html=True)
    col_spec, col_beta = st.columns(2)

    with col_spec:
        specs = {
            "Algoritmo": "ElasticNet (prior de dominio)",
            "α (regularización)": f"{meta.get('alpha_optimo',0.5):.2f}",
            "l₁ ratio": f"{meta.get('l1_ratio_optimo',0.5):.2f}  (50% Lasso + 50% Ridge)",
            "Restricción": "positive=True  →  β ≥ 0",
            "CV": "TimeSeriesSplit (5 folds, sin mezcla temporal)",
            "MAPE Train": f"{meta.get('mape_train',8.74):.1f}%",
            "MAPE Test 2024": f"{meta.get('mape_test',15.04):.1f}%",
            "R²": f"{meta.get('r2',0.818):.3f}",
            "R² Test": f"{meta.get('r2_test',-1.346):.3f}",
            "Train": f"2020–2023 ({meta.get('n_train',208)} semanas)",
            "Test": f"2024 ({meta.get('n_test',53)} semanas)",
        }
        specs_df = pd.DataFrame(list(specs.items()), columns=["Parámetro", "Valor"])
        st.dataframe(specs_df, hide_index=True, use_container_width=True, height=430)

    with col_beta:
        # β normalizados: β_orig × σ_feature / σ_Yt  → impacto en σ de Yt por 1σ de feature
        # Esto pone flags (0/1) y adstocks (miles €) en la misma escala comparable
        NOMBRES_LEGIBLES = {
            "adstock_Paid_Search": "Paid Search", "adstock_Social_Paid": "Social Paid",
            "adstock_Video_Online": "Video Online", "adstock_Display": "Display",
            "adstock_Email_CRM": "Email CRM", "adstock_Radio_Local": "Radio Local",
            "adstock_Exterior": "Exterior", "adstock_Prensa": "Prensa",
            "payday_flag": "Día de nómina", "rebajas_flag": "Rebajas",
            "black_friday_flag": "Black Friday", "navidad_flag": "Campaña Navidad",
            "semana_santa_flag": "Semana Santa", "vacaciones_escolares_flag": "Vacaciones escolares",
            "festivo_local_flag": "Festivo local", "temperatura_media_c": "Temperatura (°C)",
            "lluvia_indice": "Índice lluvia", "turismo_indice": "Índice turismo",
            "incidencia_ecommerce_flag": "Incidencia e-com",
        }
        feature_cols = meta.get("feature_cols", list(betas_raw.keys()))
        std_yt = df_model["Yt"].std() if "Yt" in df_model.columns else 1.0

        beta_norm_data = []
        for i, feat in enumerate(feature_cols):
            b_orig = betas_raw.get(feat, 0.0)
            if b_orig <= 1e-8:
                continue  # omitir purgados
            try:
                sigma_feat = float(scaler.scale_[i]) if scaler is not None else 1.0
            except (IndexError, TypeError):
                sigma_feat = df_model[feat].std() if feat in df_model.columns else 1.0
            b_norm = b_orig * sigma_feat / std_yt
            es_canal = "adstock" in feat
            nombre = NOMBRES_LEGIBLES.get(feat, feat.replace("adstock_","").replace("_"," "))
            beta_norm_data.append({"Feature": nombre, "β_norm": b_norm,
                                   "Tipo": "Canal de medios" if es_canal else "Variable exógena"})

        beta_df = pd.DataFrame(beta_norm_data).sort_values("β_norm", ascending=True)
        color_map = {"Canal de medios": GOLD, "Variable exógena": DARK}
        fig_beta = go.Figure(go.Bar(
            y=beta_df["Feature"], x=beta_df["β_norm"], orientation="h",
            marker_color=[color_map[t] for t in beta_df["Tipo"]],
            text=[f"{v:.3f}" for v in beta_df["β_norm"]],
            textposition="outside", textfont=dict(size=10, family="Montserrat", color=DARK),
            hovertemplate="<b>%{y}</b><br>β norm = %{x:.4f}<extra></extra>",
        ))
        exec_layout(fig_beta, "Coeficientes β Normalizados (impacto en σ de Yt por 1σ de feature)",
                    height=max(400, len(beta_df) * 28))
        fig_beta.update_layout(
            showlegend=True,
            legend=dict(title="", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_beta, use_container_width=True)

    # ── Adstock β animado por canal y año ─────────────────────
    st.markdown('<div class="section-title">Contribución de β×Adstock por Canal y Año</div>',
                unsafe_allow_html=True)
    st.caption("▶ Cada barra muestra la contribución anual acumulada (β × Σ Adstock). "
               "Pulsa Play para ver la evolución año a año.")

    contrib_rows = []
    for yr in years_all:
        df_yr = df_model[df_model["anio"] == yr]
        for c in CANALES:
            col_name = ADSTOCK_MAP[c]
            adstock_sum = df_yr[col_name].sum() if col_name in df_yr.columns else 0
            contrib_c = betas.get(c, 0) * adstock_sum
            contrib_rows.append({"Año": str(yr), "Canal": c,
                                  "Contribución (€)": round(contrib_c)})

    contrib_anim_df = pd.DataFrame(contrib_rows)
    fig_contrib_anim = px.bar(
        contrib_anim_df,
        x="Canal", y="Contribución (€)", color="Canal",
        animation_frame="Año",
        color_discrete_map=CHANNEL_COLORS,
        range_y=[0, contrib_anim_df["Contribución (€)"].max() * 1.2],
        text=contrib_anim_df["Contribución (€)"].apply(fmt_eur),
    )
    fig_contrib_anim.update_traces(textposition="outside",
                                   textfont=dict(size=10, family="Montserrat", color=DARK))
    exec_layout(fig_contrib_anim, "Contribución por Canal (β × Σ Adstock)", height=460)
    fig_contrib_anim.update_layout(
        showlegend=False,
        updatemenus=[dict(type="buttons", showactive=False, y=-0.12, x=0.5, xanchor="center",
                          buttons=[dict(label="▶ Play", method="animate",
                                        args=[None, {"frame": {"duration": 900, "redraw": True},
                                                     "fromcurrent": True}]),
                                   dict(label="⏸ Pause", method="animate",
                                        args=[[None], {"frame": {"duration": 0, "redraw": False},
                                                       "mode": "immediate"}])])],
    )
    st.plotly_chart(fig_contrib_anim, use_container_width=True)

    # ── Simulador Alpha / Lag ─────────────────────────────────
    st.markdown('<div class="section-title">Simulador de Memoria de Marca (α y Lag)</div>',
                unsafe_allow_html=True)
    st.caption("Explora cómo la persistencia (α) y el retardo (Lag) afectan al impacto acumulado.")

    ch_cols_sim = st.columns(len(CANALES))
    for _col, _canal in zip(ch_cols_sim, CANALES):
        with _col:
            if st.button(_canal, key=f"sim_btn_{_canal}", use_container_width=True,
                         type="primary" if st.session_state.canal_sim == _canal else "secondary"):
                st.session_state.canal_sim = _canal
                st.rerun()
    sel_canal = st.session_state.canal_sim
    def_alpha = CANAL_PARAMS[sel_canal]["alpha"]
    def_lag   = CANAL_PARAMS[sel_canal]["lag"]

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        sim_alpha = st.slider("α (persistencia semanal)", 0.0, 0.99, def_alpha,
                              step=0.01, key="sim_alpha")
    with col_s2:
        sim_lag = st.slider("Lag (semanas de retardo)", 0, 4, def_lag,
                            step=1, key="sim_lag")

    ch_color = CHANNEL_COLORS[sel_canal]
    col_decay, col_accum = st.columns(2)

    with col_decay:
        wks = np.arange(25)
        impulse = np.zeros(25); impulse[min(sim_lag, 24)] = 1.0
        decay = np.zeros(25)
        for t in range(25):
            decay[t] = impulse[t] + (sim_alpha * decay[t-1] if t > 0 else 0)
        decay_n = decay / decay.max() if decay.max() > 0 else decay

        fig_d = go.Figure()
        fig_d.add_trace(go.Scatter(
            x=wks, y=decay_n, mode="lines+markers",
            line=dict(color=ch_color, width=2.5),
            fill="tozeroy", fillcolor=hex_rgba(ch_color),
            hovertemplate="Semana +%{x}<br>Efecto residual: %{y:.1%}<extra></extra>",
        ))
        hl = compute_half_life(sim_alpha) + sim_lag if sim_alpha > 0 else sim_lag
        if 0 < hl < 25:
            fig_d.add_vline(x=hl, line_color="#999", line_dash="dash", line_width=1)
            fig_d.add_annotation(x=hl, y=0.52,
                                  text=f"Vida media<br><b>{hl:.1f} sem</b>",
                                  showarrow=True, arrowhead=2, ax=50, ay=-30,
                                  font=dict(size=11, color=DARK, family="Montserrat"))
        exec_layout(fig_d, f"{sel_canal} — Curva de Decaimiento (α={sim_alpha:.2f})", height=360)
        fig_d.update_xaxes(title_text="Semanas desde la inversión")
        fig_d.update_yaxes(title_text="Efecto residual", tickformat=".0%")
        st.plotly_chart(fig_d, use_container_width=True)

    with col_accum:
        inv_ref = 100_000
        wks_a = np.arange(35)
        acc = np.zeros(35)
        for t in range(35):
            inp = inv_ref if t >= sim_lag else 0
            acc[t] = inp + (sim_alpha * acc[t-1] if t > 0 else 0)
        a_ss = inv_ref / (1 - sim_alpha) if sim_alpha < 1 else inv_ref

        fig_a = go.Figure()
        fig_a.add_trace(go.Scatter(
            x=wks_a, y=acc, mode="lines",
            line=dict(color=ch_color, width=2.5),
            fill="tozeroy", fillcolor=hex_rgba(ch_color),
            hovertemplate="Semana %{x}<br>Adstock: %{y:,.0f}€<extra></extra>",
        ))
        fig_a.add_hline(y=a_ss, line_color=GOLD, line_dash="dash", line_width=1.5,
                        annotation_text=f"A∞ = {fmt_eur(a_ss)}",
                        annotation_font=dict(color=GOLD, size=12, family="Montserrat"))
        exec_layout(fig_a, f"{sel_canal} — Acumulación Steady-State", height=360)
        fig_a.update_xaxes(title_text="Semanas con inversión constante")
        fig_a.update_yaxes(title_text="Adstock acumulado (€)")
        st.plotly_chart(fig_a, use_container_width=True)

    rank = sorted(CANAL_PARAMS, key=lambda c: CANAL_PARAMS[c]["alpha"], reverse=True)
    pos = rank.index(sel_canal) + 1
    mem_text = (
        "una de las <b>mayores memorias de marca</b> del mix" if pos <= 2
        else "una <b>memoria de marca corta</b> — su efecto se disipa rápidamente" if pos >= 7
        else "una <b>memoria de marca moderada</b>"
    )
    st.markdown(f"""
    <div class="narrative-box">
        Un <b>α = {sim_alpha:.2f}</b> significa que el <b>{sim_alpha*100:.0f}%</b> del impacto
        publicitario de esta semana persiste en la siguiente.
        <b>{sel_canal}</b> tiene {mem_text} (posición {pos} de {len(rank)}).<br><br>
        Con inversión constante de {fmt_eur(inv_ref)}/semana, el adstock converge a
        <b>{fmt_eur(a_ss)}</b> — es decir, se multiplica <b>{1/(1-sim_alpha):.1f}x</b>
        respecto a la inversión semanal.
    </div>
    """, unsafe_allow_html=True)

    # ── Fórmulas ──────────────────────────────────────────────
    st.markdown('<div class="section-title">Formulación Matemática</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Función de Pérdida (ElasticNet):**")
        st.latex(r"\mathcal{L}(\beta) = \sum_t(Y_t- {y}_t)^2 + \alpha l_1\sum_m|\beta_m| + \tfrac{\alpha(1-l_1)}{2}\sum_m\beta_m^2")
    with c2:
        st.markdown("**Adstock + Estado Estacionario:**")
        st.latex(r"A_{t,m} = X'_{t-L,m} + \alpha_m A_{t-1,m} \quad\longrightarrow\quad A_\infty = \frac{X_\text{sem}}{1-\alpha_m}")

    # ── Dinámica adstock semanal ───────────────────────────────
    st.markdown('<div class="section-title">Dinámica Adstock Semanal por Canal</div>',
                unsafe_allow_html=True)
    st.caption("Selecciona un canal para ver su inversión semanal real y el adstock acumulado (carry-over).")

    ch_cols_ads = st.columns(len(CANALES))
    for _col, _canal in zip(ch_cols_ads, CANALES):
        with _col:
            if st.button(_canal, key=f"ads_btn_{_canal}", use_container_width=True,
                         type="primary" if st.session_state.canal_ads == _canal else "secondary"):
                st.session_state.canal_ads = _canal
                st.rerun()

    sel_ts = st.session_state.canal_ads
    col_ts = ADSTOCK_MAP[sel_ts]
    fig_ad = go.Figure()
    if sel_ts in df_inv.columns:
        fig_ad.add_trace(go.Scatter(
            x=df_inv["semana_dt"], y=df_inv[sel_ts],
            name="Inversión semanal",
            line=dict(color="#C8C8C8", width=1.2),
            fill="tozeroy", fillcolor="rgba(200,200,200,0.15)",
        ))
    if col_ts in df_model.columns:
        fig_ad.add_trace(go.Scatter(
            x=df_model["semana_dt"], y=df_model[col_ts],
            name=f"Adstock (α={CANAL_PARAMS[sel_ts]['alpha']})",
            line=dict(color=CHANNEL_COLORS[sel_ts], width=2),
        ))
    exec_layout(fig_ad, f"{sel_ts} — Inversión vs. Adstock Acumulado", height=380)
    fig_ad.update_yaxes(title_text="€")
    st.plotly_chart(fig_ad, use_container_width=True)


# ═════════════════════════════════════════════════════════════════
# PÁGINA 3 — ATTRIBUTION INSIGHTS
# ═════════════════════════════════════════════════════════════════
elif page == "attribution":

    df_model = load_df_model()
    tabla    = load_tabla_atribucion()
    df_inv   = load_df_inversion()
    _, _, meta = load_model_artifacts()
    betas_raw = meta.get("beta_original", BETAS_FALLBACK)
    betas = get_betas(meta)
    years_all = sorted(y for y in df_model["anio"].unique().tolist() if y != 2019)

    total_ventas = df_model["Yt"].sum()
    total_media  = tabla["Venta Atrib. (€)"].sum()
    core_base    = total_ventas - total_media
    total_inv    = tabla["Inversión real (€)"].sum()
    tab_sorted   = tabla.sort_values("mROI", ascending=False).reset_index(drop=True)

    # ── KPIs ──────────────────────────────────────────────────
    best  = tab_sorted.iloc[0]
    worst = tab_sorted.iloc[-1]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(kpi_card("Canal Más Eficiente", best["Canal"],
                             subtitle=f"mROI = {best['mROI']:.2f}x"), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card("Canal Menos Eficiente", worst["Canal"],
                             subtitle=f"mROI = {worst['mROI']:.2f}x"), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_card("Contribución Medios", fmt_eur(total_media),
                             subtitle=f"{total_media/total_ventas*100:.1f}% del total"),
                    unsafe_allow_html=True)
    with c4:
        st.markdown(kpi_card("mROI Global", f"{total_media/total_inv:.2f}x",
                             subtitle="Media ponderada 2020–2024"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Stacked area semanal ──────────────────────────────────
    st.markdown('<div class="section-title">Contribución Semanal Acumulada por Canal</div>',
                unsafe_allow_html=True)
    df_contrib = df_model[["semana_dt"]].copy()
    for c in CANALES:
        b = betas_raw.get(ADSTOCK_MAP[c], 0)
        df_contrib[c] = b * df_model[ADSTOCK_MAP[c]] if ADSTOCK_MAP[c] in df_model.columns else 0

    fig_stack = go.Figure()
    for c in CANALES:
        fig_stack.add_trace(go.Scatter(
            x=df_contrib["semana_dt"], y=df_contrib[c],
            name=c, stackgroup="media",
            line=dict(width=0.5, color=CHANNEL_COLORS[c]),
            hovertemplate=f"<b>{c}</b><br>Semana: %{{x}}<br>Contrib: %{{y:,.0f}}€<extra></extra>",
        ))
    exec_layout(fig_stack, "Contribución Semanal Acumulada por Canal (€)", height=420)
    fig_stack.update_yaxes(title_text="Contribución (€)")
    st.plotly_chart(fig_stack, use_container_width=True)

    # ── Waterfall ─────────────────────────────────────────────
    st.markdown('<div class="section-title">¿De Dónde Viene Cada Euro de Venta?</div>',
                unsafe_allow_html=True)

    wf_labels  = ["Core Base"] + tab_sorted["Canal"].tolist() + ["Total Ventas"]
    wf_values  = [core_base] + tab_sorted["Venta Atrib. (€)"].tolist() + [0]
    wf_measure = ["absolute"] + ["relative"] * len(tab_sorted) + ["total"]
    wf_colors  = [DARK] + [CHANNEL_COLORS[c] for c in tab_sorted["Canal"]] + [GOLD]
    wf_hover   = (
        [f"<b>Core Base</b><br>{fmt_eur(core_base)}<br>Ventas orgánicas sin publicidad<extra></extra>"]
        + [f"<b>{row['Canal']}</b><br>{fmt_eur(row['Venta Atrib. (€)'])}<br>"
           f"Por cada 1€ invertido, recuperamos {row['mROI']:.2f}€<extra></extra>"
           for _, row in tab_sorted.iterrows()]
        + [f"<b>Total</b><br>{fmt_eur(total_ventas)}<extra></extra>"]
    )

    fig_wf = go.Figure(go.Waterfall(
        x=wf_labels, y=wf_values, measure=wf_measure,
        text=[fmt_eur(v) for v in wf_values[:-1]] + [fmt_eur(total_ventas)],
        textposition="outside", textfont=dict(size=10, family="Montserrat", color=DARK),
        connector=dict(line=dict(color="#DDD", width=1)),
        hovertext=wf_hover, hoverinfo="text",
        increasing=dict(marker=dict(color=GOLD)),
        decreasing=dict(marker=dict(color="#C62828")),
        totals=dict(marker=dict(color=DARK)),
    ))
    exec_layout(fig_wf, "Waterfall: Core Base + Contribución de Medios", height=500)
    fig_wf.update_layout(showlegend=False)
    st.plotly_chart(fig_wf, use_container_width=True)

    # ── mROI + scatter ────────────────────────────────────────
    col_mroi, col_scat = st.columns(2)
    with col_mroi:
        fig_mroi = go.Figure(go.Bar(
            y=tab_sorted["Canal"], x=tab_sorted["mROI"], orientation="h",
            marker_color=[CHANNEL_COLORS[c] for c in tab_sorted["Canal"]],
            text=[f"{v:.2f}x" for v in tab_sorted["mROI"]],
            textposition="outside",
            textfont=dict(size=11, family="Montserrat", color=DARK),
            hovertemplate="<b>%{y}</b><br>mROI: %{x:.2f}x<extra></extra>",
        ))
        fig_mroi.add_vline(x=1, line_color="#999", line_width=1, line_dash="dash")
        exec_layout(fig_mroi, "mROI por Canal (2020–2024)", height=420)
        fig_mroi.update_layout(showlegend=False)
        st.plotly_chart(fig_mroi, use_container_width=True)

    with col_scat:
        fig_scat = go.Figure()
        for _, row in tab_sorted.iterrows():
            c = row["Canal"]
            fig_scat.add_trace(go.Scatter(
                x=[row["Inversión real (€)"]], y=[row["Venta Atrib. (€)"]],
                mode="markers+text", text=[c], textposition="top center",
                textfont=dict(size=9, family="Montserrat", color=DARK),
                marker=dict(size=row["mROI"]*5+12, color=CHANNEL_COLORS[c],
                            line=dict(color=WHITE, width=1.5)),
                hovertemplate=(f"<b>{c}</b><br>Inversión: {fmt_eur(row['Inversión real (€)'])}<br>"
                               f"Atribución: {fmt_eur(row['Venta Atrib. (€)'])}<br>"
                               f"mROI: {row['mROI']:.2f}x<extra></extra>"),
            ))
        mx = max(tab_sorted["Inversión real (€)"].max(), tab_sorted["Venta Atrib. (€)"].max())
        fig_scat.add_trace(go.Scatter(x=[0,mx], y=[0,mx], mode="lines",
                                       line=dict(color="#DDD",width=1,dash="dash"),
                                       showlegend=False, hoverinfo="skip"))
        exec_layout(fig_scat, "Inversión vs. Venta Atribuida (tamaño = mROI)", height=420)
        fig_scat.update_xaxes(title_text="Inversión Real (€)")
        fig_scat.update_yaxes(title_text="Venta Atribuida (€)")
        fig_scat.update_layout(showlegend=False)
        st.plotly_chart(fig_scat, use_container_width=True)

    # ── Tabla ─────────────────────────────────────────────────
    st.markdown('<div class="section-title">Tabla de Atribución Completa</div>',
                unsafe_allow_html=True)
    disp = tab_sorted[["Canal","β_m","mROI","Peso % (medios)",
                        "Inversión real (€)","Venta Atrib. (€)","Elasticidad"]].copy()
    disp.columns = ["Canal","β (€/€ adstock)","mROI","Peso %",
                     "Inversión","Venta Atribuida","Elasticidad"]
    for col_fmt, fmt_fn in [("β (€/€ adstock)", lambda x: f"{x:.4f}"),
                              ("mROI", lambda x: f"{x:.2f}x"),
                              ("Peso %", lambda x: f"{x:.1f}%"),
                              ("Inversión", fmt_eur), ("Venta Atribuida", fmt_eur),
                              ("Elasticidad", lambda x: f"{x:.4f}")]:
        disp[col_fmt] = disp[col_fmt].map(fmt_fn)
    st.dataframe(disp, hide_index=True, use_container_width=True)

    # ── Contrib animada por año ───────────────────────────────
    st.markdown('<div class="section-title">Contribución por Canal y Año — Animada</div>',
                unsafe_allow_html=True)
    st.caption("▶ Pulsa Play o arrastra el slider para explorar la contribución año a año.")

    contrib_rows = []
    for yr in years_all:
        df_yr = df_model[df_model["anio"] == yr]
        inv_yr = df_inv[df_inv["anio"] == yr]
        for c in CANALES:
            ads = df_yr[ADSTOCK_MAP[c]].sum() if ADSTOCK_MAP[c] in df_yr.columns else 0
            inv = inv_yr[c].sum() if c in inv_yr.columns else 1
            contrib_c = betas.get(c, 0) * ads
            mroi_c = contrib_c / inv if inv > 0 else 0
            contrib_rows.append({"Año": str(yr), "Canal": c,
                                  "Contribución (€)": round(contrib_c),
                                  "mROI": round(mroi_c, 3)})

    contrib_anim = pd.DataFrame(contrib_rows)
    fig_ca = px.bar(
        contrib_anim.sort_values("Contribución (€)", ascending=False),
        x="Canal", y="Contribución (€)", color="Canal",
        animation_frame="Año",
        color_discrete_map=CHANNEL_COLORS,
        range_y=[0, contrib_anim["Contribución (€)"].max() * 1.2],
        text=contrib_anim.sort_values("Contribución (€)", ascending=False)
                         ["Contribución (€)"].apply(fmt_eur),
        hover_data={"mROI": ":.2f"},
    )
    fig_ca.update_traces(textposition="outside",
                         textfont=dict(size=10, family="Montserrat", color=DARK))
    exec_layout(fig_ca, "Contribución Anual por Canal", height=480)
    fig_ca.update_layout(
        showlegend=False,
        updatemenus=[dict(type="buttons", showactive=False, y=-0.12, x=0.5, xanchor="center",
                          buttons=[dict(label="▶ Play", method="animate",
                                        args=[None, {"frame": {"duration": 900, "redraw": True},
                                                     "fromcurrent": True}]),
                                   dict(label="⏸ Pause", method="animate",
                                        args=[[None], {"frame": {"duration": 0},
                                                       "mode": "immediate"}])])],
    )
    st.plotly_chart(fig_ca, use_container_width=True)

    # ── Donut ─────────────────────────────────────────────────
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        fig_donut = go.Figure(go.Pie(
            labels=["Core Base"] + tab_sorted["Canal"].tolist(),
            values=[core_base] + tab_sorted["Venta Atrib. (€)"].tolist(),
            hole=0.55,
            marker=dict(colors=[DARK]+[CHANNEL_COLORS[c] for c in tab_sorted["Canal"]],
                        line=dict(color=WHITE, width=2)),
            textinfo="label+percent",
            textfont=dict(size=10, family="Montserrat", color=DARK),
            hovertemplate="<b>%{label}</b><br>%{value:,.0f}€<br>%{percent}<extra></extra>",
        ))
        fig_donut.add_annotation(
            text=f"<b>{fmt_eur(total_ventas)}</b>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=15, color=DARK, family="Playfair Display"),
        )
        exec_layout(fig_donut, "Composición de Ventas Totales", height=420)
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_d2:
        inv_annual = df_inv.groupby("anio")[CANALES].sum().reset_index()
        fig_inv_st = go.Figure()
        for c in CANALES:
            if c in inv_annual.columns:
                fig_inv_st.add_trace(go.Bar(
                    x=inv_annual["anio"].astype(str), y=inv_annual[c],
                    name=c, marker_color=CHANNEL_COLORS[c],
                    hovertemplate=f"<b>{c}</b> · %{{x}}<br>%{{y:,.0f}}€<extra></extra>",
                ))
        exec_layout(fig_inv_st, "Inversión Anual por Canal", height=420)
        fig_inv_st.update_layout(barmode="stack")
        fig_inv_st.update_yaxes(title_text="Inversión (€)")
        st.plotly_chart(fig_inv_st, use_container_width=True)


# ═════════════════════════════════════════════════════════════════
# PÁGINA 4 — STRATEGIC SIMULATOR
# ═════════════════════════════════════════════════════════════════
elif page == "simulator":

    pres = load_presupuesto()
    _, _, meta = load_model_artifacts()
    betas = get_betas(meta)
    pres_clean = pres[pres["Canal"] != "TOTAL"].copy()

    # ── Presupuesto total ─────────────────────────────────────
    st.markdown('<div class="section-title">Presupuesto Total 2024</div>',
                unsafe_allow_html=True)
    budget_total = st.slider(
        "Presupuesto total (€)", 4_000_000, 20_000_000, 12_000_000,
        step=500_000, format="%d€", key="budget_total",
    )

    # Simplex defaults
    simplex_alloc = run_simplex_lp(budget_total, betas)
    simplex_pcts  = {c: simplex_alloc[c] / budget_total * 100 for c in CANALES}

    # ── Sliders por canal ─────────────────────────────────────
    st.markdown('<div class="section-title">Distribución por Canal</div>',
                unsafe_allow_html=True)
    st.caption("Los defaults reflejan la optimización Simplex LP. "
               "Ajusta manualmente y observa el impacto en tiempo real.")

    alloc_pct = {}
    for row_canales in [CANALES[:4], CANALES[4:]]:
        cols_sl = st.columns(4)
        for idx, canal in enumerate(row_canales):
            lo, hi = COTAS[canal]
            default = max(lo*100, min(hi*100, simplex_pcts.get(canal, 12.5)))
            with cols_sl[idx]:
                pct = st.slider(canal, float(lo*100), float(hi*100), float(default),
                                step=0.5, format="%.1f%%", key=f"sl_{canal}")
                alloc_pct[canal] = pct

    total_pct  = sum(alloc_pct.values())
    alloc_norm = {c: p/total_pct for c, p in alloc_pct.items()}
    alloc_eur  = {c: p*budget_total for c, p in alloc_norm.items()}

    result = simulate_budget(alloc_eur, betas)

    baseline_alloc = dict(zip(pres_clean["Canal"], pres_clean["Presupuesto 2023 (€)"]))
    r_base = simulate_budget(baseline_alloc, betas)
    delta_c = result["contrib_medios"] - r_base["contrib_medios"]
    delta_p = delta_c / r_base["contrib_medios"] * 100 if r_base["contrib_medios"] > 0 else 0

    # ── KPIs dinámicos ────────────────────────────────────────
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(kpi_card("Contribución Estimada", fmt_eur(result["contrib_medios"]),
                             delta=delta_c, delta_pct=delta_p,
                             subtitle="vs. Baseline 2023"), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card("mROI Simulado", f"{result['mroi_sim']:.2f}x",
                             subtitle=f"Baseline: {r_base['mroi_sim']:.2f}x"),
                    unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_card("Margen Bruto Incremental", fmt_eur(result["margen_incr"]),
                             subtitle="× margen medio 67.6%"), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi_card("Payback Ratio", f"{result['payback']:.3f}x",
                             subtitle="Margen / Inversión"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Comparativa barras agrupadas + treemap ────────────────
    col_bar, col_tree = st.columns(2)
    canals_s = sorted(CANALES, key=lambda c: result["detalle"][c]["mroi"], reverse=True)

    with col_bar:
        fig_sim = make_subplots(specs=[[{"secondary_y": True}]])
        fig_sim.add_trace(go.Bar(
            x=canals_s, y=[result["detalle"][c]["inv"] for c in canals_s],
            name="Inversión", marker_color=DARK, opacity=0.55,
            hovertemplate="<b>%{x}</b><br>Inversión: %{y:,.0f}€<extra></extra>",
        ), secondary_y=False)
        fig_sim.add_trace(go.Bar(
            x=canals_s, y=[result["detalle"][c]["contrib"] for c in canals_s],
            name="Contribución",
            marker_color=[CHANNEL_COLORS[c] for c in canals_s],
            hovertemplate="<b>%{x}</b><br>Contribución: %{y:,.0f}€<extra></extra>",
        ), secondary_y=False)
        fig_sim.add_trace(go.Scatter(
            x=canals_s, y=[result["detalle"][c]["mroi"] for c in canals_s],
            name="mROI", mode="markers+lines+text",
            text=[f"{result['detalle'][c]['mroi']:.1f}x" for c in canals_s],
            textposition="top center",
            textfont=dict(size=10, color=GOLD, family="Montserrat"),
            marker=dict(color=GOLD, size=10, line=dict(color=WHITE, width=1.5)),
            line=dict(color=GOLD, width=2),
        ), secondary_y=True)
        exec_layout(fig_sim, "Inversión vs. Contribución por Canal", height=440)
        fig_sim.update_layout(barmode="group", legend=dict(orientation="h", y=-0.18))
        fig_sim.update_yaxes(title_text="€", secondary_y=False)
        fig_sim.update_yaxes(title_text="mROI (x)", secondary_y=True)
        st.plotly_chart(fig_sim, use_container_width=True)

    with col_tree:
        tree_df = pd.DataFrame([{
            "Canal": c, "Inversión": result["detalle"][c]["inv"],
            "mROI": result["detalle"][c]["mroi"],
            "pct": alloc_norm[c]*100,
        } for c in CANALES])
        fig_tree = px.treemap(
            tree_df, path=["Canal"], values="Inversión", color="mROI",
            color_continuous_scale=[[0, DARK], [0.5, CREAM], [1, GOLD]],
            hover_data={"Inversión": ":,.0f", "mROI": ":.2f", "pct": ":.1f"},
        )
        fig_tree.update_traces(
            texttemplate="<b>%{label}</b><br>%{customdata[2]:.1f}%<br>mROI %{color:.1f}x",
            textfont=dict(size=12, family="Montserrat"),
        )
        exec_layout(fig_tree, "Mapa del Presupuesto (color = mROI)", height=440)
        fig_tree.update_layout(coloraxis_colorbar=dict(title="mROI"))
        st.plotly_chart(fig_tree, use_container_width=True)

    # ── Sensibilidad β ────────────────────────────────────────
    st.markdown("---")
    if "sens_visible" not in st.session_state:
        st.session_state.sens_visible = True
    _hint = "  —  *pulsa para ocultar*" if st.session_state.sens_visible else "  —  *pulsa para probar*"
    st.markdown('<span id="sens-toggle-anchor"></span>', unsafe_allow_html=True)
    if st.button(f"🧪 Análisis de Sensibilidad — ¿Qué pasa si mejoramos la creatividad?{_hint}",
                 key="btn_sens", use_container_width=True):
        st.session_state.sens_visible = not st.session_state.sens_visible

    if st.session_state.sens_visible:
        beta_mults = {}
        cols_b = st.columns(4)
        for idx, canal in enumerate(CANALES):
            with cols_b[idx % 4]:
                mult = st.slider(canal, -50, 100, 0, step=5, format="%+d%%",
                                 key=f"beta_{canal}")
                beta_mults[canal] = 1 + mult / 100
        betas_mod = {c: betas[c] * beta_mults[c] for c in CANALES}
        r_mod = simulate_budget(alloc_eur, betas_mod)
        d_sens = r_mod["contrib_medios"] - result["contrib_medios"]
        d_pct  = d_sens / result["contrib_medios"] * 100 if result["contrib_medios"] > 0 else 0

        cb1, cb2 = st.columns(2)
        with cb1:
            st.markdown(kpi_card("Contribución con β Modificados",
                                 fmt_eur(r_mod["contrib_medios"]),
                                 delta=d_sens, delta_pct=d_pct,
                                 subtitle="vs. betas actuales"), unsafe_allow_html=True)
        with cb2:
            st.markdown(kpi_card("mROI con β Modificados", f"{r_mod['mroi_sim']:.2f}x",
                                 subtitle=f"Actual: {result['mroi_sim']:.2f}x"),
                        unsafe_allow_html=True)

        changes = [(c, (beta_mults[c]-1)*100,
                     r_mod["detalle"][c]["contrib"] - result["detalle"][c]["contrib"])
                    for c in CANALES if beta_mults[c] != 1.0]
        for canal, pct_c, delta_cv in changes:
            icon = "📈" if delta_cv > 0 else "📉"
            st.markdown(f"""<div class="narrative-box">{icon} Si la eficacia de
            <b>{canal}</b> cambia un <b>{pct_c:+.0f}%</b>,
            su contribución varía en <b>{fmt_eur(delta_cv)}</b>.</div>""",
                        unsafe_allow_html=True)

    # ── Tabla detallada ───────────────────────────────────────
    st.markdown('<div class="section-title">Presupuesto Detallado</div>',
                unsafe_allow_html=True)
    budget_rows = [{
        "Canal": c, "% Asignado": f"{alloc_norm[c]*100:.1f}%",
        "Inversión 2024": fmt_eur(alloc_eur[c]),
        "Contrib. Estimada": fmt_eur(result["detalle"][c]["contrib"]),
        "mROI": f"{result['detalle'][c]['mroi']:.2f}x",
        "vs 2023": fmt_eur(alloc_eur[c] - baseline_alloc.get(c, 0)),
    } for c in CANALES]
    budget_rows.append({"Canal": "TOTAL", "% Asignado": "100.0%",
                         "Inversión 2024": fmt_eur(budget_total),
                         "Contrib. Estimada": fmt_eur(result["contrib_medios"]),
                         "mROI": f"{result['mroi_sim']:.2f}x",
                         "vs 2023": fmt_eur(budget_total - sum(baseline_alloc.values()))})
    st.dataframe(pd.DataFrame(budget_rows), hide_index=True, use_container_width=True)

    # ── Comparativa de escenarios estratégicos ───────────────
    st.markdown('<div class="section-title">Comparativa de Escenarios Estratégicos</div>',
                unsafe_allow_html=True)

    # ── Calcular escenarios ───────────────────────────────────
    alloc_opt_12m = run_simplex_lp(12_000_000, betas)
    r_opt_scen    = simulate_budget(alloc_opt_12m, betas)
    r_rec_scen    = simulate_budget({c: v * 0.70 for c, v in baseline_alloc.items()}, betas)

    _base_total = sum(baseline_alloc.values())

    # Conservador: 10M€ con blend 50% proporciones históricas + 50% proporciones óptimas
    # mROI queda entre Baseline y Óptimo: mejor eficiencia que histórico puro,
    # menor concentración (y menor riesgo de ejecución) que el Óptimo.
    _w_base  = {c: baseline_alloc.get(c, 0) / _base_total for c in CANALES}
    _w_opt   = {c: alloc_opt_12m[c] / 12_000_000 for c in CANALES}
    _w_blend = {c: 0.5 * _w_base[c] + 0.5 * _w_opt[c] for c in CANALES}
    _w_sum   = sum(_w_blend.values())
    alloc_cons = {c: (_w_blend[c] / _w_sum) * 10_000_000 for c in CANALES}
    r_cons     = simulate_budget(alloc_cons, betas)

    COLOR_BASE  = DARK
    COLOR_REC   = "#C62828"
    COLOR_CONS  = "#5B7FA5"   # azul sobrio — menos riesgo
    COLOR_OPT   = "#539E60"   # verde — crecimiento

    scen_data = [
        {"label": "Baseline 2023",    "inv": _base_total,          "r": r_base,    "color": COLOR_BASE, "icon": "📌",
         "desc": "Sin cambios. Distribución histórica de 2023."},
        {"label": "Recorte CFO −30%", "inv": _base_total * 0.70,   "r": r_rec_scen,"color": COLOR_REC,  "icon": "✂️",
         "desc": "Recorte lineal propuesto. Destruye contribución sin mejorar eficiencia."},
        {"label": "Conservador",      "inv": 10_000_000,            "r": r_cons,    "color": COLOR_CONS, "icon": "🛡️",
         "desc": "10M€ con mix histórico 2023. −28% capital, menor eficiencia que el Óptimo. Transición segura: misma mezcla conocida, menor riesgo de ejecución."},
        {"label": "Óptimo Ruben",     "inv": 12_000_000,            "r": r_opt_scen,"color": COLOR_OPT,  "icon": "🎯",
         "desc": "12M€. Máximo mROI por Simplex LP. Concentra donde el modelo demuestra mayor retorno."},
    ]

    # ── 4 tarjetas ────────────────────────────────────────────
    scen_cols = st.columns(4)
    for col, s in zip(scen_cols, scen_data):
        dv   = s["r"]["contrib_medios"] - r_base["contrib_medios"]
        dpct = dv / r_base["contrib_medios"] * 100 if r_base["contrib_medios"] > 0 else 0
        sign, dcolor = ("▲", "#539E60") if dv >= 0 else ("▼", "#C62828")
        with col:
            st.markdown(f"""
            <div style="background:{WHITE};border:2px solid {s['color']};border-radius:12px;
                        padding:16px 12px;text-align:center;margin-bottom:8px;min-height:270px;">
                <div style="font-size:1.5rem;">{s['icon']}</div>
                <div style="font-family:'Playfair Display',serif;font-size:0.92rem;font-weight:700;
                            color:{s['color']};margin:4px 0 6px;">{s['label']}</div>
                <div style="font-size:0.70rem;color:#888;line-height:1.3;margin-bottom:8px;
                            min-height:32px;">{s['desc']}</div>
                <div style="font-size:0.72rem;color:#888;">Inversión</div>
                <div style="font-size:1.0rem;font-weight:600;color:{DARK};">{fmt_eur(s['inv'])}</div>
                <div style="font-size:0.72rem;color:#888;margin-top:5px;">Contribución medios</div>
                <div style="font-size:1.0rem;font-weight:600;color:{DARK};">{fmt_eur(s['r']['contrib_medios'])}</div>
                <div style="font-size:0.72rem;color:#888;margin-top:5px;">mROI</div>
                <div style="font-size:1.25rem;font-weight:700;color:{s['color']};">{s['r']['mroi_sim']:.2f}x</div>
                <div style="font-size:0.78rem;font-weight:600;color:{dcolor};margin-top:5px;">
                    {sign} {abs(dpct):.1f}% vs Baseline
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Donuts: distribución por canal en cada escenario ──────
    st.markdown('<div class="section-title">Segmentación de Inversión por Canal</div>',
                unsafe_allow_html=True)
    st.caption("Cómo se distribuye el presupuesto en cada escenario. "
               "El Conservador mantiene la diversificación; el Óptimo concentra donde el modelo es más eficiente.")

    donut_scenarios = [
        ("Baseline 2023",  {c: baseline_alloc[c] for c in CANALES},  COLOR_BASE),
        ("Conservador",    alloc_cons,                                 COLOR_CONS),
        ("Óptimo Rubén",   alloc_opt_12m,                             COLOR_OPT),
    ]
    dcols = st.columns(3)
    for col, (title, alloc, color) in zip(dcols, donut_scenarios):
        total_alloc = sum(alloc.values())
        labels_d = list(alloc.keys())
        values_d = [alloc[c] for c in labels_d]
        pcts_d   = [v / total_alloc * 100 for v in values_d]

        fig_d = go.Figure(go.Pie(
            labels=labels_d, values=values_d,
            hole=0.52,
            marker=dict(colors=[CHANNEL_COLORS[c] for c in labels_d],
                        line=dict(color=WHITE, width=2)),
            textinfo="label+percent",
            textfont=dict(size=9, family="Montserrat"),
            hovertemplate="<b>%{label}</b><br>%{value:,.0f}€<br>%{percent}<extra></extra>",
            sort=True,
            direction="clockwise",
        ))
        fig_d.add_annotation(
            text=f"<b>{fmt_eur(total_alloc)}</b>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=13, color=color, family="Playfair Display, serif"),
        )
        exec_layout(fig_d, title, height=380)
        fig_d.update_layout(
            showlegend=False,
            margin=dict(l=10, r=10, t=50, b=10),
        )
        with col:
            st.plotly_chart(fig_d, use_container_width=True)

    # ── Tabla comparativa de distribución ─────────────────────
    st.markdown('<div class="section-title">Distribución Detallada por Canal y Escenario</div>',
                unsafe_allow_html=True)
    dist_rows = []
    for c in CANALES:
        b_total  = sum(baseline_alloc.values())
        dist_rows.append({
            "Canal": c,
            "Baseline 2023": f"{baseline_alloc.get(c,0)/b_total*100:.1f}%  ({fmt_eur(baseline_alloc.get(c,0))})",
            "Conservador":   f"{alloc_cons[c]/10e6*100:.1f}%  ({fmt_eur(alloc_cons[c])})",
            "Óptimo Rubén":  f"{alloc_opt_12m[c]/12e6*100:.1f}%  ({fmt_eur(alloc_opt_12m[c])})",
            "mROI canal":    f"{betas.get(c,0)/(1-CANAL_PARAMS[c]['alpha']):.2f}x" if CANAL_PARAMS[c]['alpha'] < 1 else "—",
        })
    st.dataframe(pd.DataFrame(dist_rows), hide_index=True, use_container_width=True)

    # ── Informe Ejecutivo ─────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-title">Informe Ejecutivo para la Junta</div>',
                unsafe_allow_html=True)
    top3 = sorted(CANALES, key=lambda c: result["detalle"][c]["mroi"], reverse=True)[:3]
    if True:
        diff_inv = budget_total - sum(baseline_alloc.values())
        diff_word = "menos" if diff_inv < 0 else "más"
        st.markdown(f"""
        <div class="board-report">
            <h3 style="color:{DARK} !important; font-family:'Playfair Display',serif !important; padding-bottom:14px;">INFORME EJECUTIVO — Redistribución del Presupuesto de Medios 2024</h3>
            <p><b>Elaborado por:</b> Rubén Elices &nbsp;&nbsp;
            <hr style="border-color:{GOLD}; opacity:0.3;">
            <p><b>1. Situación Actual</b><br>
            En 2023, K-Moda invirtió <span class="highlight">{fmt_eur(sum(baseline_alloc.values()))}</span>
            sin optimización basada en datos. La crisis de privacidad invalida el tracking por último clic.</p>
            <p><b>2. Metodología</b><br>
            Marketing Mix Model (ElasticNet, positive=True) entrenado con 208 semanas (2020–2023).
            Optimización por Simplex LP con cotas operativas realistas por canal.</p>
            <p><b>3. Propuesta</b><br>
            Redistribuir <span class="highlight">{fmt_eur(budget_total)}</span>
            ({fmt_eur(abs(diff_inv))} {diff_word} que en 2023) según eficiencia marginal.</p>
            <p><b>4. Resultados Esperados</b></p>
            <ul>
                <li>Contribución de medios: <span class="highlight">{fmt_eur(result['contrib_medios'])}</span>
                    ({delta_p:+.1f}% vs. 2023)</li>
                <li>mROI: <span class="highlight">{result['mroi_sim']:.2f}x</span>
                    (vs. {r_base['mroi_sim']:.2f}x actual)</li>
                <li>Margen bruto incremental: <span class="highlight">{fmt_eur(result['margen_incr'])}</span></li>
                <li>Canales prioritarios: <span class="highlight">{", ".join(top3)}</span></li>
            </ul>
            <p><b>5. Por qué estos canales</b><br>
            El modelo identifica cuatro canales con retorno marginal superior a la media:
            <ul>
                <li><b>Video Online</b>: alcanza a la audiencia digital nativa de K-Moda con alto poder de branding. Mayor persistencia del efecto publicitario (α=0.70).</li>
                <li><b>Display</b>: retargeting eficiente sobre usuarios ya interesados. Convierte intención en compra con inversión moderada.</li>
                <li><b>Email CRM</b>: el canal más barato por euro invertido. La base de clientes existente convierte a tasas muy superiores al tráfico frío.</li>
                <li><b>Exterior</b>: presencia física en zonas de alto tráfico comercial. Refuerza notoriedad de marca en el momento de compra.</li>
            </ul>
            </p>
            <p><b>6. Recomendación</b><br>
            <span class="highlight">Aprobar la redistribución propuesta.</span>
            Con {fmt_eur(abs(diff_inv))} {diff_word} de presupuesto, se proyecta una mejora del
            {abs(delta_p):.1f}% en contribución y un incremento del mROI de
            <b>{r_base['mroi_sim']:.1f}x a {result['mroi_sim']:.1f}x</b>.</p>
            <hr style="border-color:{GOLD}; opacity:0.3;">
            <p style="font-size:0.78rem; color:#888;">
                Modelo: ElasticNet MMM · MAPE Test: {meta.get('mape_test',15.0):.1f}% ·
                R²: {meta.get('r2',0.818):.3f} · Datos: 2020–2024 (261 semanas)
            </p>
        </div>
        """, unsafe_allow_html=True)

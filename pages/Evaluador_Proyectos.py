# appturbinas.py
# =========================================================
# Evaluador de Rendimiento - Turbinas Eólicas (MVP) · versión UI PRO
# =========================================================

import io
import base64
import re
import unicodedata as _ud
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import gamma

# ============================================================
# Tabla de referencia nacional de parámetros Weibull (Chile)
# Fuente: Atlas eólico Chile + literatura técnica (rango orientativo)
# ============================================================
REGIONES_WEIBULL = {
    "Arica y Parinacota":    {"zona": "Norte",         "k_rng": (2.3, 3.0), "c_rng": (6.0, 9.0)},
    "Tarapacá":               {"zona": "Norte",         "k_rng": (2.2, 3.0), "c_rng": (6.5, 9.5)},
    "Antofagasta":            {"zona": "Norte",         "k_rng": (2.0, 2.8), "c_rng": (7.0, 10.0)},
    "Atacama":                {"zona": "Centro-Norte",  "k_rng": (1.8, 2.5), "c_rng": (6.5, 9.0)},
    "Coquimbo":               {"zona": "Centro-Norte",  "k_rng": (1.9, 2.7), "c_rng": (6.0, 8.5)},
    "Valparaíso":             {"zona": "Centro",        "k_rng": (1.8, 2.6), "c_rng": (5.5, 8.0)},
    "Metropolitana":          {"zona": "Centro",        "k_rng": (1.6, 2.4), "c_rng": (5.0, 7.0)},
    "O’Higgins":              {"zona": "Centro-Sur",    "k_rng": (1.7, 2.4), "c_rng": (4.5, 6.5)},
    "Maule":                  {"zona": "Centro-Sur",    "k_rng": (1.8, 2.5), "c_rng": (4.5, 6.5)},
    "Ñuble":                  {"zona": "Centro-Sur",    "k_rng": (1.8, 2.6), "c_rng": (5.0, 7.0)},
    "Biobío":                 {"zona": "Centro-Sur",    "k_rng": (2.0, 2.8), "c_rng": (6.0, 8.0)},
    "La Araucanía":           {"zona": "Sur",           "k_rng": (2.1, 2.9), "c_rng": (6.5, 8.5)},
    "Los Ríos":               {"zona": "Sur",           "k_rng": (2.2, 3.0), "c_rng": (6.0, 8.0)},
    "Los Lagos":              {"zona": "Sur",           "k_rng": (2.0, 2.8), "c_rng": (6.0, 8.5)},
    "Aysén":                  {"zona": "Sur Extremo",   "k_rng": (2.1, 3.0), "c_rng": (7.0, 9.0)},
    "Magallanes":             {"zona": "Sur Extremo",   "k_rng": (2.5, 3.2), "c_rng": (8.0, 11.0)}
}


# ---------------------------
# Configuración visual global
# ---------------------------
st.set_page_config(page_title="Evaluador de Rendimiento Eólico", layout="wide")

# Paleta y template (inspirada en ERNC: verdes/teales)
PRIMARY = "#0E9F6E"   # verde
SECOND  = "#0EA5E9"   # celeste
ACCENT  = "#22C55E"   # lima suave
GRAY_1  = "#111827"
GRAY_2  = "#6B7280"
GRID    = "rgba(148,163,184,.25)"

px.defaults.color_discrete_sequence = [PRIMARY, SECOND, ACCENT, "#10B981", "#06B6D4", "#84CC16"]


def apply_plotly_theme(fig: go.Figure,
                       title: str | None = None,
                       x_title: str | None = None,
                       y_title: str | None = None,
                       height: int = 340) -> go.Figure:
    """Aplica formato visual profesional consistente a cualquier figura Plotly."""
    fig.update_layout(
        title={"text": title or (getattr(fig.layout.title, 'text', "") or ""),
               "x": 0.01, "xanchor": "left",
               "font": {"size": 18, "color": GRAY_1}},
        font={"family": "Inter, Segoe UI, system-ui, -apple-system, Roboto, Helvetica",
              "size": 13, "color": GRAY_1},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02,
                "xanchor": "left", "x": 0.01, "title": ""},
        margin=dict(l=12, r=12, t=48, b=12),
        height=height,
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    fig.update_xaxes(title_text=x_title or (fig.layout.xaxis.title.text if fig.layout.xaxis and fig.layout.xaxis.title else ""),
                     showgrid=True, gridcolor=GRID, zeroline=False,
                     linecolor="rgba(0,0,0,.2)", ticks="outside", automargin=True)
    fig.update_yaxes(title_text=y_title or (fig.layout.yaxis.title.text if fig.layout.yaxis and fig.layout.yaxis.title else ""),
                     showgrid=True, gridcolor=GRID, zeroline=False,
                     linecolor="rgba(0,0,0,.2)", ticks="outside", automargin=True)
    for tr in fig.data:
        if isinstance(tr, (go.Scatter, go.Scattergl)):
            tr.update(line=dict(width=2))
        if isinstance(tr, go.Bar):
            tr.update(marker_line_color="rgba(0,0,0,0)")
    return fig


# ---------------------------
# Plantillas descargables (ahora en kW)
# ---------------------------
TEMPLATE_POWER_CURVE = """v_m_s,P_KW
0.0,0
1.0,0
2.0,0.005
2.5,0.015
3.0,0.030
4.0,0.060
5.0,0.090
6.0,0.120
7.0,0.160
8.0,0.185
9.0,0.195
10.0,0.200
11.0,0.200
12.0,0.200
15.0,0.200
20.0,0
"""

TEMPLATE_WIND_SERIES = """timestamp,v_mean_m_s,v_max_m_s,v_std_m_s,direction_deg,air_temp_C,pressure_hPa
2025-01-01 00:00:00,4.5,7.2,0.8,210,16.2,1013
2025-01-01 00:10:00,4.3,6.9,0.7,215,16.0,1012
2025-01-01 00:20:00,4.8,7.4,0.9,220,15.9,1011
"""

# URL por defecto para la serie de viento (Google Sheets publicado como CSV)
WIND_SERIES_URL_DEFAULT = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSI-Mg4NQsfXChe2BZeqQ2ysAnJ_AXp1R2nbLAGAKG3B54nBo5fNiC9c6uFfrxwdmlR5LFvDVEqZO08/pub?gid=351976858&single=true&output=csv"

# URL por defecto para la curva de potencia (Google Sheets publicado como CSV)
PC_CSV_URL_DEFAULT = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSoy1swXqDpUDbSglCims4oyCyZA8kNpEwoJ4i_DWl3kT2cIUljRCzYGJdPwrqw6n5hdp7V3K1-bF_I/pub?gid=0&single=true&output=csv"


def download_button_from_text(label: str, text: str, file_name: str) -> None:
    b64 = base64.b64encode(text.encode()).decode()
    href = f'<a style="text-decoration:none;font-weight:600" href="data:file/txt;base64,{b64}" download="{file_name}">{label}</a>'
    st.markdown(href, unsafe_allow_html=True)


# ---------------------------
# Funciones de modelo y utilidades
# ---------------------------

def _norm_text(s: str) -> str:
    s = str(s).replace("\xa0", " ").strip()
    s = " ".join(s.split())
    s = "".join(ch for ch in _ud.normalize("NFKD", s) if not _ud.combining(ch))
    return s.lower()


def _guess_col(df: pd.DataFrame, keys: list[str], regex: list[str] | None = None) -> str | None:
    regex = regex or []
    for c in df.columns:
        n = _norm_text(c)
        if any(k in n for k in keys):
            return c
        if any(re.search(rgx, n) for rgx in regex):
            return c
    return None


def _to_num(s):
    """Convierte strings con %, puntos de miles y comas decimales a float."""
    if pd.isna(s):
        return np.nan
    s = str(s)
    s = s.replace("%", "").strip()
    s = re.sub(r"[^0-9,.\-]", "", s)
    if "." in s and "," in s:
        s = s.replace(".", "").replace(",", ".")
    else:
        if s.count(".") >= 1 and re.search(r"\.\d{3}(\.|$)", s):
            s = s.replace(".", "")
        s = s.replace(",", ".")
    try:
        return float(s)
    except Exception:
        return np.nan


def read_power_curve(file_or_url) -> pd.DataFrame:
    """
    Lee curva de potencia desde archivo o URL (CSV).
    Columnas aceptadas:
      - v_m_s  (obligatoria)
      - P_KW  (potencia en kW)  -> también se expone P_W = P_KW * 1000 para cálculos internos
      - P_W   (potencia en W)   -> si viene en W, se calcula P_KW = P_W / 1000
    Soporta comas decimales.
    """
    df = pd.read_csv(file_or_url, dtype=str)
    df = df.rename(columns={c: c.strip() for c in df.columns})

    # normaliza decimales
    for c in df.columns:
        df[c] = df[c].astype(str).str.replace(",", ".", regex=False)

    if "v_m_s" not in df.columns:
        raise ValueError("El CSV de curva debe contener la columna 'v_m_s'.")

    has_kw = "P_KW" in df.columns
    has_w  = "P_W"  in df.columns
    if not (has_kw or has_w):
        raise ValueError("El CSV de curva debe contener 'P_KW' (kW) o 'P_W' (W).")

    df["v_m_s"] = pd.to_numeric(df["v_m_s"], errors="coerce")

    if has_kw:
        df["P_KW"] = pd.to_numeric(df["P_KW"], errors="coerce")
        df["P_W"]  = df["P_KW"] * 1000.0
        df.attrs["pc_units_source"] = "kW"
    else:
        df["P_W"]  = pd.to_numeric(df["P_W"], errors="coerce")
        df["P_KW"] = df["P_W"] / 1000.0
        df.attrs["pc_units_source"] = "W"

    df = df.dropna(subset=["v_m_s", "P_W", "P_KW"]).copy()
    df = df.sort_values("v_m_s").reset_index(drop=True)
    df["v_m_s"] = df["v_m_s"].clip(lower=0)
    df["P_W"]   = df["P_W"].clip(lower=0)
    df["P_KW"]  = df["P_KW"].clip(lower=0)
    return df[["v_m_s", "P_KW", "P_W"]]


def read_wind_series(file_or_url) -> pd.DataFrame:
    """
    Lee series de viento desde archivo o URL (Google Sheets CSV).
    Acepta 'timestamp', 'fecha', 'date', 'datetime' como columna de tiempo.
    Convierte comas decimales a puntos y limpia símbolos.
    """
    try:
        df = pd.read_csv(file_or_url, dtype=str)
    except Exception as e:
        raise ValueError(f"No se pudo leer el CSV: {e}")

    # Nombres limpios
    df.columns = [str(c).strip() for c in df.columns]

    # Detectar columna tiempo
    ts_col = None
    for c in df.columns:
        if any(k in c.lower() for k in ["timestamp", "fecha", "date", "datetime"]):
            ts_col = c
            break
    if not ts_col:
        raise ValueError("No se encontró columna de tiempo en el CSV (timestamp/fecha/date/datetime).")

    # Parseo fecha/hora
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col]).rename(columns={ts_col: "timestamp"})

    # Conversión numérica robusta
    for c in df.columns:
        if c != "timestamp":
            df[c] = (
                df[c].astype(str)
                    .str.replace(",", ".", regex=False)
                    .str.replace(r"[^0-9.\-]", "", regex=True)
            )
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "v_mean_m_s" not in df.columns:
        raise ValueError("El CSV debe contener la columna 'v_mean_m_s'.")

    df = df[df["v_mean_m_s"] >= 0]
    return df.sort_values("timestamp").reset_index(drop=True)


def extrapolate_to_hub_height(v_z, z_sensor, H_hub, alpha=0.14):
    if z_sensor <= 0 or H_hub <= 0:
        return v_z
    factor = (H_hub / z_sensor) ** alpha
    return v_z * factor


def air_density(temp_C=None, pressure_hPa=None):
    if temp_C is None or pressure_hPa is None:
        return 1.225
    T_K = temp_C + 273.15
    P_Pa = pressure_hPa * 100.0
    R = 287.05
    return float(P_Pa / (R * T_K))


def equivalent_wind_speed(v, rho, rho_ref=1.225):
    rho = np.asarray(rho, dtype=float)
    rho = np.where(rho <= 0, np.nan, rho)
    return v * (rho / rho_ref) ** (1.0 / 3.0)


def interpolate_power(v_array, pc_df):
    v_curve = pc_df["v_m_s"].values
    p_curve = pc_df["P_W"].values   # interpolamos en W por precisión
    v_min = v_curve.min()
    p = np.interp(v_array, v_curve, p_curve, left=p_curve[0], right=p_curve[-1])
    p = np.where(v_array <= v_min, p_curve[0], p)
    return p  # W


def apply_losses(power_W, losses_dict):
    p = power_W.astype(float)
    for _, pct in (losses_dict or {}).items():
        if pct is None:
            continue
        p *= (1.0 - float(pct) / 100.0)
    return p


def compute_energy_from_series(
    df, pc_df, z_sensor, H_hub, alpha,
    inverter_eff=0.95,
    default_temp_C=15.0, default_press_hPa=1013.0,
    losses=None
):
    df = df.copy()

    # Paso temporal (h)
    if len(df) >= 2:
        dt_seconds = (df["timestamp"].iloc[1] - df["timestamp"].iloc[0]).total_seconds()
    else:
        dt_seconds = 600
    dt_h = dt_seconds / 3600.0

    # Extrapolación a altura de buje
    v_hub = extrapolate_to_hub_height(df["v_mean_m_s"].values, z_sensor, H_hub, alpha)

    # Densidad del aire
    if "air_temp_C" in df.columns and "pressure_hPa" in df.columns:
        rho = np.array([air_density(t, p) for t, p in zip(df["air_temp_C"], df["pressure_hPa"])])
    else:
        rho = np.full_like(v_hub, air_density(default_temp_C, default_press_hPa), dtype=float)

    # Velocidad equivalente por densidad y potencia
    v_eq   = equivalent_wind_speed(v_hub, rho)
    P_W    = interpolate_power(v_eq, pc_df)                 # W
    inv_eff = float(inverter_eff) if 0 < inverter_eff <= 1.0 else 1.0
    P_ac_W = P_W * inv_eff                                   # W
    P_net_W = apply_losses(P_ac_W, losses)                   # W

    # Energías/potencias en kW y kWh
    P_ac_kW  = P_ac_W  / 1000.0
    P_net_kW = P_net_W / 1000.0
    E_kWh    = P_net_kW * dt_h

    out = pd.DataFrame({
        "timestamp": df["timestamp"],
        "v_mean_hub_m_s": v_hub,
        "rho_kg_m3": rho,
        "v_eq_m_s": v_eq,
        "P_ac_kW": P_ac_kW,
        "P_net_kW": P_net_kW,
        "E_kWh": E_kWh
    })

    # KPIs base
    total_energy_kWh = out["E_kWh"].sum()
    P_rated_W = float(pc_df["P_W"].max())
    hours = len(out) * dt_h
    capacity_factor = ((total_energy_kWh * 1000.0) / (P_rated_W * hours)) if (P_rated_W > 0 and hours > 0) else 0.0

    # KPIs extra (TI y ráfaga relativa G)
    kpis_extra = {}

    # Intensidad de turbulencia TI = std/mean
    if "v_std_m_s" in df.columns:
        with np.errstate(divide='ignore', invalid='ignore'):
            TI = df["v_std_m_s"] / df["v_mean_m_s"].replace(0, np.nan)
        TI_np = TI.to_numpy(dtype=float)
        if np.isfinite(TI_np).any():
            kpis_extra["TI_p50"] = float(np.nanmedian(TI_np))
            kpis_extra["TI_p90"] = float(np.nanpercentile(TI_np, 90))
        else:
            kpis_extra["TI_p50"] = np.nan
            kpis_extra["TI_p90"] = np.nan

    # Ráfaga relativa G = vmax/mean
    if "v_max_m_s" in df.columns:
        with np.errstate(divide='ignore', invalid='ignore'):
            G = df["v_max_m_s"] / df["v_mean_m_s"].replace(0, np.nan)
        G_np = G.to_numpy(dtype=float)
        if np.isfinite(G_np).any():
            kpis_extra["G_p50"] = float(np.nanmedian(G_np))
            kpis_extra["G_p90"] = float(np.nanpercentile(G_np, 90))  # FIX: q=90
        else:
            kpis_extra["G_p50"] = np.nan
            kpis_extra["G_p90"] = np.nan

    return out, {
        "AEP_kWh": total_energy_kWh * (8760.0 / hours) if hours > 0 else 0.0,
        "Energy_period_kWh": total_energy_kWh,
        "CapacityFactor_pct": capacity_factor * 100,
        **kpis_extra
    }



def weibull_pdf(v, k, c):
    v = np.asarray(v, dtype=float)
    k = float(k); c = float(c)
    pdf = (k / c) * (v / c) ** (k - 1) * np.exp(-(v / c) ** k)
    pdf[v < 0] = 0.0
    return pdf


try:
    _trapz = np.trapezoid
except AttributeError:  # numpy < 1.20
    _trapz = np.trapz


def aep_from_weibull(pc_df, k, c, hours_year=8760, inverter_eff=0.95, losses=None, rho=1.225, rho_ref=1.225):
    v = np.linspace(0, max(30.0, pc_df["v_m_s"].max()), 1201)
    v_eq = equivalent_wind_speed(v, rho, rho_ref)
    P_W = interpolate_power(v_eq, pc_df)             # W
    inv_eff = float(inverter_eff) if 0 < inverter_eff <= 1 else 1.0
    P_ac_W = P_W * inv_eff
    P_net_W = apply_losses(P_ac_W, losses)           # W
    P_net_kW = P_net_W / 1000.0                      # kW

    f = weibull_pdf(v, k, c)
    P_exp_kW = _trapz(P_net_kW * f, v)               # kW (esperado)
    AEP_kWh = P_exp_kW * hours_year                  # kWh
    P_rated_W = float(pc_df["P_W"].max())
    CF = ((P_exp_kW * 1000.0) / P_rated_W) if P_rated_W > 0 else 0.0
    return AEP_kWh, CF * 100.0, (v, f, v_eq, P_net_kW)


# ---------------------------
# Sidebar - Entradas (con expanders)
# ---------------------------
st.sidebar.title("Parámetros de entrada")

# Modo
modo = st.sidebar.radio("Modo de evaluación", ["Series (CSV)", "Distribución (Weibull)"], index=0)

# Curva de potencia (si quieres también plegarla)
with st.sidebar.expander("Curva de Potencia (CSV)", expanded=True):
    pc_url = st.text_input(
        "URL curva de potencia (Google Sheets CSV)",
        value=PC_CSV_URL_DEFAULT,
        help="Encabezados: v_m_s y P_KW (kW) o P_W (W). Si viene P_W, se convierte a P_KW automáticamente."
    )
    pc_file = st.file_uploader("o sube CSV local (opcional)", type=["csv"], key="pc")

    # 👇 en vez de otro expander, deja el botón directo o un container
    st.caption("Plantilla de curva de potencia (CSV):")
    download_button_from_text("📥 Descargar plantilla (power_curve.csv)", TEMPLATE_POWER_CURVE, "power_curve.csv")


# Turbina
with st.sidebar.expander("Turbina", expanded=False):
    inverter_eff = st.slider("Eficiencia inversor (η)", 0.80, 1.00, 0.95, 0.01)
    H_hub = st.number_input("Altura de buje H (m)", min_value=1.0, value=17.0, step=0.5)
    V_rated_hint = st.text_input("Referencia: V_rated (m/s) (opcional)", "")

# Sitio / Atmósfera
with st.sidebar.expander("Sitio / Atmósfera", expanded=False):
    z_sensor = st.number_input("Altura del sensor z (m)", min_value=0.1, value=13.0, step=0.5)
    alpha = st.slider("Exponente de cizalle α", 0.00, 0.50, 0.14, 0.01)
    temp_default = st.number_input("Temp media (°C, si no hay serie)", value=15.0, step=0.5)
    press_default = st.number_input("Presión media (hPa, si no hay serie)", value=1013.0, step=0.5)

# Pérdidas
with st.sidebar.expander("Pérdidas (%)", expanded=False):
    losses = {
        "availability_pct": st.slider("Indisponibilidad (Availability)", 0, 30, 5, 1),
        "electrical_pct":   st.slider("Pérdidas eléctricas", 0, 20, 3, 1),
        "curtailment_pct":  st.slider("Curtailment (limitaciones)", 0, 30, 0, 1),
        "soiling_icing_pct":st.slider("Hielo/Suciedad (soiling)", 0, 20, 2, 1),
        "wakes_pct":        st.slider("Estela (wakes, si aplica)", 0, 20, 0, 1),
    }

# Propuesta técnica
with st.sidebar.expander("Propuesta técnica (objetivos)", expanded=False):
    consumo_diario_kWh_pt = st.number_input(
        "Consumo diario objetivo (kWh/día)",
        min_value=0.0, value=36000.0, step=100.0
    )
    consumo_anual_MWh_pt = st.number_input(
        "Consumo anual estimado (MWh/año) (opcional)",
        min_value=0.0, value=0.0, step=10.0,
        help="Si queda en 0, se calcula como Consumo diario × 365 / 1000."
    )
    tarifa_actual_val = st.number_input(
        "Tarifa eléctrica actual ($/kWh)",
        min_value=0.0, value=120.0, step=10.0
    )
    tarifa_turbinas_val = st.number_input(          # 👇 NUEVO CAMPO AQUÍ
        "Tarifa turbinas ($/kWh)",
        min_value=0.0, value=100.0, step=10.0,
        help="Costo nivelado o precio objetivo de generación eólica."
    )
    margen_seguridad_pct = st.slider(
        "Margen de seguridad (%)", 0, 100, 15, 1
    )
    energia_riesgo_MWp = st.number_input(
        "Energía adicional de riesgo (MWp) (opcional)",
        min_value=0.0, value=0.0, step=0.1
    )


# Entradas específicas por modo
if modo == "Series (CSV)":
    with st.sidebar.expander("Series de viento (CSV)", expanded=True):
       wind_url = st.text_input(
           "URL CSV (Google Sheets publicado)",
           value=WIND_SERIES_URL_DEFAULT,
           help="CSV con columnas: timestamp,v_mean_m_s,(v_max_m_s),(v_std_m_s),(direction_deg),(air_temp_C),(pressure_hPa)"
      )

       wind_file = st.file_uploader("o sube CSV local (opcional)", type=["csv"], key="wind_csv")

       # 👇 Botón de descarga directo (sin expander interno)
       st.caption("Plantilla de series de viento (CSV):")
       download_button_from_text(
           "📥 Descargar plantilla (wind_series.csv)",
           TEMPLATE_WIND_SERIES,
           "wind_series.csv"
        )

    k_weibull = c_weibull = None

else:
    wind_file = None
    wind_url = ""
    with st.sidebar.expander("Weibull", expanded=True):
        k_weibull = st.number_input("k (forma)", min_value=0.5, value=2.0, step=0.1)
        c_weibull = st.number_input("c (escala, m/s)", min_value=0.1, value=6.0, step=0.1)

# (Opcional) Referencia nacional (Weibull) también plegable
with st.sidebar.expander("Referencia nacional Weibull (Chile)", expanded=False):
    macro_zonas = ["Norte","Centro-Norte","Centro","Centro-Sur","Sur","Sur Extremo"]

    # Claves únicas para evitar StreamlitDuplicateElementId
    zona_sel = st.selectbox(
        "Zona geográfica de referencia",
        macro_zonas, index=0, key="ref_weibull_zona"
    )

    regiones_filtradas = [r for r, info in REGIONES_WEIBULL.items() if info["zona"] == zona_sel]
    region_sel = st.selectbox(
        "Región de referencia",
        sorted(regiones_filtradas), key="ref_weibull_region"
    )

    info = REGIONES_WEIBULL[region_sel]
    k_sug = (float(info["k_rng"][0]) + float(info["k_rng"][1])) / 2.0
    c_sug = (float(info["c_rng"][0]) + float(info["c_rng"][1])) / 2.0

    try:
        v_media = c_sug * gamma(1 + 1.0 / k_sug)
    except Exception:
        v_media = float("nan")

    def _pexceed(v0, k, c):
        try:
            return np.exp(- (v0 / c) ** k)
        except Exception:
            return np.nan

    p_gt_5 = _pexceed(5, k_sug, c_sug)
    p_gt_8 = _pexceed(8, k_sug, c_sug)

    st.info(
        f"**{region_sel}** · zona **{zona_sel}**\n\n"
        f"- k sugerido: **{k_sug:.2f}** (rango {info['k_rng'][0]:.1f}–{info['k_rng'][1]:.1f})\n"
        f"- c sugerido: **{c_sug:.2f} m/s** (rango {info['c_rng'][0]:.1f}–{info['c_rng'][1]:.1f})\n"
        f"- v̄ estimada: **{v_media:.2f} m/s**\n"
        f"- P(V>5): **{(p_gt_5*100):.1f}%** · P(V>8): **{(p_gt_8*100):.1f}%**"
    )

    if st.button("Usar k,c sugeridos", key="ref_weibull_btn"):
        # Si estás en modo Weibull, puedes guardar en session_state para reutilizar
        st.session_state["k_weibull"] = float(k_sug)
        st.session_state["c_weibull"] = float(c_sug)


# ============================
# Sugerencias k,c por zona/REGIÓN (Chile)
# ============================

REGIONES_WEIBULL = {
    "Arica y Parinacota":  {"zona":"Norte","k_rng":(2.4,3.0),"c_rng":(7.0,9.0)},
    "Tarapacá":            {"zona":"Norte","k_rng":(2.5,3.0),"c_rng":(7.5,9.5)},
    "Antofagasta":         {"zona":"Norte","k_rng":(2.6,3.1),"c_rng":(7.8,9.8)},
    "Atacama":             {"zona":"Norte","k_rng":(2.3,2.8),"c_rng":(6.8,8.8)},
    "Coquimbo":            {"zona":"Centro-Norte","k_rng":(2.2,2.7),"c_rng":(5.8,7.2)},
    "Valparaíso":          {"zona":"Centro","k_rng":(2.1,2.6),"c_rng":(5.5,6.8)},
    "Metropolitana":       {"zona":"Centro","k_rng":(2.0,2.5),"c_rng":(5.0,6.2)},
    "O'Higgins":           {"zona":"Centro","k_rng":(2.0,2.5),"c_rng":(5.0,6.2)},
    "Maule":               {"zona":"Centro","k_rng":(1.9,2.4),"c_rng":(4.8,6.0)},
    "Ñuble":               {"zona":"Centro-Sur","k_rng":(1.9,2.3),"c_rng":(5.2,6.6)},
    "Biobío":              {"zona":"Centro-Sur","k_rng":(1.9,2.3),"c_rng":(5.5,6.8)},
    "La Araucanía":        {"zona":"Sur","k_rng":(1.8,2.2),"c_rng":(6.0,7.6)},
    "Los Ríos":            {"zona":"Sur","k_rng":(1.8,2.2),"c_rng":(6.2,7.8)},
    "Los Lagos":           {"zona":"Sur","k_rng":(1.8,2.2),"c_rng":(6.5,8.5)},
    "Aysén":               {"zona":"Sur","k_rng":(1.7,2.1),"c_rng":(7.2,9.0)},
    "Magallanes":          {"zona":"Sur Extremo","k_rng":(1.6,2.0),"c_rng":(9.0,11.0)},
}

def _mid(a,b): return (float(a)+float(b))/2.0


with st.expander("📋 Tabla de referencia nacional Weibull (k,c) — Chile", expanded=False):
    rows = []
    for reg,inf in REGIONES_WEIBULL.items():
        k_mid = _mid(*inf["k_rng"]); c_mid = _mid(*inf["c_rng"])
        try:
            vmean = c_mid * gamma(1 + 1.0/k_mid)
        except Exception:
            vmean = np.nan
        rows.append({
            "Zona": inf["zona"],
            "Región": reg,
            "k (min–max)": f"{inf['k_rng'][0]:.1f}–{inf['k_rng'][1]:.1f}",
            "c (m/s, min–max)": f"{inf['c_rng'][0]:.1f}–{inf['c_rng'][1]:.1f}",
            "k sugerido": round(k_mid,2),
            "c sugerido (m/s)": round(c_mid,2),
            "v_media estimada (m/s)": round(vmean,2)
        })
    df_ref = pd.DataFrame(rows).sort_values(["Zona","Región"])
    st.dataframe(df_ref, use_container_width=True, hide_index=True)
    st.caption("Valores referenciales; ajústalos con mediciones locales si están disponibles.")

# ---------------------------
# Encabezado
# ---------------------------
st.title("Evaluador de Rendimiento de Turbinas Eólicas")
st.markdown(
    """
**Objetivo.** Estimar AEP, factor de planta y cobertura de una demanda fija usando una **curva de potencia (kW)**
 y **datos de viento** (series o Weibull), con ajuste por **altura**, **densidad** y **pérdidas**.
"""
)

# ---------------------------
# Cargar curva (prioriza archivo; si no hay, URL; si falla, plantilla)
# ---------------------------
if pc_file is not None:
    try:
        pc_df = read_power_curve(pc_file)
        st.success("Curva de potencia cargada desde archivo local.")
    except Exception as e:
        st.error(f"Error al leer curva de potencia desde archivo: {e}")
        st.stop()
elif pc_url.strip():
    try:
        pc_df = read_power_curve(pc_url.strip())
        st.success("Curva de potencia cargada desde URL (Google Sheets CSV).")
    except Exception as e:
        st.error(f"Error al leer curva de potencia desde URL: {e}")
        st.stop()
else:
    st.info("🔧 No hay curva de potencia. Uso una curva de ejemplo incluida (kW).")
    pc_df = read_power_curve(io.StringIO(TEMPLATE_POWER_CURVE))

with st.expander("Curva de potencia cargada", expanded=True):
    st.write(f"Puntos: **{len(pc_df)}** · P_rated: **{pc_df['P_KW'].max():,.2f} kW**")
    st.caption("Nota: si subiste `P_W`, se convirtió a `P_KW = P_W ÷ 1000`.")
    st.dataframe(pc_df[["v_m_s", "P_KW"]], use_container_width=True)

    _buf_pc = io.StringIO(); pc_df[["v_m_s", "P_KW"]].to_csv(_buf_pc, index=False)
    st.download_button("📥 Descargar curva normalizada (CSV)", _buf_pc.getvalue(),
                       file_name="power_curve_kw.csv", mime="text/csv")

    fig_pc = px.line(pc_df, x="v_m_s", y="P_KW", markers=True)
    try:
        v_r = float(V_rated_hint) if V_rated_hint.strip() else None
    except Exception:
        v_r = None
    if v_r:
        fig_pc.add_vline(x=v_r, line_dash="dot", line_color=SECOND)
        fig_pc.add_annotation(x=v_r, y=pc_df["P_KW"].max(), yshift=10, showarrow=False,
                              text=f"V_rated ≈ {v_r:g} m/s", font=dict(color=SECOND))
    fig_pc = apply_plotly_theme(fig_pc, "Curva de Potencia (ρ = 1.225 kg/m³)", "v [m/s]", "P [kW]", height=360)
    st.plotly_chart(fig_pc, use_container_width=True)

# ---------------------------
# Cálculo por modo
# ---------------------------
col_kpis_1, col_kpis_2, col_kpis_3, col_kpis_4 = st.columns(4)
results_df = None
monthly_df = None
wind_df = None          # evita NameError en modo Weibull

AEP_kWh = 0.0
CF_pct = 0.0
mean_daily_kWh = 0.0
coverage_pct = None
extra_metrics = {}

if modo == "Series (CSV)":
    # ----- SERIE CSV -----
    src_label = None
    if wind_file is not None:
        try:
            wind_df = read_wind_series(wind_file)
            src_label = "archivo"
        except Exception as e:
            st.error(f"Error al leer series de viento desde archivo: {e}")
            st.stop()
    elif wind_url.strip():
        try:
            wind_df = read_wind_series(wind_url.strip())
            src_label = "URL"
        except Exception as e:
            st.error(f"Error al leer series de viento desde URL: {e}")
            st.stop()
    else:
        st.warning("Proporciona una **URL** de Google Sheets publicada como CSV o sube un **archivo CSV** para continuar.")
        st.stop()

    st.success(f"Serie de viento cargada desde {src_label}.")

    results_df, kpis = compute_energy_from_series(
        wind_df, pc_df, z_sensor=z_sensor, H_hub=H_hub, alpha=alpha,
        inverter_eff=inverter_eff, default_temp_C=temp_default, default_press_hPa=press_default,
        losses=losses
    )

    # KPIs
    AEP_kWh = kpis["AEP_kWh"]
    CF_pct = kpis["CapacityFactor_pct"]
    extra_metrics = {k: v for k, v in kpis.items() if k not in ["AEP_kWh", "Energy_period_kWh", "CapacityFactor_pct"]}

    # Energía diaria media
    if len(results_df) > 0:
        daily_energy = results_df.set_index("timestamp")["E_kWh"].resample("D").sum()
        mean_daily_kWh = float(daily_energy.mean() if len(daily_energy) else 0.0)

    # Cobertura vs. consumo objetivo
    if consumo_diario_kWh_pt > 0:
        coverage_pct = (mean_daily_kWh / consumo_diario_kWh_pt) * 100.0

    # Mostrar KPIs
    col_kpis_1.metric("AEP (P50) [kWh/año]", f"{AEP_kWh:,.0f}")
    col_kpis_2.metric("Factor de Planta [%]", f"{CF_pct:,.1f}")
    col_kpis_3.metric("Energía diaria media [kWh/d]", f"{mean_daily_kWh:,.2f}")
    col_kpis_4.metric(
        "Cobertura demanda [%]" if consumo_diario_kWh_pt > 0 else "Consumo diario (kWh/d)",
        f"{coverage_pct:,.1f}%" if coverage_pct is not None else f"{consumo_diario_kWh_pt:,.0f}"
    )

else:
    # ----- DISTRIBUCIÓN (WEIBULL) -----
    rho_site = air_density(temp_default, press_default)
    AEP_kWh, CF_pct, (v, f, v_eq, P_net_KW) = aep_from_weibull(
        pc_df, k_weibull, c_weibull,
        hours_year=8760,
        inverter_eff=inverter_eff,
        losses=losses,
        rho=rho_site,
        rho_ref=1.225
    )

    # DataFrame utilizable por plots/tablas en modo Weibull
    results_df = pd.DataFrame({
        "v": v,                      # velocidad (m/s)
        "pdf": f,                    # distribución
        "v_eq_m_s": v_eq,            # velocidad equivalente por densidad
        "P_net_kW": P_net_KW,        # potencia neta esperada por bin
        "E_kWh_bin": P_net_KW * 8760.0 * f  # energía anual esperada por bin
    })

    # KPIs y cobertura
    mean_daily_kWh = AEP_kWh / 365.0
    if consumo_diario_kWh_pt > 0:
        coverage_pct = (mean_daily_kWh / consumo_diario_kWh_pt) * 100.0

    col_kpis_1.metric("AEP (P50) [kWh/año]", f"{AEP_kWh:,.0f}")
    col_kpis_2.metric("Factor de Planta [%]", f"{CF_pct:,.1f}")
    col_kpis_3.metric("Energía diaria media [kWh/d]", f"{mean_daily_kWh:,.2f}")
    col_kpis_4.metric(
        "Cobertura demanda [%]" if consumo_diario_kWh_pt > 0 else "Consumo diario (kWh/d)",
        f"{coverage_pct:,.1f}%" if coverage_pct is not None else f"{consumo_diario_kWh_pt:,.0f}"
    )

# ==============================================================
# Construcción segura de _results (válido para Series y Weibull)
# ==============================================================

# Asegura que existan las referencias
if "results_df" not in locals() or results_df is None:
    results_df = pd.DataFrame()
if "wind_df" not in locals():
    wind_df = None  # evita NameError en modo Weibull

# 1) Normaliza tipo
if isinstance(results_df, dict):
    results_df = pd.DataFrame(results_df)
elif not isinstance(results_df, pd.DataFrame):
    results_df = pd.DataFrame()

# 2) Si está vacío → crea esqueleto
if results_df.empty:
    _results = pd.DataFrame(columns=[
        "timestamp", "v", "bin", "v_eq_m_s",
        "v_mean_hub_m_s", "P_net_kW", "E_kWh", "E_kWh_bin", "pdf"
    ])
else:
    # 3) Orden estable según columnas disponibles
    if "timestamp" in results_df.columns:
        sort_key = "timestamp"
    elif "v" in results_df.columns:
        sort_key = "v"
    elif "bin" in results_df.columns:
        sort_key = "bin"
    else:
        results_df = results_df.assign(_idx=np.arange(len(results_df)))
        sort_key = "_idx"

    _results = (
        results_df
        .sort_values(sort_key, kind="stable")
        .reset_index(drop=True)
        .copy()
    )

# 4) Garantiza potencia en kW si venía en W
if "P_net_kW" not in _results.columns and "P_net_W" in _results.columns:
    _results["P_net_kW"] = _results["P_net_W"] / 1000.0

# -------- Gráficos (Series y Weibull) --------
gcol1, gcol2 = st.columns(2)

# =========================
# 1) ENERGÍA — SOLO Series
# =========================
if modo == "Series (CSV)" and not _results.empty and {"timestamp", "E_kWh"}.issubset(_results.columns):
    _daily = (
        _results.set_index("timestamp")["E_kWh"]
        .resample("D").sum()
        .to_frame("E_kWh_d")
        .reset_index()
    )

    trend_series = None
    trend_name = ""
    try:
        import statsmodels.api as _sm  # noqa
        if len(_daily) >= 15:
            low = _sm.nonparametric.lowess(
                endog=_daily["E_kWh_d"].values,
                exog=_daily["timestamp"].astype("int64") // 10**9,
                frac=0.2, return_sorted=False
            )
            trend_series = pd.Series(low, index=_daily.index)
            trend_name = "Tendencia (LOWESS)"
    except Exception:
        trend_series = None

    if trend_series is None:
        trend_series = _daily["E_kWh_d"].rolling(window=7, min_periods=3, center=True).mean()
        trend_name = "Media móvil (7d)"

    fig_e = go.Figure()
    fig_e.add_trace(go.Bar(x=_daily["timestamp"], y=_daily["E_kWh_d"], name="kWh/día", marker_line_width=0, opacity=0.85))
    fig_e.add_trace(go.Scatter(x=_daily["timestamp"], y=trend_series, name=trend_name, mode="lines"))

    mu = float(_daily["E_kWh_d"].mean()) if len(_daily) else 0.0
    p95 = float(_daily["E_kWh_d"].quantile(0.95)) if len(_daily) else 0.0
    fig_e.add_hline(y=mu, line_dash="dot", line_color=SECOND)
    fig_e.add_hline(y=p95, line_dash="dash", line_color=ACCENT)
    fig_e.add_annotation(xref="paper", x=0.01, y=mu, yref="y", text=f"μ ≈ {mu:,.1f} kWh/d", showarrow=False, font=dict(color=SECOND))
    fig_e.add_annotation(xref="paper", x=0.01, y=p95, yref="y", text=f"p95 ≈ {p95:,.1f} kWh/d", showarrow=False, font=dict(color=ACCENT))

    fig_e = apply_plotly_theme(fig_e, "Energía neta diaria (con tendencia)", "Tiempo", "kWh/día", height=360)
    gcol1.plotly_chart(fig_e, use_container_width=True)

# ============================================
# 2) POTENCIA — SOLO Series
# ============================================
if modo == "Series (CSV)" and not _results.empty and {"v_eq_m_s", "P_net_kW"}.issubset(_results.columns):
    fig_pv = go.Figure()
    fig_pv.add_trace(go.Scattergl(x=_results["v_eq_m_s"], y=_results["P_net_kW"], mode="markers",
                                  name="Medido (neto)", opacity=0.35, marker=dict(size=5)))

    # Tendencia
    trend = None
    try:
        import statsmodels.api as _sm  # noqa
        if len(_results) >= 50:
            low = _sm.nonparametric.lowess(
                endog=_results["P_net_kW"].values,
                exog=_results["v_eq_m_s"].values,
                frac=0.15, return_sorted=True
            )
            trend = pd.DataFrame(low, columns=["v", "p"])
    except Exception:
        trend = None

    if trend is None:
        _tmp = _results[["v_eq_m_s", "P_net_kW"]].dropna().sort_values("v_eq_m_s")
        _tmp["p_roll"] = _tmp["P_net_kW"].rolling(window=50, min_periods=15, center=True).median()
        trend = _tmp.rename(columns={"v_eq_m_s": "v", "p_roll": "p"})[["v", "p"]].dropna()

    fig_pv.add_trace(go.Scatter(x=trend["v"], y=trend["p"], mode="lines", name="Tendencia (mediana móvil)"))

    # Curva nominal
    y_nom_kw = pc_df["P_KW"] if "P_KW" in pc_df.columns else (pc_df["P_W"] / 1000.0)
    fig_pv.add_trace(go.Scatter(x=pc_df["v_m_s"], y=y_nom_kw, mode="lines+markers", name="Curva nominal P(v)",
                                line=dict(width=2, color="rgba(100,116,139,.9)"), marker=dict(size=6)))
    P_rated = float(y_nom_kw.max())
    fig_pv.add_hline(y=P_rated, line_dash="dot", line_color=GRAY_2)
    fig_pv.add_annotation(xref="paper", x=0.99, y=P_rated, yref="y", text=f"P_rated ≈ {P_rated:,.2f} kW",
                          showarrow=False, font=dict(color=GRAY_2))

    fig_pv = apply_plotly_theme(fig_pv, "Potencia neta vs V_eq (con nominal)", "V_eq [m/s]", "P [kW]", height=360)
    gcol2.plotly_chart(fig_pv, use_container_width=True)

# 👉 En modo Weibull NO pintamos nada aquí; todos los plots van dentro del expander "Análisis avanzado".


# =========================
# SERIES — Gráficos extra
# =========================
if modo == "Series (CSV)" and not _results.empty:

    s1, s2 = st.columns(2)

    # 1) Histograma de V_eq (complemento al de v_hub)
    if "v_eq_m_s" in _results.columns and _results["v_eq_m_s"].notna().any():
        v_eq = _results["v_eq_m_s"].dropna()
        p50 = float(v_eq.quantile(0.50))
        p90 = float(v_eq.quantile(0.90))

        fig_hv = px.histogram(v_eq, nbins=40)
        fig_hv.add_vline(x=p50, line_dash="dash", line_color=SECOND)
        fig_hv.add_vline(x=p90, line_dash="dot",  line_color=ACCENT)
        fig_hv.add_annotation(x=p50, yref="paper", y=1.06, xref="x", text=f"p50 ≈ {p50:.2f} m/s", showarrow=False, font=dict(color=SECOND))
        fig_hv.add_annotation(x=p90, yref="paper", y=1.06, xref="x", text=f"p90 ≈ {p90:.2f} m/s", showarrow=False, font=dict(color=ACCENT))
        fig_hv = apply_plotly_theme(fig_hv, "Histograma de V_eq (densidad ajustada)", "V_eq [m/s]", "frecuencia", height=320)
        s1.plotly_chart(fig_hv, use_container_width=True)

    # 2) Curva diurna (perfil hora-del-día) en V_hub y/o Potencia
    try:
        _tmp = _results.dropna(subset=["timestamp"]).copy()
        _tmp["hour"] = _tmp["timestamp"].dt.hour
        # promedio horario
        y_cols = [c for c in ["v_mean_hub_m_s", "P_net_kW"] if c in _tmp.columns]
        for y in y_cols:
            prof = _tmp.groupby("hour")[y].mean().reset_index()
            fig_d = px.line(prof, x="hour", y=y, markers=True)
            fig_d = apply_plotly_theme(fig_d, f"Perfil diurno — {y}", "Hora del día", y, height=320)
            s2.plotly_chart(fig_d, use_container_width=True)
    except Exception:
        pass

    m1, m2 = st.columns(2)

    # 3) Boxplot mensual de V_hub (variabilidad estacional)
    if "v_mean_hub_m_s" in _results.columns and _results["v_mean_hub_m_s"].notna().any():
        _bx = _results.dropna(subset=["timestamp", "v_mean_hub_m_s"]).copy()
        _bx["mes"] = _bx["timestamp"].dt.to_period("M").astype(str)
        fig_bx = px.box(_bx, x="mes", y="v_mean_hub_m_s", points=False)
        fig_bx = apply_plotly_theme(fig_bx, "Distribución mensual de velocidad (buje)", "Mes", "v_hub [m/s]", height=330)
        m1.plotly_chart(fig_bx, use_container_width=True)

    # 4) Duration curve (curva de duración) de velocidad o potencia
    if "v_mean_hub_m_s" in _results.columns:
        v_sorted = np.sort(_results["v_mean_hub_m_s"].dropna())[::-1]
        x_pct = np.linspace(0, 100, len(v_sorted), endpoint=False)
        fig_dc = go.Figure(go.Scatter(x=x_pct, y=v_sorted, mode="lines", name="v_hub"))
        fig_dc = apply_plotly_theme(fig_dc, "Curva de duración — velocidad", "% tiempo excedencia", "v_hub [m/s]", height=330)
        m2.plotly_chart(fig_dc, use_container_width=True)

    # (opcional) duration curve de Potencia
    if "P_net_kW" in _results.columns:
        p_sorted = np.sort(_results["P_net_kW"].dropna())[::-1]
        x_pct = np.linspace(0, 100, len(p_sorted), endpoint=False)
        fig_dcp = go.Figure(go.Scatter(x=x_pct, y=p_sorted, mode="lines", name="P_net"))
        fig_dcp.add_hline(y=float(pc_df["P_KW"].max()), line_dash="dot", line_color=GRAY_2)
        fig_dcp = apply_plotly_theme(fig_dcp, "Curva de duración — potencia neta", "% tiempo excedencia", "P [kW]", height=330)
        st.plotly_chart(fig_dcp, use_container_width=True)

# ---- TI, Gust y Rosa de vientos ----
extras = st.container()
with extras:
    col_gust, col_rose = st.columns(2)

def _has_wind(cols):
    return (
        modo == "Series (CSV)"
        and isinstance(wind_df, pd.DataFrame)
        and not wind_df.empty
        and all(c in wind_df.columns for c in cols)
    )

# --- Intensidad de turbulencia (histograma + p50) ---
if _has_wind(["v_std_m_s", "v_mean_m_s"]):
    with np.errstate(divide='ignore', invalid='ignore'):
        TI = (wind_df["v_std_m_s"] / wind_df["v_mean_m_s"].replace(0, np.nan)).clip(upper=5)

    fig_ti = px.histogram(TI.dropna(), nbins=40)
    fig_ti.update_traces(marker_color=PRIMARY, opacity=0.9)

    p50 = extra_metrics.get("TI_p50", np.nan)
    if np.isfinite(p50):
        fig_ti.add_vline(x=p50, line_dash="dash", line_color=SECOND, line_width=2)
        fig_ti.add_annotation(
            x=p50, xref="x", y=1.06, yref="paper", showarrow=False,
            text=f"p50 ≈ {p50:.2f}",
            font=dict(color=SECOND),
            bgcolor="rgba(51,65,85,.08)", bordercolor=SECOND, borderwidth=1
        )

    fig_ti = apply_plotly_theme(fig_ti, "Intensidad de turbulencia TI", "TI [-]", "frecuencia", height=320)
    fig_ti.update_layout(margin=dict(l=14, r=16, t=64, b=28))
    col_gust.plotly_chart(fig_ti, use_container_width=True)
else:
    col_gust.info("No hay datos de serie con 'v_std_m_s' y 'v_mean_m_s' (modo Series).")

# --- Rosa de vientos (barpolar pro) ---
with col_rose:
    if _has_wind(["direction_deg"]):
        dir_deg = (wind_df["direction_deg"].dropna().astype(float) % 360.0)
        sector_width = 22.5
        edges = np.arange(0, 360 + sector_width, sector_width)
        cats = pd.cut(dir_deg, bins=edges, right=False, include_lowest=True, labels=edges[:-1])
        counts = cats.value_counts().sort_index()

        if counts.sum() > 0:
            r = (counts / counts.sum() * 100.0).values
            theta = counts.index.astype(float) + (sector_width / 2.0)

            fig_rose = go.Figure(go.Barpolar(
                theta=theta, r=r, width=[sector_width]*len(theta),
                marker=dict(color=PRIMARY, line=dict(color="white", width=2)),
                hovertemplate="Dir %{theta:.0f}°<br>%{r:.1f}%<extra></extra>"
            ))
            fig_rose.update_layout(
                title={"text": "Rosa de viento (frecuencia %)", "x": 0.02, "y": 0.98,
                       "xanchor": "left", "yanchor": "top"},
                showlegend=False,
                height=320,
                margin=dict(l=8, r=8, t=64, b=8),
                polar=dict(
                    bgcolor="white",
                    radialaxis=dict(ticksuffix="%", angle=90, showline=False,
                                    gridcolor=GRID, gridwidth=1),
                    angularaxis=dict(direction="clockwise", rotation=90, gridcolor=GRID)
                ),
                paper_bgcolor="white"
            )
            col_rose.plotly_chart(fig_rose, use_container_width=True)
        else:
            col_rose.info("No hay datos suficientes para construir la rosa de viento.")
    else:
        col_rose.info("No hay columna 'direction_deg' en la serie (solo aplica en modo Series).")

# =========================
# (Sólo Weibull) — Análisis avanzado
# =========================
if (modo == "Distribución (Weibull)"
    and isinstance(_results, pd.DataFrame)
    and not _results.empty
    and {"v","pdf","P_net_kW"}.issubset(_results.columns)):

    with st.expander("📈 Análisis avanzado Weibull", expanded=True):

        v = _results["v"].to_numpy(dtype=float)
        f = _results["pdf"].to_numpy(dtype=float)
        Pk = _results["P_net_kW"].to_numpy(dtype=float)
        dv = float(v[1] - v[0]) if len(v) > 1 else 0.05
        

    
               # 1) PDF + CDF
        cdf = np.clip(np.cumsum(f) * dv, 0, 1)
        fig_pdfcdf = go.Figure()
        fig_pdfcdf.add_trace(go.Scatter(x=v, y=f, mode="lines", name="PDF f(v)"))
        fig_pdfcdf.add_trace(go.Scatter(x=v, y=cdf, mode="lines", name="CDF F(v)", yaxis="y2"))
        fig_pdfcdf.update_layout(
            yaxis=dict(title="f(v)"),
            yaxis2=dict(title="F(v)", overlaying="y", side="right")
        )
        for thr in [3, 5, 8, 10]:
            fig_pdfcdf.add_vline(x=thr, line_dash="dot", line_color=GRAY_2)
        fig_pdfcdf = apply_plotly_theme(fig_pdfcdf, "Weibull — PDF y CDF", "v [m/s]", "f(v)", height=330)
        st.plotly_chart(fig_pdfcdf, use_container_width=True)

        # 1.5) Energía anual por bin + acumulado (NUEVO)
        if "E_kWh_bin" in _results.columns and _results["E_kWh_bin"].notna().any():
            v_bins = _results["v"].to_numpy(dtype=float)
            e_bin  = _results["E_kWh_bin"].fillna(0).to_numpy(dtype=float)

            e_total = float(e_bin.sum())
            e_cum_pct = (np.cumsum(e_bin) / e_total * 100.0) if e_total > 0 else np.zeros_like(e_bin)

            # Top-3 bins (para texto)
            top_idx = np.argsort(e_bin)[-3:][::-1]
            top_info = ", ".join([f"{v_bins[i]:.1f} m/s: {e_bin[i]/1000:.1f} MWh" for i in top_idx])

            fig_e_bins = make_subplots(specs=[[{"secondary_y": True}]])
            # Barras: kWh/año por bin
            fig_e_bins.add_trace(
                go.Bar(x=v_bins, y=e_bin, name="E_kWh_bin", marker_line_width=0, opacity=0.9),
                secondary_y=False
            )
            # Línea acumulada: %
            fig_e_bins.add_trace(
                go.Scatter(x=v_bins, y=e_cum_pct, name="Acumulado (%)", mode="lines"),
                secondary_y=True
            )

            # Ejes y layout
            fig_e_bins.update_xaxes(title_text="v [m/s]")
            fig_e_bins.update_yaxes(title_text="kWh/año", secondary_y=False)
            fig_e_bins.update_yaxes(title_text="% acumulado", range=[0, 100], secondary_y=True)

            fig_e_bins = apply_plotly_theme(
                fig_e_bins,
                "Energía anual por bin (Weibull) + acumulado",
                "v [m/s]",
                "kWh/año",
                height=360
            )
            st.plotly_chart(fig_e_bins, use_container_width=True)

            # Tabla + descarga
            bin_df = pd.DataFrame({
                "v_m_s": v_bins,
                "E_kWh_bin": e_bin,
                "E_%_acumulado": e_cum_pct
            })
            st.caption(f"Top energía por bin: {top_info}")
            _buf_bins = io.StringIO(); bin_df.to_csv(_buf_bins, index=False)
            st.download_button(
                "📥 Descargar energía por bin (CSV)",
                _buf_bins.getvalue(),
                file_name="energia_por_bin_weibull.csv",
                mime="text/csv",
                key="dl_weibull_bins_csv"
            )
        else:
            st.info("No hay E_kWh_bin disponible para este modo.")


        # 2) Contribución al AEP por velocidad (%)
        if "E_kWh_bin" in _results.columns and _results["E_kWh_bin"].notna().any():
            e = _results["E_kWh_bin"].fillna(0).to_numpy()
            share = (e / e.sum() * 100.0) if e.sum() > 0 else e
            fig_share = px.area(x=v, y=share)
            fig_share = apply_plotly_theme(fig_share, "Contribución al AEP por velocidad", "v [m/s]", "% del AEP", height=330)
            st.plotly_chart(fig_share, use_container_width=True)

        # 3) Potencia esperada vs nominal
        y_nom_kw = pc_df["P_KW"] if "P_KW" in pc_df.columns else (pc_df["P_W"] / 1000.0)
        fig_pv2 = go.Figure()
        fig_pv2.add_trace(go.Scatter(x=v, y=Pk, mode="lines", name="P_net(v) esperada"))
        fig_pv2.add_trace(go.Scatter(x=pc_df["v_m_s"], y=y_nom_kw, mode="lines+markers",
                                     name="Curva nominal P(v)", line=dict(width=2)))
        fig_pv2 = apply_plotly_theme(fig_pv2, "Potencia esperada vs nominal", "v [m/s]", "P [kW]", height=330)
        st.plotly_chart(fig_pv2, use_container_width=True)

        # 4) Curva de duración de potencia (Firm power)
        order = np.argsort(Pk)[::-1]
        P_sorted = Pk[order]
        f_sorted = f[order]
        exced_prob = np.clip(np.cumsum(f_sorted) * dv, 0, 1)
        exced_pct = (1.0 - exced_prob) * 100.0
        def _p_at(qpct):
            idx = np.searchsorted(exced_pct[::-1], qpct)
            idx = len(exced_pct) - 1 - idx
            idx = int(np.clip(idx, 0, len(P_sorted)-1))
            return float(P_sorted[idx])
        P50 = _p_at(50.0)
        P95 = _p_at(5.0)
        fig_dur = go.Figure(go.Scatter(x=exced_pct, y=P_sorted, mode="lines", name="P vs % excedencia"))
        fig_dur.add_vline(x=50, line_dash="dot", line_color=SECOND); fig_dur.add_hline(y=P50, line_dash="dot", line_color=SECOND)
        fig_dur.add_vline(x=5,  line_dash="dot", line_color=ACCENT); fig_dur.add_hline(y=P95, line_dash="dot", line_color=ACCENT)
        fig_dur.add_annotation(x=50, y=P50, text=f"P@50 ≈ {P50:.1f} kW", showarrow=False, yshift=10, font=dict(color=SECOND))
        fig_dur.add_annotation(x=5,  y=P95, text=f"P@95 ≈ {P95:.1f} kW", showarrow=False, yshift=10, font=dict(color=ACCENT))
        fig_dur = apply_plotly_theme(fig_dur, "Curva de duración de potencia (Firm power)", "% tiempo excedencia", "P [kW]", height=330)
        st.plotly_chart(fig_dur, use_container_width=True)

        # 5) Sensibilidad AEP(k,c) — Heatmap ±20%
        k0, c0 = float(k_weibull), float(c_weibull)
        k_vec = np.linspace(max(0.6, k0*0.8), k0*1.2, 25)
        c_vec = np.linspace(max(0.5, c0*0.8), c0*1.2, 25)
        AEP_grid = np.zeros((len(k_vec), len(c_vec)))
        for i, kk in enumerate(k_vec):
            for j, cc in enumerate(c_vec):
                AEP_grid[i, j], _, _ = aep_from_weibull(
                    pc_df, kk, cc,
                    hours_year=8760,
                    inverter_eff=inverter_eff,
                    losses=losses,
                    rho=air_density(temp_default, press_default),
                    rho_ref=1.225
                )
        fig_hm = go.Figure(data=go.Heatmap(
            z=AEP_grid/1000.0,
            x=[f"{c:.2f}" for c in c_vec],
            y=[f"{k:.2f}" for k in k_vec],
            colorbar=dict(title="AEP [MWh/año]")
        ))
        j0 = int(np.argmin(np.abs(c_vec - c0)))
        i0 = int(np.argmin(np.abs(k_vec - k0)))
        fig_hm.add_scatter(x=[f"{c_vec[j0]:.2f}"], y=[f"{k_vec[i0]:.2f}"], mode="markers",
                           marker=dict(size=10, symbol="x", line=dict(width=2, color="black")),
                           name="k,c actuales")
        fig_hm = apply_plotly_theme(fig_hm, "Sensibilidad AEP vs k,c (±20%)", "c [m/s]", "k [-]", height=420)
        st.plotly_chart(fig_hm, use_container_width=True)

        st.markdown(
            f"- **P@50** ≈ **{P50:.1f} kW** · **P@95** ≈ **{P95:.1f} kW**  \n"
            f"- Heatmap: variación de **AEP** alrededor de **k={k0:.2f}**, **c={c0:.2f} m/s**."
        )

# ---------------------------
# PROPUESTA TÉCNICA (diseño mejorado) — con Tarifa Turbinas y nuevo KPI de Ahorro
# ---------------------------

# ====== Sidebar: asegurar inputs editables (si no existen) ======
try:
    _ = tarifa_turbinas_val
except NameError:
    # Asegura que exista el input en el sidebar con paso 0.1 (centavos)
    tarifa_turbinas_val = st.sidebar.number_input(
        "Tarifa turbinas ($/kWh)",
        min_value=0.0,
        value=100.0,
        step=0.1,
        help="Tarifa nivelada o referencia del parque eólico; se compara contra la tarifa actual para el KPI de ahorro."
    )

# (Opcional) Si quieres que la tarifa actual también tenga paso 0.1, asegúrate de declararla así en el sidebar:
# tarifa_actual_val = st.sidebar.number_input("Tarifa eléctrica actual ($/kWh)", min_value=0.0, value=170.0, step=0.1)

try:
    # ====== Lectura de variables base ======
    potencia_turbina_kW = float(pc_df["P_KW"].max()) if "P_KW" in pc_df.columns else float(pc_df["P_W"].max()) / 1000.0
    energia_diaria_turbina_kWh = float(mean_daily_kWh)
    AEP_por_turbina_MWh = float(AEP_kWh) / 1000.0 if AEP_kWh else 0.0

    # Si el usuario no ingresó consumo anual explícito, se calcula desde el consumo diario
    consumo_anual_was_auto = False
    if consumo_anual_MWh_pt <= 0.0:
        consumo_anual_MWh_pt = (consumo_diario_kWh_pt * 365.0) / 1000.0  # MWh/año
        consumo_anual_was_auto = True

    # ====== Dimensionamiento de nº de turbinas (en base a demanda diaria + margen) ======
    if energia_diaria_turbina_kWh > 0:
        demanda_diaria_ajustada = consumo_diario_kWh_pt * (1.0 + margen_seguridad_pct / 100.0)
        n_turbinas = int(np.ceil(demanda_diaria_ajustada / energia_diaria_turbina_kWh))
    else:
        n_turbinas = 0

    # ====== Energías y cobertura ======
    energia_anual_proyecto_MWh = n_turbinas * AEP_por_turbina_MWh
    cobertura_anual_pct = (energia_anual_proyecto_MWh / consumo_anual_MWh_pt * 100.0) if consumo_anual_MWh_pt > 0 else 0.0

    # Energía facturable (se mantiene por si la usas en otras partes)
    energia_facturable_MWh = min(energia_anual_proyecto_MWh, consumo_anual_MWh_pt)

    # ====== KPI: Ahorro por diferencia de tarifas × Consumo anual estimado ======
    tarifa_turbinas_val = float(tarifa_turbinas_val)
    ahorro_bruto_anual = (consumo_anual_MWh_pt * 1000.0) * (float(tarifa_actual_val) - tarifa_turbinas_val)

    # ====== UI / Presentación ======
    st.markdown("---")
    st.markdown(
        """
        <style>
        .kpi-card{border:1px solid rgba(0,0,0,.08);border-radius:14px;padding:14px 16px;
                  background:linear-gradient(180deg,#FAFCFF 0%,#FFFFFF 100%);box-shadow:0 1px 2px rgba(0,0,0,.04)}
        .kpi-title{font-size:12px;color:#39424E;letter-spacing:.02em;margin:0 0 6px 0;text-transform:uppercase}
        .kpi-value{font-weight:700;font-size:28px;line-height:1.1;color:#101828;margin:0}
        .kpi-sub{font-size:12px;color:#6B7280;margin-top:6px}
        .badge-green{display:inline-block;padding:2px 8px;border-radius:999px;font-size:11px;
                     background:#DCFCE7;color:#166534;border:1px solid #86EFAC;margin-left:8px}
        </style>
        """,
        unsafe_allow_html=True
    )

    st.header("Propuesta técnica")

    # Nota si el consumo anual se autocalculó
    if consumo_anual_was_auto:
        st.caption("ℹ️ Consumo anual estimado calculado como Consumo diario × 365 / 1000.")

    # KPIs superiores (técnicos)
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            f'<div class="kpi-card"><p class="kpi-title">Potencia por turbina</p>'
            f'<p class="kpi-value">{potencia_turbina_kW:,.1f} kW</p>'
            f'<div class="kpi-sub">P<sub>rated</sub> (máx. curva)</div></div>',
            unsafe_allow_html=True
        )
    with k2:
        st.markdown(
            f'<div class="kpi-card"><p class="kpi-title">Energía diaria por turbina</p>'
            f'<p class="kpi-value">{energia_diaria_turbina_kWh:,.1f} kWh/día</p>'
            f'<div class="kpi-sub">Media estimada (η, pérdidas, ρ)</div></div>',
            unsafe_allow_html=True
        )
    with k3:
        st.markdown(
            f'<div class="kpi-card"><p class="kpi-title">Turbinas necesarias</p>'
            f'<p class="kpi-value">{n_turbinas:d}</p>'
            f'<div class="kpi-sub">Demanda diaria ⋅ (1 + margen)</div></div>',
            unsafe_allow_html=True
        )
    with k4:
        st.markdown(
            f'<div class="kpi-card"><p class="kpi-title">Energía adicional de riesgo</p>'
            f'<p class="kpi-value">{energia_riesgo_MWp:,.2f} MWp</p>'
            f'<div class="kpi-sub">Reserva sugerida</div></div>',
            unsafe_allow_html=True
        )

    # Visuales
    vcol1, vcol2 = st.columns(2)

    with vcol1:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=float(cobertura_anual_pct) if np.isfinite(cobertura_anual_pct) else 0.0,
            number={"suffix": " %", "font": {"size": 26}},
            gauge={
                "axis": {"range": [0, max(120, round((cobertura_anual_pct or 0) + 10))]},
                "bar": {"thickness": 0.25, "color": PRIMARY},
                "steps": [
                    {"range": [0, 80], "color": "#FEE2E2"},
                    {"range": [80, 100], "color": "#FEF9C3"},
                    {"range": [100, 120], "color": "#DCFCE7"}
                ],
                "threshold": {"line": {"color": SECOND, "width": 4}, "thickness": 0.75, "value": 100}
            },
            title={"text": "Cobertura anual", "font": {"size": 16}}
        ))
        gauge = apply_plotly_theme(gauge, height=240)
        st.plotly_chart(gauge, use_container_width=True)

    with vcol2:
        bal_df = pd.DataFrame({
            "Concepto": ["Consumo anual", "Energía proyecto"],
            "MWh": [float(consumo_anual_MWh_pt), float(energia_anual_proyecto_MWh)]
        })

        # Alto máximo + holgura para que no se corte el texto
        y_max = max(bal_df["MWh"].max(), 0) * 1.15

        bar = px.bar(
            bal_df,
            x="Concepto",
            y="MWh",
            text="MWh",
            color="Concepto",
            color_discrete_sequence=[SECOND, PRIMARY]
        )

        # Textos fuera y sin sobreposición
        bar.update_traces(
            texttemplate="%{text:,.1f}",
            textposition="outside",
            insidetextanchor="end",   # no afecta si es "outside", pero no estorba
            opacity=0.95,
            cliponaxis=False          # permite que el texto salga del área si hace falta
        )

        # Tema + márgenes + rango Y con holgura
        bar = apply_plotly_theme(bar, "Balance energético anual", "", "MWh", height=280)
        bar.update_layout(
            title=dict(y=0.98, x=0.02, xanchor="left", yanchor="top"),
            margin=dict(l=10, r=10, t=80, b=50),
            xaxis=dict(automargin=True),
            yaxis=dict(automargin=True, range=[0, y_max])
        )

        st.plotly_chart(bar, use_container_width=True)

    # KPIs inferiores (energéticos y económicos)
    b1, b2, b3, b4 = st.columns(4)
    with b1:
        st.markdown(
            f'<div class="kpi-card"><p class="kpi-title">Consumo anual estimado</p>'
            f'<p class="kpi-value">{consumo_anual_MWh_pt:,.1f} MWh/año</p></div>',
            unsafe_allow_html=True
        )
    with b2:
        st.markdown(
            f'<div class="kpi-card"><p class="kpi-title">Energía anual proyecto</p>'
            f'<p class="kpi-value">{energia_anual_proyecto_MWh:,.1f} MWh/año</p></div>',
            unsafe_allow_html=True
        )
    with b3:
        chip = '<span class="badge-green">≥100% objetivo</span>' if cobertura_anual_pct >= 100 else ""
        st.markdown(
            f'<div class="kpi-card"><p class="kpi-title">Cobertura anual</p>'
            f'<p class="kpi-value">{cobertura_anual_pct:,.1f} % {chip}</p></div>',
            unsafe_allow_html=True
        )
   

    # ====== Descarga CSV con las nuevas columnas ======
    with st.expander("Ver detalle / descargar CSV"):
        pt_df = pd.DataFrame({
            "Potencia_turbina_kW": [potencia_turbina_kW],
            "Energia_diaria_turbina_kWh_d": [energia_diaria_turbina_kWh],
            "Turbinas_requeridas": [n_turbinas],
            "Consumo_anual_MWh": [consumo_anual_MWh_pt],
            "Energia_anual_proyecto_MWh": [energia_anual_proyecto_MWh],
            "Cobertura_anual_pct": [cobertura_anual_pct],
            "Tarifa_actual_$por_kWh": [tarifa_actual_val],
            "Tarifa_turbinas_$por_kWh": [tarifa_turbinas_val],
            "Ahorro_bruto_anual_$": [ahorro_bruto_anual],
            "Metodo_ahorro": ["Diferencia de tarifas × Consumo anual"],
            "Margen_seguridad_pct": [margen_seguridad_pct],
            "Energia_riesgo_MWp": [energia_riesgo_MWp]
        })
        st.dataframe(pt_df, use_container_width=True)
        buf_pt = io.StringIO(); pt_df.to_csv(buf_pt, index=False)
        st.download_button(
            "📥 Descargar propuesta técnica (CSV)",
            buf_pt.getvalue(),
            file_name="propuesta_tecnica.csv", mime="text/csv"
        )

except Exception as e:
    st.warning(f"No se pudo calcular la Propuesta técnica: {e}")



# ---------------------------
# PROPUESTA FINANCIERA
# ---------------------------

st.markdown("---")
st.header("Propuesta Financiera")

# URL con parámetros editables en la hoja (para $/kW, $/kWh, % y horas)
PARAMS_CSV_URL_DEFAULT = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRb9avOuK8IILVQcc5WKSK9C01pRODcOtO60RNqwo-RstqV3kpEwEzyCM0X5TeEBf9Jje76370ly74d/pub?gid=1849740876&single=true&output=csv"
params_url = st.text_input("URL parámetros (CSV Google Sheets)", value=PARAMS_CSV_URL_DEFAULT)

def _read_params_from_sheet(url: str) -> dict:
    out = {}
    try:
        df = pd.read_csv(url, dtype=str)
    except Exception:
        return out

    col_turb = _guess_col(df, ["turbina", "$/kw", "usd/kw", "precio turb", "turb_kw"])
    col_bess = _guess_col(df, ["bess", "bateria", "bater", "$/kwh", "usd/kwh", "precio bess", "precio bateria"])
    col_pct  = _guess_col(df, ["almacen", "% almacen", "storage", "% storage", "porcentaje"])
    col_hour = _guess_col(df, ["hora", "autonomia", "h"])

    if col_turb:
        v = pd.to_numeric(df[col_turb].apply(_to_num), errors="coerce").dropna()
        if len(v): out["turb_usd_kw"] = float(v.iloc[0])
    if col_bess:
        v = pd.to_numeric(df[col_bess].apply(_to_num), errors="coerce").dropna()
        if len(v): out["bess_usd_kwh"] = float(v.iloc[0])
    if col_pct:
        v = pd.to_numeric(df[col_pct].apply(_to_num), errors="coerce").dropna()
        if len(v):
            x = float(v.iloc[0]); out["storage_pct"] = x*100 if 0 <= x <= 1 else x
    if col_hour:
        v = pd.to_numeric(df[col_hour].apply(_to_num), errors="coerce").dropna()
        if len(v): out["storage_hours"] = float(v.iloc[0])

    if not out:
        key_col = _guess_col(df, ["param", "parametro", "concepto", "clave", "campo", "nombre"])
        val_col = _guess_col(df, ["valor", "value", "monto", "dato", "cantidad"])
        if key_col and val_col:
            df2 = df[[key_col, val_col]].dropna()
            df2[key_col] = df2[key_col].map(_norm_text)

            mask = df2[key_col].str.contains(r"turb.*(kw|\$/kw|usd/kw)|\$/kw|usd/kw", regex=True)
            v = pd.to_numeric(df2.loc[mask, val_col].apply(_to_num), errors="coerce").dropna()
            if len(v): out["turb_usd_kw"] = float(v.iloc[0])

            mask = df2[key_col].str.contains(r"(bess|bater).*(/?kwh)|\$/kwh|usd/kwh", regex=True)
            v = pd.to_numeric(df2.loc[mask, val_col].apply(_to_num), errors="coerce").dropna()
            if len(v): out["bess_usd_kwh"] = float(v.iloc[0])

            mask = df2[key_col].str.contains(r"almacen|storage|%.*almacen", regex=True)
            v = pd.to_numeric(df2.loc[mask, val_col].apply(_to_num), errors="coerce").dropna()
            if len(v):
                x = float(v.iloc[0]); out["storage_pct"] = x*100 if 0 <= x <= 1 else x

            mask = df2[key_col].str.contains(r"hora|autonom", regex=True)
            v = pd.to_numeric(df2.loc[mask, val_col].apply(_to_num), errors="coerce").dropna()
            if len(v): out["storage_hours"] = float(v.iloc[0])

    return out


# ====== Cálculo del costeo directo ======
if ("n_turbinas" not in locals() or "potencia_turbina_kW" not in locals() or
        n_turbinas <= 0 or potencia_turbina_kW <= 0):
    st.info("🔎 Primero ejecuta la **Propuesta técnica** para obtener número de turbinas y potencia por turbina.")
else:
    # Defaults + lectura desde la hoja
    defaults = dict(turb_usd_kw=1482.0, bess_usd_kwh=303.0, storage_pct=20.0, storage_hours=4.0)
    sheet_params = _read_params_from_sheet(params_url) if params_url else {}
    pars = {**defaults, **sheet_params}

    # KPIs superiores + entradas editables
    p0, p1, p2, p3 = st.columns(4)
    with p0: st.metric("Turbinas", f"{int(n_turbinas)} u.")
    with p1: st.metric("Potencia total", f"{(n_turbinas * potencia_turbina_kW):,.0f} kW")
    with p2: turb_usd_per_kw  = st.number_input("Precio turbina ($/kW)", min_value=0.0, value=float(pars["turb_usd_kw"]), step=1.0)
    with p3: bess_usd_per_kwh = st.number_input("Precio BESS ($/kWh)",   min_value=0.0, value=float(pars["bess_usd_kwh"]), step=1.0)

    b0, b1 = st.columns(2)
    with b0:
        storage_pct = st.slider("% almacenamiento sobre kW", min_value=0, max_value=100,
                                value=int(round(pars["storage_pct"])), step=1)
    with b1:
        storage_hours = st.number_input("Horas de autonomía", min_value=0.25,
                                        value=float(pars["storage_hours"]), step=0.25)

    # Cálculos
    total_kW = float(n_turbinas) * float(potencia_turbina_kW)
    costo_turbinas = total_kW * turb_usd_per_kw
    bess_kWh       = total_kW * (storage_pct / 100.0) * float(storage_hours)
    costo_bess     = bess_kWh * bess_usd_per_kwh
    capex_total_directo = costo_turbinas + costo_bess
    st.session_state["capex_total_directo"] = float(capex_total_directo)

    df_cost = (
        pd.DataFrame({
            "Componente": ["Turbinas", "BESS"],
            "Costo_USD": [costo_turbinas, costo_bess],
            "USD_por_turbina": [
                costo_turbinas / max(1, int(n_turbinas)),
                costo_bess     / max(1, int(n_turbinas))
            ],
        })
        .sort_values("Costo_USD", ascending=False)
        .reset_index(drop=True)
    )

    # === KPIs pro refinados
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f'<div class="kpi-card"><p class="kpi-title">CAPEX total (directo)</p>'
                    f'<p class="kpi-value">${capex_total_directo:,.0f}</p>'
                    f'<div class="kpi-sub">Turbinas + BESS</div></div>', unsafe_allow_html=True)
    with k2:
        st.markdown(f'<div class="kpi-card"><p class="kpi-title">BESS dimensionado</p>'
                    f'<p class="kpi-value">{bess_kWh:,.0f} kWh</p>'
                    f'<div class="kpi-sub">{storage_pct}% · {storage_hours:g} h</div></div>', unsafe_allow_html=True)
    with k3:
        st.markdown(f'<div class="kpi-card"><p class="kpi-title">USD/turbina (total)</p>'
                    f'<p class="kpi-value">${(capex_total_directo/max(1,int(n_turbinas))):,.0f}</p>'
                    f'<div class="kpi-sub">Promedio</div></div>', unsafe_allow_html=True)
    with k4:
         ahorro_fmt = f"${ahorro_bruto_anual:,.0f}"
         color = "#16A34A" if ahorro_bruto_anual >= 0 else "#DC2626"
         subt = f"Actual: {tarifa_actual_val:,.1f} $/kWh · Turbinas: {tarifa_turbinas_val:,.1f} $/kWh"
         st.markdown(
             f'<div class="kpi-card"><p class="kpi-title">Ahorro bruto anual</p>'
             f'<p class="kpi-value" style="color:{color}">{ahorro_fmt}</p>'
             f'<div class="kpi-sub">{subt}</div></div>',
            unsafe_allow_html=True
    )


        # === Visual PRO: paleta, abreviación de montos y modo barra minimal ===
    COLOR_MAP = {"Turbinas": PRIMARY, "BESS": SECOND}

    def _abbr_usd(x: float) -> str:
        if x is None or not np.isfinite(x): return ""
        if abs(x) >= 1e9:  return f"${x/1e9:,.2f}B"
        if abs(x) >= 1e6:  return f"${x/1e6:,.2f}M"
        if abs(x) >= 1e3:  return f"${x/1e3:,.0f}k"
        return f"${x:,.0f}"

    PLOT_CFG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}

    # === Gráfico 1: CAPEX directo por componente (barra vertical, rótulo abreviado, ordenado) ===
    with st.container():
        st.markdown("### CAPEX directo por componente")

        _df = df_cost.sort_values("Costo_USD", ascending=False).copy()
        _df["Label"] = _df["Costo_USD"].apply(_abbr_usd)

        y_max = float(_df["Costo_USD"].max()) * 1.18  # holgura para etiquetas fuera

        fig_capex_comp = px.bar(
            _df,
            x="Componente", y="Costo_USD", text="Label",
            color="Componente", color_discrete_map=COLOR_MAP
        )
        fig_capex_comp.update_traces(
            textposition="outside",
            cliponaxis=False,
            marker=dict(line=dict(color="rgba(0,0,0,.12)", width=1.2)),
            hovertemplate="<b>%{x}</b><br>$%{y:,.0f}<extra></extra>"
        )
        fig_capex_comp.update_xaxes(
            title_text="Componente",
            categoryorder="array",
            categoryarray=_df["Componente"].tolist()
        )
        fig_capex_comp.update_yaxes(title_text="USD", range=[0, y_max], automargin=True)

        fig_capex_comp = apply_plotly_theme(fig_capex_comp, "", "Componente", "USD", height=380)
        fig_capex_comp.update_layout(
            title=dict(text="CAPEX directo por componente", x=0.02, y=0.98, xanchor="left"),
            legend=dict(orientation="h", x=0.0, y=1.06),
            margin=dict(l=10, r=10, t=70, b=60)
        )
        st.plotly_chart(fig_capex_comp, use_container_width=True, config=PLOT_CFG)

    # === Gráfico 2: Participación CAPEX directo (donut con total en el centro) ===
    with st.container():
        st.markdown("### Participación CAPEX directo")

        fig_share = px.pie(
            _df, names="Componente", values="Costo_USD", hole=0.55,
            color="Componente", color_discrete_map=COLOR_MAP
        )
        fig_share.update_traces(
            textposition="inside",
            texttemplate="%{percent:.1%}",
            marker=dict(line=dict(color="white", width=2)),
            hovertemplate="<b>%{label}</b><br>%{percent:.1%}<br>$%{value:,.0f}<extra></extra>"
        )
        total_capex = float(_df["Costo_USD"].sum())
        fig_share.add_annotation(
            text=f"Total<br{''}> {_abbr_usd(total_capex)}",
            x=0.5, y=0.5, showarrow=False, xref="paper", yref="paper",
            font=dict(size=15, color=GRAY_1)
        )
        fig_share = apply_plotly_theme(fig_share, "", "", "", height=340)
        fig_share.update_layout(
            legend=dict(orientation="h", x=0.0, y=1.06),
            margin=dict(l=10, r=10, t=30, b=60)
        )
        st.plotly_chart(fig_share, use_container_width=True, config=PLOT_CFG)

    # === Gráfico 3: Costo por turbina (horizontal, rótulos abreviados) ===
    with st.container():
        st.markdown("### Costo por turbina (componentes)")

        _pt = df_cost.sort_values("USD_por_turbina", ascending=True).copy()
        _pt["Label"] = _pt["USD_por_turbina"].apply(_abbr_usd)

        fig_by_turb = px.bar(
            _pt, x="USD_por_turbina", y="Componente", orientation="h",
            text="Label", color="Componente", color_discrete_map=COLOR_MAP
        )
        fig_by_turb.update_traces(
            textposition="outside",
            cliponaxis=False,
            marker=dict(line=dict(color="rgba(0,0,0,.12)", width=1.2)),
            hovertemplate="<b>%{y}</b><br>$%{x:,.0f} por turbina<extra></extra>"
        )
        fig_by_turb.update_xaxes(
            title_text="USD/turbina",
            tickprefix="$", separatethousands=True,
            automargin=True
        )
        fig_by_turb.update_yaxes(title_text="", automargin=True)

        fig_by_turb = apply_plotly_theme(fig_by_turb, "", "USD/turbina", "", height=360)
        fig_by_turb.update_layout(
            legend=dict(orientation="h", x=0.0, y=1.06),
            margin=dict(l=10, r=10, t=30, b=60)
        )
        st.plotly_chart(fig_by_turb, use_container_width=True, config=PLOT_CFG)

    # === Descarga CSV (costos directos)
    buf_cost = io.StringIO()
    df_cost.to_csv(buf_cost, index=False)
    st.download_button("📥 Descargar costos (CSV)", buf_cost.getvalue(),
                       file_name="costos_directos_turbinas_bess.csv",
                       mime="text/csv", key="dl_costos_directos_csv")

# =========================
# CAPEXV2 (Item/Participación/Monto/Categoría) — DESDE URL
# =========================
CAPEX_CSV_URL_DEFAULT = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRb9avOuK8IILVQcc5WKSK9C01pRODcOtO60RNqwo-RstqV3kpEwEzyCM0X5TeEBf9Jje76370ly74d/pub?gid=1849740876&single=true&output=csv"
capex_url = st.text_input("URL CAPEX (Item / Categoría / Participación / Monto) – CSV", value=CAPEX_CSV_URL_DEFAULT)

def _read_capex_table(url: str) -> pd.DataFrame | None:
    """Devuelve columnas normalizadas: [Item, Categoria, Participacion_%, Monto_USD]"""
    try:
        df_raw = pd.read_csv(url, header=None, dtype=str)
    except Exception:
        return None

    def nrm(x: str) -> str:
        if pd.isna(x): return ""
        x = str(x).strip()
        x = "".join(ch for ch in _ud.normalize("NFKD", x) if not _ud.combining(ch))
        return x.lower()

    # Detecta fila de cabeceras
    header_row = None
    for i in range(len(df_raw)):
        vals = [nrm(v) for v in df_raw.iloc[i].tolist()]
        if sum(v != "" for v in vals) < 2:
            continue
        has_item = any(k in v for v in vals for k in ["item","concepto","descripcion","nombre"])
        has_part = any(("particip" in v or "%" in v or "porcentaje" in v) for v in vals)
        if has_item and has_part:
            header_row = i
            break
    if header_row is None:
        return None

    df = pd.read_csv(url, header=header_row, dtype=str)

    # Localiza columnas
    def find_col(keys, regexes=()):
        for c in df.columns:
            nc = nrm(c)
            if any(k in nc for k in keys): return c
            if any(re.search(rx, nc) for rx in regexes): return c
        return None

    col_item = find_col(["item","concepto","descripcion","nombre"])
    col_cat  = find_col(["categoria","categoría","grupo","familia"])
    col_part = find_col(["particip","%","porcentaje"], [r"particip.*%|%.*particip"])
    col_mto  = find_col(["monto","usd","estim","costo","total"],
                        [r"monto.*usd|usd.*estim|costo.*usd|total.*usd|usd.*total"])

    if not (col_item and col_part):
        return None

    cols = [col_item, col_part] + ([col_cat] if col_cat else []) + ([col_mto] if col_mto else [])
    out = df[cols].copy().rename(columns={
        col_item: "Item",
        col_part: "Participacion_%",
        **({col_cat: "Categoria"} if col_cat else {}),
        **({col_mto: "Monto_USD"} if col_mto else {}),
    }).dropna(how="all")

    # Limpieza
    out["Item"] = out["Item"].astype(str).str.strip()
    out = out[~out["Item"].str.lower().str.fullmatch("item|concepto|descripcion|nombre", na=False)]
    if "Categoria" not in out.columns: out["Categoria"] = "Sin categoría"

    # Normaliza % robusto
    out["Participacion_%"] = pd.to_numeric(
        out["Participacion_%"].astype(str).str.replace("%","",regex=False).apply(_to_num),
        errors="coerce"
    )
    if "Monto_USD" in out.columns:
        out["Monto_USD"] = pd.to_numeric(out["Monto_USD"].astype(str).apply(_to_num), errors="coerce")

    out = out.dropna(subset=["Item"]).reset_index(drop=True)
    return out if len(out) else None


# ====== Visualización CAPEXV2 PRO ======
st.subheader("Desgloce del Capex")
capex_df = _read_capex_table(capex_url)

if capex_df is None:
    st.warning("No se pudo leer la tabla CAPEXV2 desde la URL. Verifica permisos y formato.")
else:
    # Completa % desde Monto si la hoja ya trae montos
    if "Monto_USD" in capex_df.columns and capex_df["Monto_USD"].notna().any():
        total_hoja = capex_df["Monto_USD"].sum(skipna=True)
        if total_hoja:
            mask = capex_df["Participacion_%"].isna() & capex_df["Monto_USD"].notna()
            capex_df.loc[mask, "Participacion_%"] = (capex_df.loc[mask, "Monto_USD"] / total_hoja) * 100.0

    # Calcula Monto_USD desde % usando CAPEX total de la 4ª etapa
    capex_base = st.session_state.get("capex_total_directo", None)

    # Si los % vinieran en 0–1, pasa a 0–100
    if capex_df["Participacion_%"].max(skipna=True) <= 1.0:
        capex_df["Participacion_%"] = capex_df["Participacion_%"] * 100.0

    if capex_base is not None and np.isfinite(capex_base):
        capex_df["Monto_USD"] = (capex_df["Participacion_%"].fillna(0.0) / 100.0) * float(capex_base)
    else:
        if "Monto_USD" not in capex_df.columns:
            capex_df["Monto_USD"] = np.nan

    # Aviso si % totales difieren de 100
    pct_total = capex_df["Participacion_%"].sum(skipna=True)
    if np.isfinite(pct_total) and not (99.5 <= pct_total <= 100.5):
        st.info(f"⚠️ La suma de Participación_% en la tabla base es {pct_total:,.2f} %. Revisa o ajusta la hoja si corresponde.")

    # ========= Filtros (por Ítem) =========
    with st.expander("🎛️ Filtros de visualización (por Ítem)", expanded=True):
        st.markdown("<style>.hint{font-size:12px; color:#6B7280; margin-top:.25rem}</style>", unsafe_allow_html=True)

        items_all = sorted(capex_df["Item"].dropna().astype(str).unique().tolist())
        q_items = st.text_input("🔎 Buscar ítems (separa varias palabras con espacio)", value="")
        if q_items.strip():
            words = [w.strip().lower() for w in q_items.split() if w.strip()]
            def _matches_all(name: str) -> bool:
                n = str(name).lower()
                return all(w in n for w in words)
            items_options = [it for it in items_all if _matches_all(it)]
        else:
            items_options = items_all

        key_ms = "items_ms_key"
        if key_ms not in st.session_state:
            st.session_state[key_ms] = items_options[:]    # valor inicial

        # Botones rápidos
        c1, c2, _ = st.columns([1,1,3])
        with c1:
            if st.button("Seleccionar todo", key="btn_sel_all"):
                st.session_state[key_ms] = items_options[:]
        with c2:
            if st.button("Limpiar", key="btn_clear_all"):
                st.session_state[key_ms] = []

        # Importante: no pasamos 'default' para evitar el warning; al tener 'key',
        # Streamlit toma el valor desde session_state.
        items_sel = st.multiselect(
            "Filtrar por ítem",
            options=items_options,
            key=key_ms,
            help="Usa la búsqueda para acotar opciones; luego selecciona uno o varios ítems."
        )
        st.markdown(
            f"<div class='hint'>Ítems disponibles: <b>{len(items_options)}</b> · Ítems seleccionados: <b>{len(items_sel)}</b></div>",
            unsafe_allow_html=True
        )

    # ========= Vista filtrada (por Ítem) =========
    df_view = capex_df.copy()
    if items_sel:
        df_view = df_view[df_view["Item"].isin(items_sel)]
    elif q_items.strip():
        df_view = df_view[df_view["Item"].isin(items_options)]

    # === Totales y subtotales robustos
    has_monto = df_view["Monto_USD"].notna().any()
    tot_monto = df_view["Monto_USD"].sum(skipna=True) if has_monto else np.nan

    agg_dict = {"Participacion_%": "sum"}
    if has_monto:
        agg_dict["Monto_USD"] = "sum"

    grp = df_view.groupby("Categoria", dropna=False).agg(agg_dict).reset_index()

    # === Editor interactivo (con formateo de % y $ en la vista) ===
    df_view_disp = df_view.copy()
    df_view_disp["Participacion_%"] = df_view_disp["Participacion_%"].round(2)

    if has_monto:
        df_view_disp["Monto_USD_fmt"] = df_view_disp["Monto_USD"].apply(
            lambda x: f"${x:,.0f}" if pd.notna(x) else ""
        )
    else:
        df_view_disp["Monto_USD_fmt"] = ""

    st.data_editor(
        df_view_disp[["Item", "Participacion_%", "Categoria", "Monto_USD_fmt"]],
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic",
        column_config={
            "Item": st.column_config.TextColumn("Item", width="medium"),
            "Categoria": st.column_config.SelectboxColumn(
                "Categoría", options=sorted(capex_df["Categoria"].dropna().unique())
            ),
            "Participacion_%": st.column_config.NumberColumn(
                "Participación (%)",
                help="Participación del ítem en el CAPEX",
                format="%.2f %%"
            ),
            "Monto_USD_fmt": st.column_config.TextColumn(
                "Monto (USD)",
                help="Monto calculado = (%/100) × CAPEX total",
                disabled=True
            ),
        }
    )

    # === KPIs
    k1, k2, k3 = st.columns(3)
    with k1: st.metric("Ítem", f"{len(df_view):,}")
    with k2: st.metric("Total USD ", f"${tot_monto:,.0f}" if has_monto and np.isfinite(tot_monto) else "—")
    with k3: st.metric("Suma de % ", f"{df_view['Participacion_%'].sum(skipna=True):,.1f} %")

   

# --- Pie por participación (sobre lo filtrado) ---
st.markdown("#### Participación CAPEX [%]")

fig_capex_pie = px.pie(
    df_view.fillna({"Participacion_%": 0}),
    names="Item",
    values="Participacion_%",
    hole=0.55
)
fig_capex_pie.update_traces(textposition="inside", texttemplate="%{percent:.1%}")
# título fuera + leyenda abajo
fig_capex_pie.update_layout(
    title=None,
    legend=dict(orientation="h", y=-0.2, x=0.0),
    margin=dict(l=10, r=10, t=10, b=80)
)
fig_capex_pie = apply_plotly_theme(fig_capex_pie, "", "", "", height=360)

if has_monto and np.isfinite(tot_monto):
    fig_capex_pie.add_annotation(
        text=f"Total<br>${tot_monto:,.0f}",
        showarrow=False,
        font=dict(size=14, color=GRAY_1),
        x=0.5, y=0.5, xref="paper", yref="paper"
    )

st.plotly_chart(fig_capex_pie, use_container_width=True)

# === Descargas
capex_buf = io.StringIO()
capex_df.to_csv(capex_buf, index=False)
st.download_button("📥 Descargar CAPEXV2 (CSV limpio)", capex_buf.getvalue(),
                   file_name="capexv2.csv", mime="text/csv")

grp_buf = io.StringIO()
grp.to_csv(grp_buf, index=False)
st.download_button("📥 Descargar Subtotales por Categoría (CSV)", grp_buf.getvalue(),
                   file_name="capexv2_subtotales_categoria.csv", mime="text/csv")
# =========================
# CNE · Hoja 1 (URL -> Tabla técnica + Gráficos)
# =========================
st.markdown("---")
st.header("Propuesta Capex para PILOTO")

CNE_URL_DEFAULT = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRb9avOuK8IILVQcc5WKSK9C01pRODcOtO60RNqwo-RstqV3kpEwEzyCM0X5TeEBf9Jje76370ly74d/pub?gid=574902636&single=true&output=csv"
cne_url = st.text_input("URL CNE (Hoja 1 en CSV)", value=CNE_URL_DEFAULT)

def _read_cne_table(url: str) -> pd.DataFrame | None:
    """Lee la hoja CNE (Hoja 1). Devuelve columnas: Item, Categoria, CATE2, Participacion_% (float)."""
    try:
        df = pd.read_csv(url, dtype=str)
    except Exception as e:
        st.error(f"No se pudo leer la hoja CNE: {e}")
        return None

    df.columns = [str(c).strip() for c in df.columns]

    def find_col(keys: list[str]) -> str | None:
        for c in df.columns:
            lc = c.lower()
            if any(k in lc for k in keys):
                return c
        return None

    c_item = find_col(["item", "concepto", "descripcion", "descripción", "nombre"])
    c_cat  = find_col(["categoria", "categoría"])
    c_c2   = find_col(["cate 2", "cate2", "tipo", "montaje", "suministro"])
    c_pct  = find_col(["particip", "%", "norm_%", "porcentaje"])

    if not (c_item and c_pct):
        st.warning("No se detectaron las columnas mínimas (Item y %). Revisa la hoja.")
        return None

    out = pd.DataFrame({
        "Item": df[c_item].astype(str).str.strip(),
        "Participacion_%": df[c_pct].astype(str),
    })
    out["Categoria"] = df[c_cat].astype(str).str.strip() if c_cat else "—"
    out["CATE2"]     = df[c_c2].astype(str).str.strip() if c_c2 else "—"

    # Normalizar % (coma/punto y símbolos)
    out["Participacion_%"] = pd.to_numeric(
        out["Participacion_%"]
          .str.replace("%", "", regex=False)
          .str.replace(",", ".", regex=False)
          .str.replace(r"[^0-9.\-]", "", regex=True),
        errors="coerce"
    )

    out = out.dropna(subset=["Item"]).reset_index(drop=True)

    # Si viene 0–1 => pasar a 0–100
    max_pct = out["Participacion_%"].max(skipna=True)
    if pd.notna(max_pct) and max_pct <= 1.0:
        out["Participacion_%"] = out["Participacion_%"] * 100.0

    out["Participacion_%"] = out["Participacion_%"].clip(lower=0)
    return out

# 1) Leer tabla CNE
cne_df = _read_cne_table(cne_url)
if cne_df is None or not len(cne_df):
    st.stop()

# 2) CAPEX base por 1 turbina
st.markdown("### Desgloce de Capex Piloto")

# Potencia sugerida desde la curva (si existe)
try:
    pot_kw_sugerida = float(pc_df["P_KW"].max())
    if not np.isfinite(pot_kw_sugerida) or pot_kw_sugerida <= 0:
        pot_kw_sugerida = 80.0
except Exception:
    pot_kw_sugerida = 80.0

capex_mode = st.radio(
    "Fuente del CAPEX",
    ["Desde kW × $/kW", "Ingresar monto manual"],
    index=0, horizontal=True,
    help="Para un piloto de 1 turbina, normalmente usa kW × $/kW."
)

col_cap1, col_cap2, col_cap3 = st.columns([1,1,1])

with col_cap1:
    pot_kw_turbina = st.number_input(
        "Potencia de la turbina (kW)",
        min_value=1.0, value=float(pot_kw_sugerida), step=1.0
    )
with col_cap2:
    usd_por_kw = st.number_input(
        "Precio turbina ($/kW)",
        min_value=0.0, value=1700.0, step=10.0
    )
with col_cap3:
    if capex_mode == "Desde kW × $/kW":
        capex_total_input = float(pot_kw_turbina) * float(usd_por_kw)
        st.number_input("CAPEX total del piloto (USD) (auto)", value=float(capex_total_input),
                        step=1000.0, disabled=True)
    else:
        capex_total_input = st.number_input(
            "CAPEX total del piloto (USD) (manual)",
            min_value=0.0,
            value=float(st.session_state.get("capex_total_directo", 0.0)) or 0.0,
            step=1000.0
        )

st.metric("CAPEX 1 turbina", f"${capex_total_input:,.0f}")

# 3) Montos por fila con el CAPEX de 1 turbina
cne_df = cne_df.copy()
if capex_total_input > 0:
    cne_df["Monto_USD"] = (cne_df["Participacion_%"].fillna(0.0) / 100.0) * capex_total_input
else:
    cne_df["Monto_USD"] = np.nan

# =========================
# UI PRO — KPIs + Tablas (a prueba de orden)
# Requiere: cne_df (con Participacion_% y, opcionalmente, Monto_USD) y capex_total_input
# =========================

def _ensure_kpis_and_aggregates(cne_df: pd.DataFrame, capex_total_input: float):
    # KPIs base
    sum_pct = float(cne_df["Participacion_%"].sum(skipna=True))

    by_c2 = (
        cne_df.groupby("CATE2", dropna=False)["Participacion_%"]
              .sum().reset_index()
    )
    pct_suministro = float(
        by_c2.loc[by_c2["CATE2"].str.lower().str.contains("suministro", na=False),
                  "Participacion_%"].sum()
    )
    pct_montaje = float(
        by_c2.loc[by_c2["CATE2"].str.lower().str.contains("montaje", na=False),
                  "Participacion_%"].sum()
    )

    # Subtotales
    agg = {"Participacion_%": "sum"}
    if capex_total_input > 0:
        agg["Monto_USD"] = "sum"

    sub_item = (
        cne_df.groupby("Item", dropna=False)
              .agg(agg).reset_index()
              .sort_values(("Monto_USD" if capex_total_input>0 else "Participacion_%"), ascending=False)
    )
    sub_cat = (
        cne_df.groupby("Categoria", dropna=False)
              .agg(agg).reset_index()
              .sort_values(("Monto_USD" if capex_total_input>0 else "Participacion_%"), ascending=False)
    )
    return sum_pct, pct_suministro, pct_montaje, by_c2, sub_item, sub_cat

# Calcula/recupera todo de forma segura (evita NameError)
sum_pct, pct_suministro, pct_montaje, by_c2, sub_item, sub_cat = _ensure_kpis_and_aggregates(cne_df, capex_total_input)

# ---------- Estilos UI ----------
st.markdown("""
<style>
.kpi-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:10px 0 18px}
.kpi-card{border:1px solid rgba(0,0,0,.07);border-radius:14px;padding:14px 16px;
          background:linear-gradient(180deg,#FAFCFF 0%,#FFFFFF 100%);box-shadow:0 1px 2px rgba(0,0,0,.04)}
.kpi-title{font-size:12px;color:#39424E;letter-spacing:.02em;margin:0 0 6px 0;text-transform:uppercase}
.kpi-value{font-weight:700;font-size:28px;line-height:1.1;color:#101828;margin:0}
.kpi-sub{font-size:12px;color:#6B7280;margin-top:6px}
.badge-ok{display:inline-block;padding:2px 8px;border-radius:999px;font-size:11px;background:#DCFCE7;
          color:#166534;border:1px solid #86EFAC;margin-left:8px}
.badge-warn{display:inline-block;padding:2px 8px;border-radius:999px;font-size:11px;background:#FEF3C7;
            color:#92400E;border:1px solid #FCD34D;margin-left:8px}
[data-testid="stDataEditor"] table{font-size:13px}
[data-testid="stDataEditor"] thead tr th{background:#F8FAFC}
[data-testid="stDataEditor"] tbody tr:nth-child(odd){background:#FCFCFD}
</style>
""", unsafe_allow_html=True)

badge_norm = '<span class="badge-ok">Normalizado</span>' if 99.5 <= sum_pct <= 100.5 else '<span class="badge-warn">Revisar</span>'

# ---------- Tarjetas KPI ----------
st.markdown(f"""
<div class="kpi-grid">
  <div class="kpi-card">
    <p class="kpi-title">CAPEX 1 turbina</p>
    <p class="kpi-value">${capex_total_input:,.0f}</p>
    <div class="kpi-sub">Potencia: {pot_kw_turbina:,.0f} kW · ${usd_por_kw:,.0f}/kW</div>
  </div>
  <div class="kpi-card">
    <p class="kpi-title">Suma de %</p>
    <p class="kpi-value">{sum_pct:,.2f}% {badge_norm}</p>
    <div class="kpi-sub">Objetivo ≈ 100%</div>
  </div>
  <div class="kpi-card">
    <p class="kpi-title">% Suministro</p>
    <p class="kpi-value">{pct_suministro:,.2f}%</p>
    <div class="kpi-sub">CATE2 = Suministro</div>
  </div>
  <div class="kpi-card">
    <p class="kpi-title">% Montaje</p>
    <p class="kpi-value">{pct_montaje:,.2f}%</p>
    <div class="kpi-sub">CATE2 = Montaje</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------- Tablas “Subtotales” (con formato) ----------
st.subheader("Subtotales")

def _fmt_usd(x): 
    return f"${x:,.0f}" if pd.notna(x) and np.isfinite(x) else ""

_item_disp = sub_item.copy()
_cat_disp  = sub_cat.copy()
_item_disp["Participacion_%"] = _item_disp["Participacion_%"].round(2)
_cat_disp["Participacion_%"]  = _cat_disp["Participacion_%"].round(2)

if "Monto_USD" in _item_disp.columns:
    _item_disp["Monto_USD_fmt"] = _item_disp["Monto_USD"].map(_fmt_usd)
if "Monto_USD" in _cat_disp.columns:
    _cat_disp["Monto_USD_fmt"]  = _cat_disp["Monto_USD"].map(_fmt_usd)

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Por Ítem**")
    st.data_editor(
        _item_disp[["Item","Participacion_%"] + (["Monto_USD_fmt"] if "Monto_USD_fmt" in _item_disp.columns else [])],
        use_container_width=True, hide_index=True, disabled=True,
        column_config={
            "Item": st.column_config.TextColumn("Item", width="medium"),
            "Participacion_%": st.column_config.NumberColumn("Participación (%)", format="%.2f %%"),
            "Monto_USD_fmt": st.column_config.TextColumn("Monto (USD)"),
        }
    )
with c2:
    st.markdown("**Por Categoría**")
    st.data_editor(
        _cat_disp[["Categoria","Participacion_%"] + (["Monto_USD_fmt"] if "Monto_USD_fmt" in _cat_disp.columns else [])],
        use_container_width=True, hide_index=True, disabled=True,
        column_config={
            "Categoria": st.column_config.TextColumn("Categoría", width="medium"),
            "Participacion_%": st.column_config.NumberColumn("Participación (%)", format="%.2f %%"),
            "Monto_USD_fmt": st.column_config.TextColumn("Monto (USD)"),
        }
    )


## =========================
# 6) Gráficos — paleta profesional
# =========================
st.subheader("Gráficos")

# Paleta pro (slate + teal, desaturada)
PALETTE_PRO = {
    "ink":   "#1F2937",  # gray-800
    "slate": "#334155",  # slate-700
    "slateL":"#64748B",  # slate-500
    "teal":  "#0F766E",  # teal-700
    "tealL": "#14B8A6",  # teal-500
}

metrica_y    = "Monto_USD" if capex_total_input > 0 else "Participacion_%"
etiqueta_txt = "$%{x:,.0f}" if capex_total_input > 0 else "%{x:.2f}%"
ylab         = "USD" if capex_total_input > 0 else "%"

# ---------------- Barras por Ítem (horizontal) ----------------
_sub_item = sub_item.sort_values(metrica_y, ascending=True)

fig_item = px.bar(
    _sub_item,
    x=metrica_y, y="Item",
    orientation="h",
    text=metrica_y
)
fig_item = apply_plotly_theme(fig_item, "Participación por ítem", "", ylab, height=420)
fig_item.update_traces(
    texttemplate=etiqueta_txt,
    textposition="outside",
    cliponaxis=False,
    opacity=0.95,
    marker=dict(
        color=PALETTE_PRO["slate"],
        line=dict(color="rgba(0,0,0,.15)", width=1.2)
    ),
    hovertemplate="<b>%{y}</b><br>" + ("$%{x:,.0f}" if capex_total_input>0 else "%{x:.2f}%") + "<extra></extra>"
)
fig_item.update_layout(showlegend=False)
fig_item.update_xaxes(title_text=ylab, automargin=True)
fig_item.update_yaxes(title_text="", automargin=True)
st.plotly_chart(fig_item, use_container_width=True)

# ---------------- Donut por CATE2 (Suministro / Montaje) ----------------
# Mapeo de color explícito para nombres comunes (con fallback por minúsculas)
color_map_c2 = {
    "Suministro": PALETTE_PRO["teal"],  "suministro": PALETTE_PRO["teal"],
    "Montaje":    PALETTE_PRO["slate"], "montaje":    PALETTE_PRO["slate"],
}

fig_c2 = px.pie(
    by_c2, names="CATE2", values="Participacion_%",
    hole=0.55, color="CATE2", color_discrete_map=color_map_c2
)
fig_c2 = apply_plotly_theme(fig_c2, "Distribución por CATE2", "", "", height=320)
fig_c2.update_traces(
    textposition="inside",
    texttemplate="%{percent:.1%}",
    marker=dict(line=dict(color="white", width=2))
)
fig_c2.update_layout(legend=dict(orientation="h", y=-0.2, x=0.0))
fig_c2.add_annotation(
    text=f"Total<br>{sum_pct:,.1f}%",
    showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper",
    font=dict(size=14, color=PALETTE_PRO["ink"])
)
st.plotly_chart(fig_c2, use_container_width=True)

# ---------------- Barras por Categoría (horizontal) ----------------
_sub_cat = sub_cat.sort_values(metrica_y, ascending=True)

fig_cat = px.bar(
    _sub_cat,
    x=metrica_y, y="Categoria",
    orientation="h",
    text=metrica_y
)
fig_cat = apply_plotly_theme(fig_cat, "Participación por categoría", "", ylab, height=420)
fig_cat.update_traces(
    texttemplate=etiqueta_txt,
    textposition="outside",
    cliponaxis=False,
    opacity=0.95,
    marker=dict(
        color=PALETTE_PRO["teal"],
        line=dict(color="rgba(0,0,0,.15)", width=1.2)
    ),
    hovertemplate="<b>%{y}</b><br>" + ("$%{x:,.0f}" if capex_total_input>0 else "%{x:.2f}%") + "<extra></extra>"
)
fig_cat.update_layout(showlegend=False)
fig_cat.update_xaxes(title_text=ylab, automargin=True)
fig_cat.update_yaxes(title_text="", automargin=True)
st.plotly_chart(fig_cat, use_container_width=True)

# 7) Descargas
cne_buf = io.StringIO(); cne_df.to_csv(cne_buf, index=False)
st.download_button("📥 Descargar tabla CNE limpia (CSV)", cne_buf.getvalue(),
                   file_name="cne_hoja1_limpia.csv", mime="text/csv")

sub_item_buf = io.StringIO(); sub_item.to_csv(sub_item_buf, index=False)
st.download_button("📥 Descargar subtotales por Ítem (CSV)", sub_item_buf.getvalue(),
                   file_name="cne_subtotales_item.csv", mime="text/csv")

sub_cat_buf = io.StringIO(); sub_cat.to_csv(sub_cat_buf, index=False)
st.download_button("📥 Descargar subtotales por Categoría (CSV)", sub_cat_buf.getvalue(),
                   file_name="cne_subtotales_categoria.csv", mime="text/csv")


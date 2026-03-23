import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from math import pi, sqrt
from contextlib import contextmanager
from html import escape
from pathlib import Path
import base64
import io
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
)
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Paleta fija para TODOS los gráficos Plotly
COLOR_SEQ = [
    "#194BC9",  # azul profundo
    "#eb0a0a",  # verde
    "#74d1f5",  # rosado
    "#eaf63b",  # azul medio
    "#22c55e",  # verde extra
    "#a855f7",  # violeta
]
px.defaults.color_discrete_sequence = COLOR_SEQ

st.set_page_config(page_title="Diseño VAWT – Aerodinámica + Generador GDG-1100", layout="wide")
st.markdown('<div id="top"></div>', unsafe_allow_html=True)


# ====== ESTILO GLOBAL (comentarios + KPIs) ======


st.markdown("""
<style>

.main .block-container {
    padding-top: 0.5rem;
}

.kpi-card {
    background: linear-gradient(135deg, #0E1525 0%, #1A2233 100%);
    border-radius: 12px;
    padding: 0.7rem 1.0rem;       /* MÁS COMPACTO */
    border: 1px solid rgba(255,255,255,0.05);
    box-shadow: 0 2px 8px rgba(0,0,0,0.35);
    transition: 0.15s ease-in-out;
    min-height: 115px;           /* ALTURA REDUCIDA */
    display: flex;
    flex-direction: column;
    justify-content: center;
}

.kpi-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 14px rgba(0,0,0,0.45);
}

.hero-banner {
    width: 100%;
    height: 280px;
    border-radius: 24px;
    background-size: cover;
    background-position: center 20%;
    margin-bottom: 1.6rem;
    box-shadow: 0 18px 35px rgba(15,23,42,0.25);
    border: 1px solid rgba(255,255,255,0.15);
    position: relative;
    overflow: hidden;
}

.kpi-title {
    font-size: 0.65rem;          /* MÁS CHICO */
    text-transform: uppercase;
    letter-spacing: 0.09em;
    color: #8BA2BF;
    margin-bottom: 0.35rem;      /* TEXTO MÁS ARRIBA */
}

.kpi-value {
    font-size: 1.55rem;          /* REDUCIDO */
    font-weight: 700;
    color: #FFFFFF;
    margin-bottom: 0.1rem;
}

.kpi-sub {
    font-size: 0.75rem;
    color: #9BA6B9;
    margin-top: 0.15rem;
}

/* Menos espacio entre filas */
.kpi-container {
    margin-bottom: 0.7rem;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

.comment-box {
    background: #F6F9FC;
    border-left: 6px solid #2B73FF;
    padding: 1rem 1.3rem;
    border-radius: 6px;
    margin-top: 1.2rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
}

.comment-title {
    font-weight: 700;
    font-size: 1rem;
    color: #1A3C78;
    margin-bottom: 0.4rem;
    display: flex;
    align-items: center;
}

.comment-title::before {
    content: " ";
    font-size: 1.1rem;
    margin-right: 0.3rem;
}

.comment-box p {
    font-size: 0.95rem;
    line-height: 1.45;
    color: #333;
}

.sidebar-section {
    margin: 1.2rem 0 0.4rem;
    padding: 0.35rem 0.7rem;
    border-left: 4px solid #2563eb;
    border-radius: 10px;
    background: rgba(37, 99, 235, 0.08);
    font-size: 0.95rem;
    font-weight: 600;
    color: #0f172a;
}

.section-header {
    margin: 2rem 0 1rem;
    padding: 0.85rem 1.1rem;
    border-left: 6px solid #2563eb;
    border: 1px solid rgba(37, 99, 235, 0.25);
    border-radius: 10px;
    background: linear-gradient(90deg, rgba(219,234,254,0.55), rgba(255,255,255,0.9));
    font-size: 1.25rem;
    font-weight: 600;
    color: #0f172a;
    box-shadow: 0 6px 14px rgba(15, 23, 42, 0.08);
}

.section-subheader {
    margin: 1.5rem 0 0.75rem;
    padding: 0.7rem 0.95rem;
    border-left: 5px solid #f97316;
    border: 1px solid rgba(249, 115, 22, 0.35);
    border-radius: 10px;
    background: linear-gradient(90deg, rgba(255, 237, 213, 0.55), rgba(255, 255, 255, 0.9));
    font-size: 1.05rem;
    font-weight: 600;
    color: #7c2d12;
    box-shadow: 0 4px 10px rgba(120, 53, 15, 0.08);
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.question-prompt {
    margin: 0.5rem 0 1rem 0;
    padding: 0.75rem 1rem;
    border-left: 4px solid #f97316;
    background: rgba(249,115,22,0.08);
    border-radius: 6px;
    font-weight: 500;
    color: #7c2d12;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.margin-card {
    background: #0f172a;
    border-radius: 14px;
    padding: 0.75rem 0.9rem;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: inset 0 0 0 1px rgba(255,255,255,0.03), 0 6px 20px rgba(15,23,42,0.25);
    min-height: 115px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    gap: 0.4rem;
}
.margin-card__title {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #cbd5f5;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.margin-card__value {
    font-size: 1.9rem;
    font-weight: 700;
}
.margin-card__badge {
    width: 20px;
    height: 20px;
    border-radius: 50%;
    background: rgba(226,232,240,0.2);
    color: #f8fafc;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 0.65rem;
    cursor: help;
}
.margin-ok .margin-card__value {
    color: #22c55e;
}
.margin-warn .margin-card__value {
    color: #facc15;
}
.margin-danger .margin-card__value {
    color: #f87171;
}
.margin-neutral .margin-card__value {
    color: #e2e8f0;
}
.range-card {
    background: #0d1324;
    border-radius: 14px;
    padding: 0.8rem 1rem;
    border: 1px solid rgba(255,255,255,0.05);
    box-shadow: 0 4px 16px rgba(10,14,26,0.4);
    min-height: 110px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    gap: 0.2rem;
}
.range-card__label {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #cbd5f5;
}
.range-card__value {
    font-size: 2rem;
    font-weight: 700;
    color: #f8fafc;
}
.range-card__sub {
    font-size: 0.85rem;
    color: #9ca3af;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.alert-jump-link {
    display: block;
    font-weight: 600;
    text-align: center;
    padding: 0.6rem 0.4rem;
    border-radius: 8px;
    background: linear-gradient(120deg, #f97316, #f43f5e);
    color: #fff !important;
    text-decoration: none !important;
    margin-bottom: 0.8rem;
    box-shadow: 0 3px 8px rgba(0,0,0,0.25);
}
.alert-jump-link:hover {
    filter: brightness(1.05);
}
.alert-jump-floating {
    position: fixed;
    right: 1.8rem;
    top: 50%;
    transform: translateY(-50%);
    z-index: 999;
    writing-mode: vertical-rl;
    text-orientation: mixed;
    padding: 0.4rem 0.3rem;
    border-radius: 12px;
    background: linear-gradient(180deg, #f97316, #f43f5e);
    color: #fff !important;
    text-decoration: none !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.35);
}
.alert-jump-floating:hover {
    filter: brightness(1.08);
}
.top-jump-floating {
    position: fixed;
    right: 1.8rem;
    bottom: 2.2rem;
    z-index: 999;
    padding: 0.5rem 0.9rem;
    border-radius: 999px;
    background: linear-gradient(120deg, #2563eb, #0ea5e9);
    color: #fff !important;
    text-decoration: none !important;
    font-weight: 600;
    box-shadow: 0 4px 12px rgba(0,0,0,0.35);
}
.top-jump-floating:hover {
    filter: brightness(1.12);
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<a href="#alertas" class="alert-jump-floating">🚨 Alertas</a>',
    unsafe_allow_html=True
)
st.markdown(
    '<a href="#top" class="top-jump-floating">⬆️ Inicio</a>',
    unsafe_allow_html=True
)


def kpi_card(title: str, value: str, subtitle: str, accent: str = "blue") -> None:
    """
    Tarjeta KPI homogénea para todo el dashboard.
    accent: 'blue', 'green', 'orange' o cualquier color hex.
    """
    color_map = {
        "blue":   "#38bdf8",
        "green":  "#22c55e",
        "orange": "#f97316",
        "red":    "#ef4444",
        "yellow": "#eab308",
    }
    accent_color = color_map.get(accent, accent if accent.startswith("#") else "#38bdf8")

    st.markdown(
        f"""
        <div class="kpi-card">
          <div class="kpi-title">{title}</div>
          <div class="kpi-value" style="color:{accent_color};">
            {value}
          </div>
          <div class="kpi-subtitle">
            {subtitle}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def question_prompt(text: str) -> None:
    st.markdown(f"<div class='question-prompt'>❓ {text}</div>", unsafe_allow_html=True)


def section_header(text: str, level: int = 2, anchor: str | None = None) -> None:
    """
    Renderiza un encabezado destacado; level<=2 usa estilo principal, level>=3 usa variante compacta.
    """
    cls = "section-header" if level <= 2 else "section-subheader"
    anchor_attr = f" id='{anchor}'" if anchor else ""
    st.markdown(f"<div class='{cls}'{anchor_attr}>{escape(text)}</div>", unsafe_allow_html=True)


@contextmanager
def section_block(title: str, expanded: bool = True):
    """Crea un bloque plegable con encabezado estilizado."""
    with st.expander(title, expanded=expanded):
        section_header(title)
        yield


def sidebar_section(title: str) -> None:
    st.markdown(f"<div class='sidebar-section'>{escape(title)}</div>", unsafe_allow_html=True)


def comment_box(title: str, body_segments) -> str:
    """Construye un bloque HTML reutilizable para notas de interpretación."""
    if isinstance(body_segments, str):
        body = body_segments
    else:
        body = "".join(body_segments)
    return f"<div class='comment-box'><div class='comment-title'>{title}</div>{body}</div>"


def comment_paragraph(text: str) -> str:
    return f"<p>{text}</p>"


# Recursos compartidos (imágenes hero, etc.)
_HERO_CANDIDATES = [
    (Path(__file__).parent / "assets" / "hero_vawt.jpg").resolve(),
    (Path(__file__).parent / "hero_vawt.jpg").resolve(),
]
_HERO_CANDIDATES[0].parent.mkdir(parents=True, exist_ok=True)


def _hero_path():
    for candidate in _HERO_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def render_hero_banner() -> None:
    """Inserta la imagen panorámica superior de manera responsiva."""
    path = _hero_path()
    if path:
        suffix = path.suffix.lower()
        mime = "image/jpeg"
        if suffix == ".png":
            mime = "image/png"
        elif suffix == ".webp":
            mime = "image/webp"
        encoded = base64.b64encode(path.read_bytes()).decode()
        st.markdown(
            f"<div class='hero-banner' style=\"background-image:url('data:{mime};base64,{encoded}');\"></div>",
            unsafe_allow_html=True,
        )
    else:
        st.info(
            "Agrega la imagen panorámica en `pages/assets/hero_vawt.jpg` "
            "o en `pages/hero_vawt.jpg` para mostrarla en el encabezado.",
            icon="🖼️",
        )


def parse_float_list(text: str) -> list[float]:
    """
    Convierte una cadena separada por comas en una lista de floats.
    Ignora entradas vacías o no numéricas.
    """
    values = []
    for raw in str(text).split(","):
        token = raw.strip()
        if not token:
            continue
        try:
            values.append(float(token))
        except ValueError:
            continue
    return values


st.markdown("""
<style>

.block-container {
    padding-top: 1.2rem !important;     /* Estaba en 5–6rem → reducimos a ~1 */
}

header[data-testid="stHeader"] {
    height: 2rem;
    padding-top: 0rem !important;
    padding-bottom: 0rem !important;
}

</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>

/* Caja de recomendaciones (modo dark, tipo panel técnico) */
.rec-wrapper {
    margin-top: 1.4rem;
    margin-bottom: 1.6rem;
    padding: 1rem 1.3rem;
    border-radius: 12px;
    background: #0F172A;
    border: 1px solid rgba(148,163,184,0.45);
    box-shadow: 0 8px 22px rgba(15,23,42,0.65);
    color: #E5E7EB;
}

/* Cabecera de la sección */
.rec-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 0.7rem;
}

.rec-header-icon {
    font-size: 1.4rem;
}

.rec-header-text-main {
    font-size: 1.05rem;
    font-weight: 600;
}

.rec-header-chip {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    color: #9CA3AF;
}

/* Lista de recomendaciones */
.rec-item {
    font-size: 0.9rem;
    margin-bottom: 0.35rem;
    padding-left: 0.6rem;
    position: relative;
}

.rec-item::before {
    content: "●";
    position: absolute;
    left: -0.1rem;
    top: 0.05rem;
    font-size: 0.6rem;
    color: #22C55E;   /* punto verde tipo “OK técnico” */
}

/* Bloque de fórmulas dentro de la misma caja */
.formula-box {
    margin-top: 1rem;
    padding: 0.8rem 1rem;
    border-radius: 10px;
    background: rgba(15,23,42,0.96);
    border: 1px dashed rgba(148,163,184,0.8);
    font-size: 0.85rem;
}

.formula-title {
    font-weight: 600;
    margin-bottom: 0.45rem;
    color: #E5E7EB;
}

.formula-box ul {
    padding-left: 1.1rem;
    margin: 0;
}

.formula-box li {
    margin-bottom: 0.25rem;
}

</style>
""", unsafe_allow_html=True)



# =========================================================
# Utilidades base
# =========================================================
def rpm_from_tsr(v, D, tsr):
    R = D / 2.0
    return (30.0 / (pi * R)) * tsr * v


def tip_speed(v, tsr):
    return tsr * v


def solidity_int(N, c, R):
    """
    Solidez interna: σ_int = (N·c)/R ≈ π·σ_convencional.
    La solidez convencional es σ_conv = N·c / (π·R).
    """
    return (N * c) / R

def rpm_rotor_mppt(v_array, D, lam_opt, v_cut_in, v_rated, v_cut_out, rpm_rotor_rated):
    """
    Ley de control MPPT por regiones:
    - v < v_cut_in            -> rotor parado (rpm = 0)
    - v_cut_in ≤ v ≤ v_rated  -> MPPT: λ ≈ λ_opt  → rpm ∝ v
    - v_rated < v ≤ v_cut_out -> potencia limitada: rpm ≈ rpm_rotor_rated
    - v > v_cut_out           -> rotor parado (rpm = 0)
    """
    R = D / 2.0
    v_array = np.asarray(v_array, dtype=float)

    # rpm que mantiene λ = λ_opt (MPPT puro)
    rpm_mppt = (30.0 / (pi * R)) * lam_opt * v_array

    # iniciamos todo en 0 (parado)
    rpm = np.zeros_like(v_array)

    # Región MPPT (λ ≈ λ_opt)
    mask_reg2 = (v_array >= v_cut_in) & (v_array <= v_rated)
    rpm[mask_reg2] = rpm_mppt[mask_reg2]

    # Región potencia limitada (rpm constante)
    mask_reg3 = (v_array > v_rated) & (v_array <= v_cut_out)
    rpm[mask_reg3] = rpm_rotor_rated

    # v < cut-in o v > cut-out → rpm = 0
    return rpm


# =========================================================
# Modelo Cp(λ) con efectos de perfil de pala
# =========================================================
def build_cp_params(
    lam_opt_base=2.6,
    cmax_base=0.33,
    shape=1.0,
    sigma=0.24,
    helical=True,
    helix_angle_deg=60.0,      # 👈 NUEVO PARÁMETRO
    endplates=True,
    trips=True,
    struts_perf=True,
    airfoil_thickness=18.0,
    symmetric=True,
    pitch_deg=0.0,
):
    """
    Modelo paramétrico para Cp(λ) incluyendo:
    - Solidez σ
    - Helicoidal (con ángulo), end-plates, trips, struts perfilados
    - Perfil de pala: espesor relativo, simetría, ángulo de calaje
    - Efectos upwind / downwind (dynamic stall lumped)
    """
    lam_opt = lam_opt_base
    cmax    = cmax_base

    # -------------------------------
    # 0) Factor helicoidal (0–1)
    # -------------------------------
    # φ = 0° → f_h = 0  (pala recta)
    # φ = 90° → f_h = 1 (helicoidal "plena")
    helix_angle_deg = float(np.clip(helix_angle_deg, 0.0, 90.0))
    helix_factor = helix_angle_deg / 90.0

    # 1) Solidez: más σ → Cp↑ pero λ_opt↓
    lam_opt -= 0.30 * (sigma - 0.20)
    cmax    += 0.05 * (sigma - 0.20)

    # 2) Configuración global del rotor
    #    Aquí es donde la hélice entra en Cp_max y λ_opt
    if helical:
        # Cp_max(φ) = Cp_max,0 * (1 + k_Cp * f_h)
        cmax    += 0.03 * helix_factor
        # λ_opt(φ) = λ_opt,0 * (1 + k_λ * f_h) (lo aproximamos sumando)
        lam_opt += 0.10 * helix_factor

    if endplates:
        cmax += 0.01
    if trips:
        cmax += 0.015
    if not struts_perf:
        cmax -= 0.03

    # 3) Efectos del perfil: espesor relativo
    delta_t = (airfoil_thickness - 18.0) / 18.0
    drag_factor = 1.0 + 0.40 * max(delta_t, 0.0)      # >18% => más drag
    lam_opt *= (1.0 - 0.15 * delta_t)
    cmax    *= (1.0 - 0.25 * delta_t) / drag_factor

    # 4) Simetría vs asimétrico
    if not symmetric:
        cmax *= 1.08

    # 5) Pitch (calaje) y stall efectivo
    pitch_abs = abs(pitch_deg)
    stall_factor = np.exp(- (pitch_abs / 7.0) ** 2)   # α_char ~ 7°
    cmax *= stall_factor
    lam_opt *= (1.0 - 0.03 * pitch_abs / 5.0)

    # 6) Dynamic stall / upwind vs downwind
    f_up = 1.0
    f_down = 0.85 if symmetric else 0.80

    if helical:
        # f_up(φ)   = f_up,0   * (1 + k_up   * f_h)
        # f_down(φ) = f_down,0 * (1 + k_down * f_h)
        f_up   *= 1.0 + 0.03 * helix_factor
        f_down *= 1.0 + 0.05 * helix_factor

    f_avg = 0.5 * (f_up + f_down)
    if f_avg <= 0:
        f_avg = 1.0
    f_up_norm   = f_up   / f_avg
    f_down_norm = f_down / f_avg

    # 7) Límites físicos razonables
    lam_opt = float(np.clip(lam_opt, 1.6, 3.5))
    cmax    = float(np.clip(cmax,   0.15, 0.42))

    return {
        "lam_opt": lam_opt,
        "cmax":    cmax,
        "shape":   shape,
        "f_up":    f_up_norm,
        "f_down":  f_down_norm,
        "airfoil": {
            "t_rel":        airfoil_thickness,
            "symmetric":    symmetric,
            "pitch_deg":    pitch_deg,
            "stall_factor": stall_factor,
            "drag_factor":  drag_factor,
        },
        "helical": {
            "active":         helical,
            "helix_angle_deg": helix_angle_deg,
            "helix_factor":   helix_factor,
        }
    }



def cp_components(lambda_val, params):
    lam_opt = params["lam_opt"]
    cmax    = params["cmax"]
    shape   = params["shape"]
    f_up    = params.get("f_up", 1.0)
    f_down  = params.get("f_down", 1.0)

    lam = np.asarray(lambda_val, dtype=float)
    x = np.maximum(lam, 1e-6) / lam_opt

    cp_base = cmax * x * np.exp(1 - x) ** shape
    cp_base = np.clip(cp_base, 0.0, 0.5)

    f_avg = 0.5 * (f_up + f_down)
    if f_avg <= 0:
        f_avg = 1.0

    cp_up   = cp_base * (f_up   / f_avg)
    cp_down = cp_base * (f_down / f_avg)
    cp_avg  = cp_base

    return cp_avg, cp_up, cp_down


def cp_model(lambda_val, params):
    cp_avg, _, _ = cp_components(lambda_val, params)
    return cp_avg


def cp_curve_for_plot(cp_params):
    lam_vals = np.linspace(1.0, 4.0, 200)
    cp_avg, cp_up, cp_down = cp_components(lam_vals, cp_params)
    return pd.DataFrame({
        "λ":           lam_vals,
        "Cp_prom":     cp_avg,
        "Cp_upwind":   cp_up,
        "Cp_downwind": cp_down,
    })
# =========================================================
# Polar genérica Lift–Drag del perfil (modelo simplificado)
# =========================================================
def build_lift_drag_polar(t_rel: float, symmetric: bool):
    """
    Genera un polar Cl(α), Cd(α) y Cl/Cd(α) simplificado:
    - α en [-10°, 20°]
    - Pendiente dCl/dα ≈ 0.11 1/deg
    - α0 ≈ 0° simétrico, ≈ -2° camberado
    - Cd0 aumenta con espesor relativo
    - k_ind fija (drag inducido ~ Cl^2)
    """
    alpha_deg = np.linspace(-10.0, 20.0, 61)
    alpha0 = 0.0 if symmetric else -2.0          # α de sustentación nula

    # Pendiente de Cl (aprox. 2π rad ≈ 0.11 /deg)
    cl_slope = 0.11
    cl_lin = cl_slope * (alpha_deg - alpha0)

    # Stall suave usando saturación tipo tanh
    stall_deg = 12.0 if symmetric else 10.0
    cl_max_ref = cl_slope * (stall_deg - alpha0)
    cl_max = cl_max_ref * (1.0 if symmetric else 1.1)
    Cl = cl_max * np.tanh(cl_lin / max(cl_max, 1e-3))

    # Drag: Cd = Cd0 + k * Cl^2
    base_cd0 = 0.01 + 0.002 * (t_rel - 12.0) / 10.0
    base_cd0 = float(np.clip(base_cd0, 0.008, 0.04))
    k_ind = 0.02
    Cd = base_cd0 + k_ind * (Cl ** 2)

    ClCd = np.divide(Cl, Cd, out=np.zeros_like(Cl), where=(Cd > 0))

    return pd.DataFrame({
        "alpha_deg": alpha_deg,
        "Cl": Cl,
        "Cd": Cd,
        "ClCd": ClCd,
    })




# Potencia aerodinámica → eje generador (aplica solo pérdidas mecánicas)
def power_to_generator(v, D, H, lambda_eff, rho, eta_mec, cp_params):
    A   = D * H
    v   = np.asarray(v, dtype=float)
    lam = np.asarray(lambda_eff, dtype=float)

    cp_arr = cp_model(lam, cp_params)     # Cp(λ_efectiva)
    P_a = 0.5 * rho * A * (v ** 3) * cp_arr       # W rotor
    P_m = P_a * eta_mec                           # W eje generador
    return P_a, P_m, cp_arr


# Weibull
def weibull_pdf(v, k, c):
    return (k / c) * (v / c) ** (k - 1) * np.exp(-(v / c) ** k)


def aep_from_weibull(v_grid, P_grid_W, k, c):
    pdf = weibull_pdf(v_grid, k, c)
    Pw  = P_grid_W * pdf
    P_mean = np.trapz(Pw, v_grid)                 # W
    AEP_kWh = P_mean * 8760.0 / 1000.0           # kWh/año
    return AEP_kWh, P_mean

def alpha_cycle_deg(theta_deg, lam, pitch_deg=0.0):
    """
    Ángulo de ataque cinemático (modelo 2D ideal).
    alpha(θ) = atan2(sinθ, λ - cosθ) + pitch
    Retorna α en grados.
    """
    th = np.deg2rad(np.asarray(theta_deg, dtype=float))
    alpha_rad = np.arctan2(np.sin(th), (lam - np.cos(th)))
    return np.rad2deg(alpha_rad) + float(pitch_deg)



# =========================================================
# PDF
# =========================================================
def build_pdf_report(df_view, figs_dict, kpi_text=""):
    """
    Genera un PDF en memoria con:
    - Portada simple
    - Comentario de alto nivel
    - Tabla (vista actual, primeras 15 filas)
    - Gráficos clave como imágenes, cada uno con título + interpretación
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    # Portada
    story.append(Paragraph("Reporte técnico – VAWT + Generador", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph("Síntesis para ingeniería de alto nivel", styles["Heading2"]))
    story.append(Spacer(1, 18))

    if kpi_text:
        story.append(Paragraph(kpi_text, styles["BodyText"]))
        story.append(Spacer(1, 18))

    # Tabla principal (vista actual)
    story.append(Paragraph(
        "Tabla de resultados (vista actual – primeras 15 filas)",
        styles["Heading2"]
    ))
    story.append(Spacer(1, 6))

    df_short = df_view.head(10).reset_index(drop=True)

    if not df_short.empty:
        df_horizontal = df_short.T.reset_index()
        df_horizontal = df_horizontal.rename(columns={"index": "Variable"})

        bin_labels = []
        for idx in range(df_short.shape[0]):
            v_val = df_short.loc[idx].get("v (m/s)")
            if pd.notna(v_val):
                bin_labels.append(f"Bin {idx+1} | v={v_val:.1f} m/s")
            else:
                bin_labels.append(f"Bin {idx+1}")

        df_horizontal.columns = ["Variable"] + bin_labels
        table_data = [df_horizontal.columns.tolist()] + df_horizontal.values.tolist()
    else:
        table_data = [["Sin datos"]]

    # Mejorar legibilidad de los encabezados (salto de línea Bin / velocidad)
    header_style = styles["BodyText"].clone("HeaderTable")
    header_style.textColor = colors.whitesmoke
    header_style.fontName = "Helvetica-Bold"

    header_cells = []
    for col_name in table_data[0]:
        text = str(col_name)
        if text.startswith("Bin"):
            text = text.replace(" | ", "<br/>")
        header_cells.append(Paragraph(text, header_style))
    table_data[0] = header_cells

    # Ajustar ancho de columnas
    page_width, _ = A4
    table_width = page_width - 2 * cm
    n_cols = len(table_data[0])
    col_widths = [table_width / max(n_cols, 1)] * n_cols

    table = Table(table_data, colWidths=col_widths, repeatRows=1)

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0b1120")),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.whitesmoke),
        ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, 0), 8),
        ("FONTSIZE",   (0, 1), (-1, -1), 7),
        ("GRID",       (0, 0), (-1, -1), 0.25, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.whitesmoke, colors.lightgrey]),
    ]))

    story.append(table)
    story.append(Spacer(1, 12))
    story.append(PageBreak())

    interpretaciones = {
        "rpm rotor / generador vs velocidad de viento":
            "Muestra cómo crecen las rpm del rotor y del generador según la ley de control por regiones.",
        "Curva de potencia (según vista seleccionada)":
            "Relaciona potencia aerodinámica, mecánica y eléctrica para validar la integración aero–generador.",
        "Par en rotor / generador":
            "Dimensiona ejes y confirma que no se exceden los límites IEC ni los de ficha del generador.",
        "Momento flector en unión pala–struts":
            "Evolución del momento flector combinado (torque + fuerza centrífuga) para validar límites FEM/IEC en la raíz de pala.",
        "Cp equivalente por etapa":
            "Localiza la etapa con mayor degradación de rendimiento (rotor, tren mecánico o electrónica).",
        "Pérdidas por etapa":
            "Cuantifica dónde se concentran las pérdidas para priorizar rediseños.",
        "Corriente estimada vs velocidad de viento":
            "Asegura compatibilidad eléctrica y evita sobrecorrientes.",
        "Frecuencias 1P / 3P del rotor":
            "Chequea resonancias entre cargas periódicas y modos estructurales.",
        "Curva Cp(λ) – promedio y componentes":
            "Verifica que el TSR de control coincida con el máximo Cp disponible.",
        "Ruido estimado vs velocidad de viento":
            "Valida el cumplimiento acústico en el receptor crítico.",
        "🌬️ Distribución de viento vs curva de potencia":
            "Mezcla Weibull del sitio con la curva de potencia para derivar AEP y factor de planta."
    }

    if isinstance(figs_dict, dict):
        figs_iter = figs_dict.items()
    else:
        figs_iter = figs_dict

    # Gráficos
    for title, fig in figs_iter:
        story.append(Paragraph(title, styles["Heading2"]))
        story.append(Spacer(1, 6))

        png_bytes = fig.to_image(format="png", scale=2)
        img_buffer = io.BytesIO(png_bytes)
        img = Image(img_buffer, width=480, height=280)
        story.append(img)
        story.append(Spacer(1, 6))

        if title in interpretaciones:
            story.append(Paragraph(interpretaciones[title], styles["BodyText"]))
            story.append(Spacer(1, 18))

        story.append(PageBreak())

    doc.build(story)
    pdf_value = buffer.getvalue()
    buffer.close()
    return pdf_value


# =========================================================
# Curvas de generadores axiales (80 kW y 10 kW)
# =========================================================

# --- GDG-1100 – 80 kW (lo que ya tenías) ---
GDG_POWER_TABLE_80 = pd.DataFrame(
    [
        (0, 0),
        (24, 2),
        (48, 3),
        (72, 7),
        (96, 12),
        (120, 19),
        (144, 28),
        (168, 38),
        (192, 50),
        (216, 64),
        (240, 80),
        (264, 97),
    ],
    columns=["rpm", "P_kW"],
)

GDG_VOLT_TABLE_80 = pd.DataFrame(
    [
        (0, 0),
        (24, 40),
        (48, 80),
        (72, 120),
        (96, 160),
        (120, 200),
        (144, 240),
        (168, 280),
        (192, 320),
        (216, 360),
        (240, 400),
        (264, 440),
    ],
    columns=["rpm", "V_LL"],
)

GDG_RATED_RPM_80   = 240.0
GDG_RATED_PkW_80   = 80.0
GDG_RATED_VLL_80   = 400.0
GDG_RATED_I_80     = 115.0
GDG_RATED_T_Nm_80  = 3460.0
GDG_POLES_80       = 48
GDG_OMEGA_RATED_80 = 2 * pi * GDG_RATED_RPM_80 / 60.0
GDG_KE_DEFAULT_80  = GDG_RATED_VLL_80 / GDG_OMEGA_RATED_80
GDG_KT_DEFAULT_80  = GDG_RATED_T_Nm_80 / GDG_RATED_I_80

# --- GDG-860 – 10 kW (desde la ficha adjunta) ---
GDG_POWER_TABLE_10 = pd.DataFrame(
    [
        (0, 0),
        (7, 0.2),
        (14, 0.4),
        (21, 0.9),
        (28, 1.5),
        (35, 2.4),
        (42, 3.5),
        (49, 4.7),
        (56, 6.2),
        (63, 8.0),
        (70, 10.0),
        (77, 12.1),
    ],
    columns=["rpm", "P_kW"],
)

GDG_VOLT_TABLE_10 = pd.DataFrame(
    [
        (0, 0),
        (7, 40),
        (14, 80),
        (21, 120),
        (28, 160),
        (35, 200),
        (42, 240),
        (49, 280),
        (56, 320),
        (63, 360),
        (70, 400),
        (77, 440),
    ],
    columns=["rpm", "V_LL"],
)

GDG_RATED_RPM_10   = 70.0
GDG_RATED_PkW_10   = 10.0
GDG_RATED_VLL_10   = 400.0
GDG_RATED_I_10     = 14.0
GDG_RATED_T_Nm_10  = 1483.0     # según ficha GDG-860
GDG_POLES_10       = 20
GDG_OMEGA_RATED_10 = 2 * pi * GDG_RATED_RPM_10 / 60.0
GDG_KE_DEFAULT_10  = GDG_RATED_VLL_10 / GDG_OMEGA_RATED_10
GDG_KT_DEFAULT_10  = GDG_RATED_T_Nm_10 / GDG_RATED_I_10

# --- Catálogo común de generadores para la UI ---
GENERATORS = {
    "GDG_80k": {
        "label": "GDG-1100 – 80 kW",
        "P_nom_kW": GDG_RATED_PkW_80,
        "rpm_nom": GDG_RATED_RPM_80,
        "V_LL_nom": GDG_RATED_VLL_80,
        "I_nom": GDG_RATED_I_80,
        "T_nom": GDG_RATED_T_Nm_80,
        "poles": GDG_POLES_80,
        "Ke_default": GDG_KE_DEFAULT_80,
        "Kt_default": GDG_KT_DEFAULT_80,
        "power_table": GDG_POWER_TABLE_80,
        "volt_table": GDG_VOLT_TABLE_80,
    },
    "GDG_10k": {
        "label": "GDG-860 – 10 kW",
        "P_nom_kW": GDG_RATED_PkW_10,
        "rpm_nom": GDG_RATED_RPM_10,
        "V_LL_nom": GDG_RATED_VLL_10,
        "I_nom": GDG_RATED_I_10,
        "T_nom": GDG_RATED_T_Nm_10,
        "poles": GDG_POLES_10,
        "Ke_default": GDG_KE_DEFAULT_10,
        "Kt_default": GDG_KT_DEFAULT_10,
        "power_table": GDG_POWER_TABLE_10,
        "volt_table": GDG_VOLT_TABLE_10,
    },
}


def interp_curve(x, x_tab, y_tab):
    """
    Interpolación lineal sencilla con extrapolación plana
    (mantiene el primer y último valor fuera de rango).
    """
    x = np.asarray(x)
    return np.interp(x, x_tab, y_tab, left=y_tab[0], right=y_tab[-1])



# =========================================================
# UI – Entradas
# =========================================================
render_hero_banner()
st.title("🔬 Plataforma técnica VAWT – Aerodinámica · Tren mecánico · Generador")

# =========================================================
# Catálogo de perfiles aerodinámicos (NACA + típicos eólicos)
# =========================================================

AIRFOIL_LIBRARY = {
    # ---- SIMÉTRICOS (buenos para Darrieus / VAWT) ----
    "NACA 0012": {
        "t_rel": 12.0,
        "symmetric": True,
        "descripcion": "Perfil simétrico clásico, drag bajo y uso extendido en turbinas de eje horizontal."
    },
    "NACA 0015": {
        "t_rel": 15.0,
        "symmetric": True,
        "descripcion": "Simétrico, compromiso entre arrastre y rigidez. Muy usado en prototipos VAWT."
    },
    "NACA 0018": {
        "t_rel": 18.0,
        "symmetric": True,
        "descripcion": "Más grueso, mayor rigidez estructural, buen comportamiento en Re moderados."
    },
    "NACA 0021": {
        "t_rel": 21.0,
        "symmetric": True,
        "descripcion": "Perfil robusto; buena opción para palas con mayores cargas y fabricación FRP."
    },
    "NACA 0024": {
        "t_rel": 24.0,
        "symmetric": True,
        "descripcion": "Muy grueso, prioriza rigidez y fatiga sobre rendimiento aerodinámico máximo."
    },

    # ---- CAMBERADOS (más lift, más sensibilidad a ángulo) ----
    "NACA 2412": {
        "t_rel": 12.0,
        "symmetric": False,
        "descripcion": "Camber moderado, usado históricamente en alas; mayor Cl/Cd pero más sensible a pitch."
    },
    "NACA 4412": {
        "t_rel": 12.0,
        "symmetric": False,
        "descripcion": "Muy utilizado en eólica HAWT; buen rendimiento en Cl, mayor complejidad en control."
    },
    "NACA 4415": {
        "t_rel": 15.0,
        "symmetric": False,
        "descripcion": "Similar al 4412 pero más grueso; buena combinación de aerodinámica y rigidez."
    },
    "NACA 4418": {
        "t_rel": 18.0,
        "symmetric": False,
        "descripcion": "Perfil con camber y espesor altos; pensado para cargas importantes y alta sustentación."
    },

    # ---- Ejemplo “VAWT-friendly” genérico ----
    "NACA 0022 (VAWT FRP)": {
        "t_rel": 22.0,
        "symmetric": True,
        "descripcion": "Perfil grueso y simétrico como el que estás usando en el piloto; robusto y tolerante a stall dinámico."
    },
}


with st.sidebar:

    st.markdown('<a href="#alertas" class="alert-jump-link">🚨 Ir a alertas</a>', unsafe_allow_html=True)

    sidebar_section("1️⃣ Geometría y pala")
    # Geometría
    with st.expander("Geometría", expanded=False):
        D = st.number_input("Diámetro D [m]",  min_value=2.0, value=11.0, step=0.5)
        H = st.number_input("Altura H [m]",    min_value=2.0, value=18.0, step=0.5)
        N = st.number_input("Nº de palas N",   min_value=2,   value=3, step=1)
        c = st.number_input("Cuerda c [m]",    min_value=0.1, value=0.80, step=0.05)
    
    # Perfil de pala / masa
    with st.expander("Perfil de pala / masa", expanded=False):
        # === Modo de selección de perfil ===
        modo_perfil = st.radio(
            "Modo de selección de perfil",
            ["Catálogo NACA", "Personalizado"],
            horizontal=True
        )

        if modo_perfil == "Catálogo NACA":
            # Ordenamos para que quede bonito en el selector
            airfoil_keys = sorted(AIRFOIL_LIBRARY.keys())

            airfoil_name = st.selectbox(
                "Perfil NACA",
                airfoil_keys,
                index=airfoil_keys.index("NACA 0022 (VAWT FRP)") if "NACA 0022 (VAWT FRP)" in airfoil_keys else 0,
            )

            af_data = AIRFOIL_LIBRARY[airfoil_name]
            t_rel = af_data["t_rel"]
            is_symmetric = af_data["symmetric"]
            tipo_perfil = "Simétrico" if is_symmetric else "Asimétrico"

            st.caption(
                f"e/c ≈ {t_rel:.0f} % – {af_data['descripcion']}"
            )

            # Permitimos ajustar finamente el pitch aunque el perfil venga predefinido
            pitch_deg = st.slider(
                "Ángulo de calaje (pitch) [°]",
                min_value=-10.0, max_value=10.0,
                value=0.0,
                step=0.25,
                help="Controla el pitch del perfil seleccionado y refresca α(θ) en tiempo real."
            )

        else:
            # === Modo completamente personalizado ===
            airfoil_name = st.text_input("Perfil (ej: NACA 0018)", "NACA 0022")
            tipo_perfil  = st.selectbox("Tipo de perfil", ["Simétrico", "Asimétrico"])
            is_symmetric = (tipo_perfil == "Simétrico")

            t_rel = st.number_input(
                "Espesor relativo e/c [%]",
                min_value=8.0,
                max_value=40.0,
                value=22.0,
                step=1.0
            )

            pitch_deg = st.slider(
                "Ángulo de calaje (pitch) [°]",
                min_value=-10.0, max_value=10.0,
                value=0.0,
                step=0.25,
                help="Controla el pitch del perfil seleccionado y refresca α(θ) en tiempo real."
            )

        # ---- Parámetros de masa / geometría helicoidal (comunes a ambos modos) ----
        st.markdown("**Tweaks aerodinámicos / masa**")
        helical     = st.checkbox("Helicoidal 60–90°", True, help="Activa la pala helicoidal y aplica su ángulo en Cp(λ).")
        endplates   = st.checkbox("End-plates / winglets", False)
        trips       = st.checkbox("Trips / micro-tabs", False)
        struts_perf = st.checkbox("Struts perfilados (0012)", False)
        m_blade = st.number_input(
            "Masa por pala [kg]",
            min_value=10.0,
            value=180.0,
            step=10.0
        )

        helix_angle_deg = st.number_input(
            "Ángulo helicoidal pala [°]",
            min_value=0.0, max_value=90.0,
            value=60.0,
            step=5.0
        )

        helix_rad = np.deg2rad(helix_angle_deg)
        blade_span = H / max(np.cos(helix_rad), 1e-3)
        st.caption(f"Longitud de pala estimada ≈ {blade_span:.1f} m (helix {helix_angle_deg:.0f}°)")

        struts_per_blade = st.number_input(
            "N° de struts por pala",
            min_value=1,
            value=3,
            step=1,
            help="Cantidad de vigas/brazos que conectan cada pala con la torre; se usa para repartir el momento flector."
        )

        # Configuración detallada por strut
        default_distances = np.array([13.0, 11.0, 13.0], dtype=float)
        default_weights = np.full(int(struts_per_blade), 1.0 / max(struts_per_blade, 1))

        if "strut_dist_input" not in st.session_state or st.session_state.get("strut_dist_count") != int(struts_per_blade):
            st.session_state["strut_dist_input"] = ", ".join(f"{d:.1f}" for d in default_distances)
            st.session_state["strut_dist_count"] = int(struts_per_blade)

        if "strut_weight_input" not in st.session_state or st.session_state.get("strut_weight_count") != int(struts_per_blade):
            st.session_state["strut_weight_input"] = ", ".join(f"{w:.2f}" for w in default_weights)
            st.session_state["strut_weight_count"] = int(struts_per_blade)

        strut_dist_input = st.text_input(
            "Distancias de struts [m] (separadas por coma)",
            key="strut_dist_input",
            help="Ejemplo: 17, 9, 2  → representa las distancias desde el eje a cada viga."
        )
        strut_weight_input = st.text_input(
            "Ponderación relativa por strut",
            key="strut_weight_input",
            help="Normaliza cómo reparte el momento cada viga (por defecto iguales)."
        )

        strut_distances = parse_float_list(strut_dist_input)
        if not strut_distances:
            strut_distances = default_distances.tolist()
        if len(strut_distances) < int(struts_per_blade):
            strut_distances.extend([strut_distances[-1]] * (int(struts_per_blade) - len(strut_distances)))
        elif len(strut_distances) > int(struts_per_blade):
            strut_distances = strut_distances[: int(struts_per_blade)]

        strut_weights = parse_float_list(strut_weight_input)
        if len(strut_weights) != len(strut_distances) or not strut_weights:
            strut_weights = [1.0] * len(strut_distances)

        total_weight = sum(strut_weights)
        if total_weight <= 0:
            total_weight = len(strut_weights)
            strut_weights = [1.0] * len(strut_distances)
        lever_arm_pala = float(np.dot(strut_distances, strut_weights) / total_weight)
        weights_norm = [w / total_weight for w in strut_weights]

        df_struts = pd.DataFrame({
            "Strut #": list(range(1, len(strut_distances) + 1)),
            "Distancia [m]": np.round(strut_distances, 2),
            "Peso relativo": np.round(weights_norm, 3),
        })
        st.dataframe(df_struts, hide_index=True, use_container_width=True)
        st.caption(
            f"Brazo efectivo calculado ≈ {lever_arm_pala:.2f} m (suma de ponderaciones = {sum(weights_norm):.2f})."
        )
        st.caption(
            "Tip: ingresa las alturas reales de unión (ej. 2, 9, 17 m para una pala de 18 m) y asigna pesos mayores a "
            "los struts que capturan más carga según tu FEM. Si todos comparten la misma palanca, deja distancias iguales "
            "y solo ajusta la ponderación."
        )

    with st.expander("Propiedades estructurales avanzadas", expanded=False):
        section_modulus_root = st.number_input(
            "Módulo resistente raíz W [m³]",
            min_value=0.001,
            value=0.075,
            step=0.005,
            help="Define la capacidad a flexión en la unión pala–struts. Valores mayores implican perfiles más robustos."
        )
        sigma_y_pala_mpa = st.number_input(
            "σ_y pala / raíz [MPa]",
            min_value=50.0,
            value=180.0,
            step=5.0,
            help="Límite de fluencia o admisible del laminado / unión en la raíz."
        )
        strut_area_cm2 = st.number_input(
            "Área efectiva strut [cm²]",
            min_value=5.0,
            value=40.0,
            step=1.0,
            help="Área metálica equivalente por strut para estimar esfuerzos axiales."
        )
        sigma_allow_strut_mpa = st.number_input(
            "σ admisible strut [MPa]",
            min_value=50.0,
            value=250.0,
            step=5.0,
            help="Tensión axial permitida en los struts (considera material + soldaduras)."
        )
        safety_target = st.number_input(
            "Factor de seguridad objetivo",
            min_value=1.0,
            value=1.5,
            step=0.1,
            help="Usado como referencia para sombrear los gráficos de stress."
        )
        show_guides = st.checkbox("Mostrar guías y rangos sugeridos", value=False)
        if show_guides:
            st.markdown("""
**Raíz FRP / aluminio (pilotos 10–60 kW)**
- Módulo resistente W: 0.04–0.10 m³ según espesor del laminado.
- σ_y pala: 120–200 MPa (laminados infundidos + insertos metálicos).

**Struts tubulares de acero ASTM A500**
- Área efectiva típica: 30–60 cm² (tubos 120–180 mm, t=5–8 mm).
- σ admisible: 200–260 MPa (fluencia 345 MPa con FS≈1.5).

**Struts de aluminio 6061-T6**
- Área efectiva: 45–80 cm² (perfiles más gruesos para compensar módulo).
- σ admisible: 140–180 MPa (fluencia 240 MPa / FS 1.3–1.5).

**Recomendaciones**
- FS objetivo ≥1.3 para operación normal, ≥1.7 si el sitio tiene ráfagas severas.
- Si no tienes FEM, arranca por W ≈ (π·c·t³)/6 para el bloque de raíz y ajusta con datos de pruebas.
""")

    sidebar_section("2️⃣ Operación y entorno")
    # Precalcular λ_opt estimado con la configuración actual
    R_preview = D / 2.0
    sig_int_preview = solidity_int(N, c, R_preview)
    lam_ctrl_default = 2.5
    cp_preview = None
    try:
        cp_preview = build_cp_params(
            lam_opt_base=2.6,
            cmax_base=0.33,
            shape=1.0,
            sigma=sig_int_preview,
            helical=helical,
            helix_angle_deg=helix_angle_deg,
            endplates=endplates,
            trips=trips,
            struts_perf=struts_perf,
            airfoil_thickness=t_rel,
            symmetric=is_symmetric,
            pitch_deg=pitch_deg,
        )
        lam_ctrl_default = float(cp_preview.get("lam_opt", lam_ctrl_default))
    except Exception:
        pass

    # Operación / control
    with st.expander("Operación / Control", expanded=False):

        control_mode = st.radio(
            "Modo de control",
            options=["MPPT (λ constante)", "RPM fija (sin MPPT)"],
            index=0,
            help="MPPT mantiene λ≈constante en Región 2; sin MPPT usa rpm fija entre cut-in y cut-out.",
        )

        if control_mode == "MPPT (λ constante)":
            lam_opt_ctrl = st.number_input(
                "TSR objetivo λ (control)",
                min_value=1.5,
                max_value=5.0,
                value=lam_ctrl_default,
                step=0.01,
                help="Setpoint MPPT utilizado para la ley rpm–v en Región 2. Por defecto igual al λ_opt estimado."
            )
            tsr_ctrl = lam_opt_ctrl
        else:
            lam_opt_ctrl = lam_ctrl_default
            tsr_ctrl = None
            rpm_v_exp = st.slider(
                "Exponente rpm vs viento (sin MPPT)",
                min_value=0.50,
                max_value=1.50,
                value=0.85,
                step=0.05,
                help="rpm_gen ∝ (v/v_rated)^a. a=1 mantiene TSR casi constante; a≠1 hace variar TSR y Cp."
            )
            st.caption(
                "Modo sin MPPT: la TSR y el Cp resultan de la velocidad del viento y la rpm fijada, "
                "no de un setpoint constante."
            )

        rho = st.number_input("Densidad aire ρ [kg/m³]", min_value=1.0, value=1.225, step=0.025)
        mu  = st.number_input(
            "Viscosidad dinámica μ [Pa·s]",
            min_value=1.0e-5, max_value=3.0e-5,
            value=1.8e-5, step=0.1e-5, format="%.6f"
        )
        v_cut_in  = st.number_input("v_cut-in [m/s]",  min_value=0.5, value=3.0, step=0.5)
        v_rated   = st.number_input("v_rated [m/s]",   min_value=v_cut_in + 0.5, value=12.0, step=0.5)
        v_cut_out = st.number_input("v_cut-out [m/s]", min_value=v_rated + 0.5, value=20.0, step=0.5)

    # Rango de vientos
    with st.expander("Rango de vientos / Muestreo", expanded=False):
        v_min  = st.number_input("v mín [m/s]", min_value=0.5, value=4.0, step=0.5)
        v_max  = st.number_input("v máx [m/s]", min_value=v_min+0.5, value=20.0, step=0.5)

    # Ruido aeroacústico
    with st.expander("Ruido aeroacústico (dB)", expanded=False):
        use_noise = st.checkbox("Estimar ruido (Lw / Lp)", True)
        Lw_ref_dB = st.number_input(
            "Lw_ref @ v_rated [dB]",
            min_value=0.0, max_value=150.0,
            value=100.0, step=1.0,
            help="Nivel de potencia sonora de referencia a v_rated"
        )
        r_obs = st.number_input(
            "Distancia observador [m]",
            min_value=1.0, max_value=1000.0,
            value=50.0, step=5.0
        )
        n_noise = st.number_input(
            "Exponente n (U_tip^n)",
            min_value=1.0, max_value=8.0,
            value=5.0, step=0.5,
            help="Sensibilidad del ruido a la velocidad de punta"
        )

    sidebar_section("3️⃣ Tren de potencia y electrónica")
    # --- Tren de potencia / Generador ---
    with st.expander("Tren de potencia / Generador", expanded=False):

        # 0) Selección de modelo de generador
        gen_key = st.selectbox(
            "Modelo generador axial-flux",
            options=list(GENERATORS.keys()),
            format_func=lambda k: GENERATORS[k]["label"],
            index=0,
        )
        GEN = GENERATORS[gen_key]

        # --- Alias globales para compatibilidad con el resto del código ---
        GDG_RATED_T_Nm = GEN["T_nom"]
        GDG_RATED_I    = GEN["I_nom"]
        GDG_RATED_RPM  = GEN["rpm_nom"]

        st.markdown(
            f"""
**Generador seleccionado**

- Modelo: `{GEN['label']}`
- P_nom: **{GEN['P_nom_kW']:.1f} kW**
- rpm_nom: **{GEN['rpm_nom']:.0f} rpm**
- V_LL_nom: **{GEN['V_LL_nom']:.0f} Vac**
- I_nom: **{GEN['I_nom']:.1f} A**
- T_nom: **{GEN['T_nom']:.0f} N·m**
- Nº de polos: **{GEN['poles']}**
"""
        )

        # rpm sugerida por aerodinámica (referencia si no hay MPPT)
        tsr_sugerida = lam_opt_ctrl if control_mode == "MPPT (λ constante)" else lam_ctrl_default
        rpm_sugerida = float(rpm_from_tsr(v_rated, D, tsr_sugerida))
        st.caption(
            f"rpm rotor rated sugerida por diseño aerodinámico (TSR ref y v_rated): "
            f"≈ **{rpm_sugerida:.1f} rpm**"
        )

        usar_rpm_auto = st.checkbox(
            "Usar rpm sugerida (TSR y v_rated)",
            value=True,
            help="Si está activo, la rpm nominal del rotor se toma del diseño aerodinámico."
        )

        if usar_rpm_auto:
            rpm_rotor_rated = rpm_sugerida
            st.write(f"rpm_rotor_rated (auto) = **{rpm_rotor_rated:.1f} rpm**")
        else:
            rpm_rotor_rated = st.number_input(
                "rpm rotor rated",
                min_value=10.0,
                value=float(rpm_sugerida),
                step=1.0,
            )

        # Generador + relación G
        rpm_gen_rated = st.number_input(
            "rpm gen rated",
            min_value=10.0,
            value=float(GEN["rpm_nom"]),
            step=1.0,
        )

        auto_G = st.checkbox("Calcular G con rpm rated", True)
        if auto_G:
            G = rpm_gen_rated / max(rpm_rotor_rated, 1e-6)
            st.write(f"**G (calc)** = {G:.2f}")
        else:
            G = st.number_input(
                "Relación G = rpm_gen/rpm_rotor",
                min_value=1.0,
                value=6.0,
                step=0.05,
            )

        # Eficiencias mecánicas
        eta_bear = st.number_input("η rodamientos", min_value=0.90, value=0.98, step=0.005)
        eta_gear = st.number_input("η caja",       min_value=0.85, value=0.96, step=0.005)

        # Parámetros del generador
        poles_total    = st.number_input("N° de polos (total)", min_value=4, value=int(GEN["poles"]), step=2)
        eta_gen_max    = st.number_input("η_gen máx (tope)", min_value=0.80, value=0.93, step=0.005)
        Ke_vsr_default = st.number_input("Ke [V·s/rad]", min_value=1.0, value=float(GEN["Ke_default"]), step=0.1)
        Kt_nm_per_A    = st.number_input("Kt [N·m/A]", min_value=1.0, value=float(GEN["Kt_default"]), step=0.1)

        st.caption("Puedes subir una curva alternativa del generador (cols: rpm, P_kW, V_LL).")
        gen_csv = st.file_uploader("CSV rendimiento generador", type=["csv"])

        eta_elec = st.number_input("η electrónica (rect+inv)", min_value=0.90, value=0.975, step=0.005)

        P_nom_kW  = st.number_input(
            "P_nom [kW]",
            min_value=1.0,
            value=float(GEN["P_nom_kW"]),
            step=1.0,
        )
        T_gen_max = st.number_input(
            "T_gen máx [N·m] (opcional)",
            min_value=0.0,
            value=float(GEN["T_nom"]),
            step=50.0,
        )

    with st.expander("Electrónica / red avanzada", expanded=False):
        pf_setpoint = st.slider(
            "PF operativo (cos φ)",
            min_value=0.80,
            max_value=1.00,
            value=0.95,
            step=0.01,
            help="Setpoint de control de factor de potencia que usará la electrónica."
        )
        pf_min_grid = st.slider(
            "PF mínimo exigido por red",
            min_value=0.80,
            max_value=1.00,
            value=0.90,
            step=0.01,
        )
        thd_cap_pct = st.number_input(
            "THD estimada (filtro LCL) [%]",
            min_value=1.0,
            value=3.0,
            step=0.5,
            help="Distorsión armónica total esperada en bornes de red tras filtros."
        )
        thd_req_pct = st.number_input(
            "THD límite normativa [%]",
            min_value=2.0,
            value=5.0,
            step=0.5,
        )
        lvrt_cap_voltage_pu = st.number_input(
            "LVRT tensión soportada [pu]",
            min_value=0.05,
            max_value=1.00,
            value=0.15,
            step=0.01,
            help="Profundidad de hueco (pu) que el inversor soporta sin dispararse."
        )
        lvrt_req_voltage_pu = st.number_input(
            "LVRT tensión requerida [pu]",
            min_value=0.05,
            max_value=1.00,
            value=0.20,
            step=0.01,
            help="Requisito del código de red (normalmente 0.2–0.3 pu)."
        )
        lvrt_cap_time_ms = st.number_input(
            "LVRT tiempo soportado [ms]",
            min_value=50.0,
            value=180.0,
            step=5.0,
        )
        lvrt_req_time_ms = st.number_input(
            "LVRT tiempo requerido [ms]",
            min_value=50.0,
            value=150.0,
            step=5.0,
        )
        I_inv_thermal_A = st.number_input(
            "Corriente térmica inversor [A]",
            min_value=50.0,
            value=140.0,
            step=1.0,
            help="Corriente RMS máxima continua que soporta el inversor."
        )
        V_dc_nom = st.number_input(
            "Tensión DC nominal [V]",
            min_value=400.0,
            value=750.0,
            step=10.0,
        )
        I_dc_nom = st.number_input(
            "Corriente DC nominal [A]",
            min_value=20.0,
            value=120.0,
            step=5.0,
        )

    sidebar_section("4️⃣ Normativa, recurso y datos")
    # --- IEC 61400-2 – límites de diseño ---
    with st.expander("Límites IEC 61400-2 (diseño)", expanded=False):
        rpm_rotor_max_iec = st.number_input(
            "rpm_rotor máx IEC",
            min_value=10.0,
            value=40.0,
            step=1.0,
            help="Límite estructural de rpm del rotor definido por IEC 61400-2 (fatiga, estabilidad)."
        )
        T_rotor_max_iec = st.number_input(
            "T_rotor máx IEC [N·m]",
            min_value=1000.0,
            value=20000.0,
            step=500.0,
            help="Torque máximo admisible en el eje rotor según diseño estructural IEC-61400-2."
        )
        v_shutdown_iec = st.number_input(
            "v_shutdown IEC [m/s]",
            min_value=v_rated,
            value=v_cut_out,
            step=0.5,
            help="Velocidad de viento a la cual el sistema debe ejecutar parada segura (shutdown)."
        )
        g_max_pala_iec = st.number_input(
            "Aceleración radial máx en pala [g]",
            min_value=5.0,
            value=25.0,
            step=1.0,
            help="Máximo n° de g admisible en la raíz de la pala según criterio estructural/FEM."
        )
        M_base_max_iec = st.number_input(
            "Momento flector máx en raíz [kN·m]",
            min_value=10.0,
            value=350.0,
            step=10.0,
            help="Límite estructural de momento flector en la raíz de la pala / base de torre."
        )

    # Weibull
    with st.expander("Weibull", expanded=False):
        k_w = st.number_input("k (forma)",  min_value=1.0, value=2.0, step=0.1)
        c_w = st.number_input("c (escala) [m/s]", min_value=2.0, value=7.5, step=0.5)

    # Datos piloto (SCADA) para calibración
    with st.expander("Datos piloto (SCADA)", expanded=False):
        file_scada = st.file_uploader(
            "CSV SCADA (viento, potencia, rpm, corriente)",
            type=["csv"],
            help="Sube un CSV con columnas de viento, potencia y opcionalmente rpm/corriente.",
        )

        if file_scada is not None:
            df_scada = pd.read_csv(file_scada)
            st.session_state["df_scada_raw"] = df_scada

            st.caption(f"Columnas detectadas: {', '.join(df_scada.columns.astype(str))}")

            cols = df_scada.columns.tolist()

            # Heurística simple para defaults
            def guess_col(substr, default_idx=0):
                substr = substr.lower()
                for i, c in enumerate(cols):
                    if substr in str(c).lower():
                        return i
                return default_idx

            v_col = st.selectbox(
                "Columna velocidad viento [m/s]",
                cols,
                index=guess_col("viento"),
            )
            P_col = st.selectbox(
                "Columna potencia [kW]",
                cols,
                index=guess_col("pot"),
            )
            rpm_rotor_col = st.selectbox(
                "Columna rpm rotor (opcional)",
                ["(ninguna)"] + cols,
                index=0,
            )
            rpm_gen_col = st.selectbox(
                "Columna rpm generador (opcional)",
                ["(ninguna)"] + cols,
                index=0,
            )
            I_col = st.selectbox(
                "Columna corriente [A] (opcional)",
                ["(ninguna)"] + cols,
                index=0,
            )

            st.session_state["scada_map"] = {
                "v": v_col,
                "P": P_col,
                "rpm_rotor": None if rpm_rotor_col == "(ninguna)" else rpm_rotor_col,
                "rpm_gen":  None if rpm_gen_col   == "(ninguna)" else rpm_gen_col,
                "I":        None if I_col          == "(ninguna)" else I_col,
            }

            st.caption("La calibración se mostrará en el cuerpo principal cuando se complete la simulación.")

        

# =========================================================
# Cálculos base
# =========================================================
R   = D / 2.0
A   = D * H
sig_int = solidity_int(N, c, R)
sig_conv = sig_int / pi  # solidez convencional
eta_mec = eta_bear * eta_gear

cp_params = build_cp_params(
    lam_opt_base=2.6,
    cmax_base=0.33,
    shape=1.0,
    sigma=sig_int,
    helical=helical,
    helix_angle_deg=helix_angle_deg,   # 👈 AQUÍ ENTRA EL ÁNGULO
    endplates=endplates,
    trips=trips,
    struts_perf=struts_perf,
    airfoil_thickness=t_rel,
    symmetric=is_symmetric,
    pitch_deg=pitch_deg,
)

# λ óptimo aerodinámico que entrega el modelo Cp(λ)
lambda_opt_teo = cp_params["lam_opt"]

# λ que usará el control MPPT para la ley rpm–v en región 2
lambda_mppt = lam_opt_ctrl if control_mode == "MPPT (λ constante)" else None

if control_mode == "MPPT (λ constante)" and abs(lambda_mppt - lambda_opt_teo) > 0.05:
    st.warning(
        f"λ_control ({lambda_mppt:.2f}) difiere del λ óptimo aerodinámico estimado ({lambda_opt_teo:.2f}). "
        "Operarás fuera de Cp_max a menos que alinees TSR de control y geometría."
    )


# Grid de vientos
v_grid = np.arange(v_min, v_max + 1e-9, 0.5 if v_max - v_min > 1 else 0.1)

# Ley de operación por regiones:
# En MPPT usamos λ_mppt (igualado a λ_opt_teo para que el control sea óptimo).
if control_mode == "MPPT (λ constante)":
    rpm_rotor = rpm_rotor_mppt(
        v_array=v_grid,
        D=D,
        lam_opt=lam_opt_ctrl,
        v_cut_in=v_cut_in,
        v_rated=v_rated,
        v_cut_out=v_cut_out,
        rpm_rotor_rated=rpm_rotor_rated,
    )
    rpm_gen = rpm_rotor * G
else:
    # Sin MPPT: rpm_gen ∝ (v/v_rated)^a para forzar TSR variable.
    rpm_gen = np.zeros_like(v_grid)
    mask_reg2 = (v_grid >= v_cut_in) & (v_grid <= v_rated)
    if v_rated > 0:
        rpm_gen[mask_reg2] = rpm_gen_rated * (v_grid[mask_reg2] / v_rated) ** rpm_v_exp
    mask_reg3 = (v_grid > v_rated) & (v_grid <= v_cut_out)
    rpm_gen[mask_reg3] = rpm_gen_rated
    rpm_rotor = rpm_gen / max(G, 1e-6)

# rpm nominal coherente con el λ_mppt utilizado
rpm_rated_val = (
    rpm_from_tsr(v_rated, D, lambda_mppt)
    if lambda_mppt is not None
    else np.nan
)

# Chequeo de consistencia entre rpm_rotor_rated y la ley MPPT en v_rated
if control_mode == "MPPT (λ constante)":
    rpm_rated_ctrl = float(np.interp(v_rated, v_grid, rpm_rotor))
    if abs(rpm_rotor_rated - rpm_rated_ctrl) > 5:
        st.warning(
            f"⚠️ rpm_rotor_rated ({rpm_rotor_rated:.1f} rpm) difiere de la rpm MPPT @ v_rated "
            f"({rpm_rated_ctrl:.1f} rpm). Revisa consistencia entre diseño aerodinámico, λ_opt y control MPPT."
        )


omega_rot = 2 * pi * rpm_rotor / 60.0
omega_gen = 2 * pi * rpm_gen   / 60.0

# TSR efectiva λ(v) y U_tip
lambda_eff = np.zeros_like(v_grid, dtype=float)
mask_v = v_grid > 0
lambda_eff[mask_v] = (omega_rot[mask_v] * R) / v_grid[mask_v]
U_tip = lambda_eff * v_grid

# TSR de referencia para UI (setpoint MPPT o valor resultante @ v_rated)
if control_mode == "MPPT (λ constante)":
    tsr_ref = float(tsr_ctrl) if tsr_ctrl is not None else float(lambda_mppt)
else:
    tsr_ref = float(np.interp(v_rated, v_grid, lambda_eff)) if v_grid.size else np.nan

# Potencias con Cp(λ_efectiva)
P_aero_W, P_mec_gen_W, cp_used = power_to_generator(v_grid, D, H, lambda_eff, rho, eta_mec, cp_params)

# Curvas reales del generador seleccionado (o CSV alternativo)
if gen_csv is not None:
    df_gen = pd.read_csv(gen_csv)
    if not {"rpm", "P_kW", "V_LL"}.issubset(df_gen.columns):
        st.error("El CSV debe tener columnas: rpm, P_kW, V_LL")
        st.stop()
    tab_power = df_gen[["rpm", "P_kW"]].sort_values("rpm").reset_index(drop=True)
    tab_volt  = df_gen[["rpm", "V_LL"]].sort_values("rpm").reset_index(drop=True)
else:
    tab_power = GEN["power_table"][["rpm", "P_kW"]].sort_values("rpm").reset_index(drop=True)
    tab_volt  = GEN["volt_table"][["rpm", "V_LL"]].sort_values("rpm").reset_index(drop=True)


P_gen_curve_W = interp_curve(rpm_gen, tab_power["rpm"].values, tab_power["P_kW"].values) * 1000.0
V_LL_curve    = interp_curve(rpm_gen, tab_volt["rpm"].values,  tab_volt["V_LL"].values)

# Torques: cargas aero vs eje generador
T_rotor_Nm = np.divide(P_aero_W, np.maximum(omega_rot, 1e-6))
T_gen_raw  = np.divide(P_mec_gen_W, np.maximum(omega_gen, 1e-6))

# Fuerza centrípeta / aceleración radial por pala en cada bin
F_centripetal_series = m_blade * R * (omega_rot ** 2)
g_per_blade_series = np.divide(
    F_centripetal_series,
    max(m_blade * 9.81, 1e-3),
    out=np.zeros_like(F_centripetal_series),
    where=(m_blade > 0)
)
M_root_series_Nm = np.divide(T_rotor_Nm, max(N, 1)) + F_centripetal_series * lever_arm_pala
M_strut_series_Nm = np.divide(M_root_series_Nm, max(struts_per_blade, 1))

# Tensiones aproximadas en raíz y struts (usa parámetros estructurales del panel)
W_root = max(section_modulus_root, 1e-6)  # m^3
sigma_root_MPa = (M_root_series_Nm / W_root) / 1e6
allow_root_MPa = sigma_y_pala_mpa / max(safety_target, 1e-6)
margin_root = np.divide(
    (allow_root_MPa - sigma_root_MPa),
    max(allow_root_MPa, 1e-6),
    out=np.zeros_like(sigma_root_MPa),
    where=np.isfinite(sigma_root_MPa),
)

strut_area_m2 = max(strut_area_cm2 * 1e-4, 1e-9)
F_strut_series_N = np.divide(M_strut_series_Nm, max(lever_arm_pala, 1e-3))
sigma_strut_MPa = (F_strut_series_N / strut_area_m2) / 1e6
allow_strut_MPa = sigma_allow_strut_mpa / max(safety_target, 1e-6)
margin_strut = np.divide(
    (allow_strut_MPa - sigma_strut_MPa),
    max(allow_strut_MPa, 1e-6),
    out=np.zeros_like(sigma_strut_MPa),
    where=np.isfinite(sigma_strut_MPa),
)

if T_gen_max > 0:
    T_gen_Nm = np.minimum(T_gen_raw, T_gen_max)
else:
    T_gen_Nm = T_gen_raw

# Potencia mecánica que realmente puede transmitir el eje rápido tras limitar par
P_mec_to_gen_W = np.minimum(P_mec_gen_W, T_gen_Nm * omega_gen)

# Control por regiones (similar a control de rpm): limita potencia en Región 3 y apaga fuera de operación
mask_reg2_ctrl = (v_grid >= v_cut_in) & (v_grid <= v_rated)
mask_reg3_ctrl = (v_grid > v_rated) & (v_grid <= v_cut_out)
mask_off_ctrl  = (v_grid < v_cut_in) | (v_grid > v_cut_out)

eta_chain = max(eta_gen_max * eta_elec, 1e-6)
P_mec_cap_nom_W = (P_nom_kW * 1000.0) / eta_chain if P_nom_kW > 0 else np.inf
P_mec_cap_curve_W = P_gen_curve_W / max(eta_gen_max, 1e-6)
P_mec_cap_W = np.minimum(P_mec_cap_nom_W, P_mec_cap_curve_W)

P_mec_to_gen_W = np.where(
    mask_reg3_ctrl,
    np.minimum(P_mec_to_gen_W, P_mec_cap_W),
    P_mec_to_gen_W,
)
P_mec_to_gen_W = np.where(mask_off_ctrl, 0.0, P_mec_to_gen_W)

# Actualizar par reportado con la potencia limitada por control
T_gen_Nm = np.minimum(T_gen_Nm, np.divide(P_mec_to_gen_W, np.maximum(omega_gen, 1e-6)))

# Usar potencia mecánica controlada como referencia de eje generador
P_mec_gen_W = P_mec_to_gen_W.copy()

# Retroalimentar límite al resto de etapas
P_el_gen_W = np.minimum(P_mec_to_gen_W * eta_gen_max, P_gen_curve_W)
P_el_ac    = P_el_gen_W * eta_elec
P_el_ac_clip = np.minimum(P_el_ac, P_nom_kW * 1000.0)

# Eficiencia instantánea del generador (considerando límite de par)
eta_gen_curve = np.divide(
    P_el_gen_W,
    np.maximum(P_mec_to_gen_W, 1.0),
    out=np.zeros_like(P_el_gen_W),
    where=(P_mec_to_gen_W > 0)
)
eta_gen_curve = np.clip(eta_gen_curve, 0.0, eta_gen_max)


# Frecuencia eléctrica
p_pairs = poles_total / 2.0
f_e_Hz  = p_pairs * rpm_gen / 60.0

PF = pf_setpoint

# Corriente estimada en bornes del generador (antes de electrónica/clipping)
V_eff = np.maximum(V_LL_curve, 1.0)
P_for_I = P_el_gen_W.copy()
I_from_power = np.where(
    V_LL_curve < 10.0,
    0.0,
    np.divide(
        P_for_I,
        np.sqrt(3) * V_eff * PF,
        out=np.zeros_like(P_for_I),
        where=(P_for_I > 0)
    )
)
Kt_safe = max(float(Kt_nm_per_A), 1e-6)
I_from_torque = T_gen_Nm / Kt_safe
I_A = np.maximum(I_from_power, I_from_torque)
max_I_inv = float(np.nanmax(I_A)) if I_A.size else 0.0

V_LL_from_Ke = Ke_vsr_default * omega_gen
dc_link_capacity_W = max(V_dc_nom * I_dc_nom, 1e3)
dc_util_series = np.divide(
    P_el_gen_W,
    dc_link_capacity_W,
    out=np.zeros_like(P_el_gen_W),
    where=(dc_link_capacity_W > 0)
)

# Cp equivalente por etapa
P_out_W = P_el_ac_clip
Cp_aero = np.divide(
    P_aero_W,
    0.5 * rho * A * (v_grid ** 3),
    out=np.zeros_like(v_grid), where=(v_grid > 0)
)
Cp_shaft = np.divide(
    P_mec_to_gen_W,
    0.5 * rho * A * (v_grid ** 3),
    out=np.zeros_like(v_grid), where=(v_grid > 0)
)
Cp_el = np.divide(
    P_out_W,
    0.5 * rho * A * (v_grid ** 3),
    out=np.zeros_like(v_grid), where=(v_grid > 0)
)

# Reynolds en pala (aprox. con velocidad relativa local)
U_rel = np.sqrt((lambda_eff * v_grid) ** 2 + v_grid ** 2)   # componente tangencial + flujo incident
Re_mid = np.zeros_like(v_grid)
if mu > 0:
    Re_mid = rho * U_rel * c / mu

# Ruido aeroacústico
Lw_dB = np.full_like(v_grid, np.nan, dtype=float)
Lp_dB = np.full_like(v_grid, np.nan, dtype=float)

if use_noise:
    if v_grid[0] <= v_rated <= v_grid[-1]:
        U_tip_ref = float(np.interp(v_rated, v_grid, U_tip))
    else:
        U_tip_ref = float(U_tip[-1])

    U_ratio = np.divide(
        U_tip,
        max(U_tip_ref, 1e-3),
        out=np.ones_like(U_tip),
        where=(U_tip_ref > 0)
    )

    Lw_dB = Lw_ref_dB + 10.0 * n_noise * np.log10(
        np.maximum(U_ratio, 1e-6)
    )

    Lp_dB = Lw_dB - 20.0 * np.log10(max(r_obs, 1.0)) - 11.0

# Frecuencias 1P / 3P
f_1P = rpm_rotor / 60.0
f_3P = 3.0 * f_1P

# =========================================================
# Tabla principal
# =========================================================
df = pd.DataFrame({
    "v (m/s)":           np.round(v_grid, 3),
    "rpm_rotor":         np.round(rpm_rotor, 2),
    "rpm_gen":           np.round(rpm_gen, 2),
    "λ_efectiva":        np.round(lambda_eff, 2),
    "U_tip (m/s)":       np.round(U_tip, 2),
    "Cp(λ_efectiva)":    np.round(cp_used, 3),
    "Cp_aero_equiv":     np.round(Cp_aero, 3),
    "Cp_shaft_equiv":    np.round(Cp_shaft, 3),
    "Cp_el_equiv":       np.round(Cp_el, 3),
    "Re (mid-span)":     np.round(Re_mid, 0),
    "P_aero (kW)":       np.round(P_aero_W / 1000.0, 2),
    "P_mec_gen (kW)":    np.round(P_mec_gen_W / 1000.0, 2),
    "P_gen_curve (kW)":  np.round(P_gen_curve_W / 1000.0, 2),
    "η_gen (curve)":     np.round(eta_gen_curve, 3),
    "V_LL (V)":          np.round(V_LL_curve, 1),
    "V_LL (Ke) [V]":     np.round(V_LL_from_Ke, 1),
    "f_e (Hz)":          np.round(f_e_Hz, 1),
    "f_1P (Hz)":         np.round(f_1P, 2),
    "f_3P (Hz)":         np.round(f_3P, 2),
    "T_rotor (N·m)":     np.round(T_rotor_Nm, 0),
    "T_gen (N·m)":       np.round(T_gen_Nm, 0),
    "F_cen/pala (kN)":   np.round(F_centripetal_series / 1000.0, 2),
    "a_cen (g)":         np.round(g_per_blade_series, 2),
    "M_base (kN·m)":     np.round(M_root_series_Nm / 1000.0, 2),
    "M_por_strut (kN·m)": np.round(M_strut_series_Nm / 1000.0, 2),
    "sigma_root (MPa)":  np.round(sigma_root_MPa, 2),
    "sigma_strut (MPa)": np.round(sigma_strut_MPa, 2),
    "margen_root (%)":   np.round(margin_root * 100.0, 1),
    "margen_strut (%)":  np.round(margin_strut * 100.0, 1),
    "P_el (kW)":         np.round(P_el_ac / 1000.0, 2),
    "P_out (clip) kW":   np.round(P_el_ac_clip / 1000.0, 2),
    "I_est (A)":         np.round(I_A, 1),
    "Duty_DC (%)":       np.round(dc_util_series * 100.0, 1),
    "Lw (dB)":           np.round(Lw_dB, 1),
    "Lp_obs (dB)":       np.round(Lp_dB, 1),
})
# =========================
# PÉRDIDAS POR ETAPA [W]
# =========================
P_loss_mec_W  = np.maximum(P_aero_W    - P_mec_to_gen_W, 0.0)
P_loss_gen_W  = np.maximum(P_mec_to_gen_W - P_el_gen_W,  0.0)
P_loss_elec_W = np.maximum(P_el_gen_W  - P_el_ac,     0.0)
P_loss_clip_W = np.maximum(P_el_ac     - P_el_ac_clip,0.0)

# Pasar a kW y guardar en el DataFrame
df["P_loss_mec (kW)"]  = np.round(P_loss_mec_W  / 1000.0, 2)
df["P_loss_gen (kW)"]  = np.round(P_loss_gen_W  / 1000.0, 2)
df["P_loss_elec (kW)"] = np.round(P_loss_elec_W / 1000.0, 2)
df["P_loss_clip (kW)"] = np.round(P_loss_clip_W / 1000.0, 2)

st.markdown("""
<style>

/* ===== Tabs del panel de KPIs ===== */
[data-testid="stTabs"] button {
    font-weight: 600;
    font-size: 0.9rem;          /* un poco más chico */
    padding-top: 0.5rem;
    padding-bottom: 0.5rem;
}

[data-testid="stTabs"] button[aria-selected="true"] {
    border-bottom: 3px solid #f97316 !important;
    color: #f97316 !important;
}

/* ===== Tarjetas KPI (25% más pequeñas) ===== */
.kpi-card {
    background: radial-gradient(circle at top left,#020617,#020617 55%,#02091b);
    border-radius: 16px;
    padding: 0.75rem 1.05rem;          /* antes 1.0 / 1.4 */
    border: 1px solid rgba(148,163,184,0.35);
    box-shadow: 0 14px 30px rgba(15,23,42,0.55);
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    gap: 0.30rem;
    height: 100%;
}

.kpi-title {
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 0.65rem;                /* antes 0.75 */
    color: #9ca3af;
}

.kpi-value {
    font-size: 1.65rem;                /* antes 2.2rem */
    font-weight: 700;
    color: #f9fafb;
}

.kpi-subtitle {
    font-size: 0.8rem;                 /* antes 0.9rem */
    color: #9ca3af;
}

/* Menos espacio vertical entre elementos del panel */
.element-container:has(.kpi-card) {
    margin-bottom: 0.6rem !important;
}

</style>
""", unsafe_allow_html=True)


# INICIO DEL WRAPPER
st.markdown('<div id="kpi-wrapper">', unsafe_allow_html=True)


# =========================================================
# Panel técnico de KPIs
# =========================================================
omega_rated = 2 * pi * rpm_rotor_rated / 60.0
P_rated_W   = P_nom_kW * 1000.0
T_rated     = P_rated_W / omega_rated if omega_rated > 0 else 0.0
k_mppt      = T_rated / (omega_rated ** 2) if omega_rated > 0 else 0.0

mass_total_blades = N * m_blade
I_blades = N * m_blade * (R ** 2)
F_centripetal_per_blade = m_blade * R * (omega_rated ** 2)
g_per_blade_rated = (R * (omega_rated ** 2) / 9.81) if omega_rated > 0 else 0.0
M_root_rated = (T_rated / max(N, 1)) + F_centripetal_per_blade * lever_arm_pala
M_strut_rated = M_root_rated / max(struts_per_blade, 1)

Re_8 = np.interp(8.0, v_grid, Re_mid) if (v_grid[0] <= 8.0 <= v_grid[-1]) else Re_mid[-1]
Re_max = Re_mid[-1] if len(Re_mid) > 0 else 0.0

section_header("📊 Panel técnico de KPIs")

tab_pala, tab_rotor, tab_tren = st.tabs(
    ["Pala & cargas inerciales", "Rotor & aerodinámica", "Tren de potencia"]
)


with tab_rotor:
    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card("Área barrida A = D·H", f"{A:.1f} m²", "Superficie efectiva de captura de viento")
    with c2:
        kpi_card(
            "Solidez σ_int = (N·c)/R",
            f"{sig_int:.2f}",
            f"σ_conv ≈ {sig_conv:.2f} (N·c/(πR))"
        )
    with c3:
        tsr_title = "TSR objetivo λ" if control_mode == "MPPT (λ constante)" else "TSR @ v_rated"
        tsr_sub = (
            "Setpoint de control aerodinámico"
            if control_mode == "MPPT (λ constante)"
            else "TSR resultante en nominal"
        )
        kpi_card(tsr_title, f"{tsr_ref:.2f}", tsr_sub)

    c4, c5, c6 = st.columns(3)
    with c4:
        kpi_card("λ_opt estimado", f"{cp_params['lam_opt']:.2f}", "Óptimo teórico de Cp(λ) para esta geometría")
    with c5:
        kpi_card("Cp_max estimado", f"{cp_params['cmax']:.2f}", "Rendimiento aerodinámico máximo esperado")
    with c6:
        kpi_card("U_tip @ v_max", f"{U_tip[-1]:.1f} m/s", "Velocidad de punta – ruido y fatiga")

    c7, c8 = st.columns(2)
    with c7:
        kpi_card(
            "λ_efectiva @ v_rated",
            f"{np.interp(v_rated, v_grid, lambda_eff):.2f}",
            "Qué tan cerca opera del λ_opt en nominal"
        )
    with c8:
        kpi_card(
            "Cp_el_equiv @ v_rated",
            f"{np.interp(v_rated, v_grid, Cp_el):.3f}",
            "Eficiencia global viento → eléctrica en nominal"
        )

    st.caption(
        "Rotor dimensionado para trabajar cercano a λ_opt y Cp_max con la geometría y solidez definidas. "
        "λ_efectiva refleja la ley de control por regiones (cut-in / rated / cut-out)."
    )

with tab_tren:
    t1, t2, t3 = st.columns(3)
    with t1:
        kpi_card("G = rpm_gen / rpm_rotor", f"{G:.2f}", "Relación de transmisión del tren de potencia")
    with t2:
        kpi_card("Polos totales", f"{int(poles_total)}", "Define rango de frecuencia eléctrica del generador")
    with t3:
        kpi_card("T_rated", f"{T_rated:,.0f} N·m", "Par objetivo a potencia nominal")

    t4, t5, t6 = st.columns(3)
    with t4:
        kpi_card("k_MPPT", f"{k_mppt:.3e} N·m·s²", "Constante de control T = k·ω² para MPPT")
    with t5:
        kpi_card("η_mec = η_rodam·η_caja", f"{eta_mec:.3f}", "Eficiencia combinada del tren mecánico")
    with t6:
        kpi_card("η_elec (rect+inv)", f"{eta_elec:.3f}", "Eficiencia típica electrónica de potencia")

    st.caption("Estos parámetros definen el comportamiento del tren de potencia y el ajuste de control MPPT para el piloto.")

with tab_pala:
    p1, p2, p3 = st.columns(3)
    with p1:
        kpi_card("Perfil aerodinámico", airfoil_name, "Base para performance y curva Cp(λ)")
    with p2:
        kpi_card("Tipo de perfil", tipo_perfil, "Simétrico vs asimétrico – stall y lift")
    with p3:
        kpi_card("Espesor relativo e/c", f"{t_rel:.1f} %", "Influye en drag, rigidez y rango de Re")

    p4, p5, p6 = st.columns(3)
    with p4:
        kpi_card("Masa total palas", f"{mass_total_blades:,.0f} kg", "Carga inercial rotativa")
    with p5:
        kpi_card("Inercia palas I ≈ N·m·R²", f"{I_blades:,.0f} kg·m²", "Respuesta dinámica del rotor")
    with p6:
        kpi_card(
            "F CEN. / PALA ≈ m·R·ω²",
            f"{F_centripetal_per_blade/1000:.1f} kN",
            f"≈ {g_per_blade_rated:.1f} g @ rpm_rated"
        )

    p7, p8, p9 = st.columns(3)
    with p7:
        kpi_card("Re @ 8 m/s ≈ (ρ·U_tip·c)/u",f"{Re_8:,.0f}", "Régimen aerodinámico de diseño (ρ: densidad; U_tip: punta; c: cuerda; μ: viscosidad)",)
    with p8:
        kpi_card("Re @ v_max ≈ (ρ·U_tip,max·c)/u",f"{Re_max:,.0f}","Régimen aerodinámico límite operativo para alta velocidad",)
    with p9:
        kpi_card(
            "M_base ≈ T/N + F·L",
            f"{M_root_rated/1000:.1f} kN·m",
            f"~{M_strut_rated/1000:.1f} kN·m/strut (n={int(struts_per_blade)})"
        )

    st.caption(
        "Las propiedades de la pala permiten evaluar esfuerzos en uniones, ejes y rodamientos, "
        "además de la respuesta dinámica del rotor. Re indica el régimen aerodinámico del perfil."
    )
st.markdown('</div>', unsafe_allow_html=True)


# Especificaciones a revisar
st.markdown(f"""
<div class="comment-box">
  <div class="comment-title">📐 Especificaciones bajo revisión</div>
  <p>
    D = {D:.1f} m, H = {H:.1f} m, N = {int(N)}, cuerda = {c:.2f} m, TSR ref = {tsr_ref:.2f},
    relación G = {G:.2f}, η_mec ≈ {eta_mec:.3f}, η_elec ≈ {eta_elec:.3f}. Usa estos valores como referencia al analizar cada gráfico.
  </p>
</div>
""", unsafe_allow_html=True)


# Tabla de resultados + filtro tipo píldoras
# =========================================================

modulos_columnas = {
    "Rotor (aero + dinámica)": [
        "v (m/s)", "λ_efectiva", "U_tip (m/s)",
        "Re (mid-span)", "Cp(λ_efectiva)", "Cp_aero_equiv",
        "rpm_rotor", "T_rotor (N·m)", "F_cen/pala (kN)", "a_cen (g)", "M_base (kN·m)", "M_por_strut (kN·m)",
        "sigma_root (MPa)", "sigma_strut (MPa)", "margen_root (%)", "margen_strut (%)",
        "f_1P (Hz)", "f_3P (Hz)"
    ],
    "Tren mecánico": [
        "v (m/s)", "P_aero (kW)", "P_mec_gen (kW)",
        "Cp_shaft_equiv"
    ],
    "Generador + eléctrico": [
        "v (m/s)", "rpm_gen", "P_gen_curve (kW)",
        "V_LL (V)", "V_LL (Ke) [V]", "f_e (Hz)",
        "η_gen (curve)", "T_gen (N·m)",
        "P_el (kW)", "P_out (clip) kW", "I_est (A)", "Duty_DC (%)",
        "Cp_el_equiv"
    ],
    "Ruido": [
        "v (m/s)", "Lw (dB)", "Lp_obs (dB)"
    ],
}

if "modulo_tabla" not in st.session_state:
    st.session_state["modulo_tabla"] = "Todas"

# ---------- ESTILO SELECTOR + TABLA ----------
st.markdown("""
<style>

/* ===== PÍLDORAS DEL SELECTOR (st.radio) ===== */
div[data-testid="stRadio"] > label {
    font-weight: 600;
    margin-bottom: 0.35rem;
}

div[data-testid="stRadio"] > div {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    justify-content: center;
}

div[data-testid="stRadio"] > div > label {
    border-radius: 999px;
    padding: 0.25rem 0.9rem;
    border: 1px solid #e5e7eb;
    background: #f9fafb;
    cursor: pointer;
    font-size: 0.9rem;
    color: #111827;
    transition: all 0.12s ease-in-out;
}

div[data-testid="stRadio"] > div > label:hover {
    background: #e0f2fe;
    border-color: #60a5fa;
}

div[data-testid="stRadio"] > div > label[data-checked="true"] {
    background: linear-gradient(135deg,#1d4ed8,#0ea5e9);
    color: #ffffff;
    border-color: transparent;
    box-shadow: 0 3px 10px rgba(15,23,42,0.35);
}

/* ===== CONTENEDOR TABLA (st.dataframe) ===== */
[data-testid="stDataFrame"] {
    border-radius: 16px;
    border: 1px solid rgba(148,163,184,0.7);
    box-shadow: 0 18px 40px rgba(15,23,42,0.55);
    overflow: hidden;
    background: #020617;
}

/* Contenido scrolleable dentro de la “card” */
[data-testid="stDataFrame"] > div {
    max-height: 460px;
    overflow: auto;
}

/* Scrollbar sutil */
[data-testid="stDataFrame"]::-webkit-scrollbar,
[data-testid="stDataFrame"] > div::-webkit-scrollbar {
    height: 8px;
    width: 8px;
}
[data-testid="stDataFrame"]::-webkit-scrollbar-thumb,
[data-testid="stDataFrame"] > div::-webkit-scrollbar-thumb {
    background: rgba(148,163,184,0.6);
    border-radius: 999px;
}
[data-testid="stDataFrame"]::-webkit-scrollbar-track,
[data-testid="stDataFrame"] > div::-webkit-scrollbar-track {
    background: transparent;
}

/* ===== NUEVO SISTEMA — PRIMERA COLUMNA REAL ===== */

/* HEADER de la primera columna (v (m/s)) */
div[data-testid="stDataFrame"] div[aria-colindex="0"][data-testid="column-header-cell"] {
    background-color: #0f172a !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    border-right: 1px solid #1e293b !important;
}

/* CELDAS de la primera columna (valores de viento) */
div[data-testid="stDataFrame"] div[aria-colindex="0"][data-testid="cell"] {
    background-color: #111827 !important;
    color: #f1f5f9 !important;
    font-weight: 600 !important;
    border-right: 1px solid #1e293b !important;
    text-align: left !important;
}

/* HOVER de la primera columna */
div[data-testid="stDataFrame"] div[aria-colindex="0"][data-testid="cell"]:hover {
    background-color: #1e293b !important;
}

</style>
""", unsafe_allow_html=True)

# ---------- TÍTULO + SELECTOR ----------
st.subheader("📊 Tabla de resultados por viento")
question_prompt("¿Qué rango de viento revela discrepancias entre variables aero, mecánicas y eléctricas que debamos priorizar en las siguientes simulaciones?")

# ---------- KPIs rápidos ----------
df_range = df.copy()

range_kpis = [
    {
        "label": "λ promedio (rango)",
        "value": float(df_range["λ_efectiva"].mean()) if not df_range.empty else np.nan,
        "fmt": lambda v: f"{v:.2f}",
        "sub": "TSR efectiva media del intervalo."
    },
    {
        "label": "P_out máx [kW]",
        "value": float(df_range["P_out (clip) kW"].max()) if not df_range.empty else np.nan,
        "fmt": lambda v: f"{v:.1f}",
        "sub": "Potencia eléctrica máxima disponible."
    },
    {
        "label": "I_est máx [A]",
        "value": float(df_range["I_est (A)"].max()) if not df_range.empty else np.nan,
        "fmt": lambda v: f"{v:.1f}",
        "sub": "Corriente trifásica estimada en el rango."
    },
    {
        "label": "Cp_el promedio",
        "value": float(df_range["Cp_el_equiv"].mean()) if not df_range.empty else np.nan,
        "fmt": lambda v: f"{v:.3f}",
        "sub": "Eficiencia Cp eléctrica ponderada."
    },
    {
        "label": "T_gen máx [N·m]",
        "value": float(df_range["T_gen (N·m)"].max()) if not df_range.empty else np.nan,
        "fmt": lambda v: f"{v:.0f}",
        "sub": "Par máximo en el eje rápido."
    },
]

col_kpis = st.columns(len(range_kpis))
for col, card in zip(col_kpis, range_kpis):
    val = card["value"]
    value_text = card["fmt"](val) if np.isfinite(val) else "—"
    col.markdown(
        f"""
        <div class="range-card">
            <div class="range-card__label">{escape(card["label"])}</div>
            <div class="range-card__value">{value_text}</div>
            <div class="range-card__sub">{escape(card["sub"])}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

pill_labels = {
    "🟢 Todas": "Todas",
    "⚙️ Rotor": "Rotor (aero + dinámica)",
    "🔧 Tren mecánico": "Tren mecánico",
    "⚡ Generador": "Generador + eléctrico",
    "🔈 Ruido": "Ruido",
}

left, center, right = st.columns([1, 4, 1])
with center:
    sel_label = st.radio(
        "",
        list(pill_labels.keys()),
        horizontal=True,
        key="radio_modulos",
    )

mod_sel = pill_labels[sel_label]
st.session_state["modulo_tabla"] = mod_sel

# ---------- FILTRO DE COLUMNAS ----------
if mod_sel == "Todas":
    df_view = df_range
else:
    cols = [c for c in modulos_columnas.get(mod_sel, []) if c in df_range.columns]
    df_view = df_range[cols] if cols else df_range

# ---------- CLASIFICACIÓN REGIÓN IEC Y ESTILO ----------
reg2_label = "MPPT" if control_mode == "MPPT (λ constante)" else "RPM fija"

def region_tag(v):
    if (v_cut_in is not None) and v < v_cut_in:
        return "Pre cut-in"
    if (v_cut_in is not None) and (v_rated is not None) and (v_cut_in <= v < v_rated):
        return reg2_label
    if (v_rated is not None) and (v_cut_out is not None) and (v_rated <= v <= v_cut_out):
        return "Potencia limitada"
    if (v_cut_out is not None) and v > v_cut_out:
        return "Sobre cut-out"
    return "Sin clasificar"

region_colors = {
    "Pre cut-in": "rgba(148,163,184,0.08)",
    "MPPT": "rgba(34,197,94,0.10)",
    "RPM fija": "rgba(34,197,94,0.10)",
    "Potencia limitada": "rgba(234,179,8,0.12)",
    "Sobre cut-out": "rgba(239,68,68,0.12)",
}

if not df_view.empty:
    df_view = df_view.copy()
    df_view["Región IEC"] = df_view["v (m/s)"].apply(region_tag)
    numeric_cols = df_view.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        df_view[numeric_cols] = df_view[numeric_cols].round(2)

def highlight_region(row):
    color = region_colors.get(row.get("Región IEC"), "transparent")
    return [f"background-color: {color}"] * len(row)

if not df_view.empty:
    style_obj = df_view.style.apply(highlight_region, axis=1)
    numeric_cols = df_view.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        style_obj = style_obj.format({col: "{:.2f}" for col in numeric_cols})
    df_to_render = style_obj
else:
    df_to_render = df_view

# ---------- TABLA + DESCARGA ----------
column_config_map = {

        "v (m/s)": st.column_config.NumberColumn(
            "v (m/s)",
            help=(
                "**Descripción:** velocidad del viento usada como eje base.\n"
                "**Origen:** simulación Weibull o datos SCADA.\n"
                "**Uso:** limita el rango visible con el slider superior (no recalcula la física)."
            )
        ),
        "rpm_rotor": st.column_config.NumberColumn(
            "rpm_rotor",
            help=(
                "**Descripción:** rpm del rotor bajo control MPPT.\n"
                "**Fórmula:** MPPT → (30/πR)·λ_ctrl·v. Sin MPPT → rpm fija entre cut-in y cut-out.\n"
                "**Control:** MPPT usa ‘TSR objetivo’; sin MPPT usa rpm_rotor_rated."
            )
        ),
        "rpm_gen": st.column_config.NumberColumn(
            "rpm_gen",
            help=(
                "**Descripción:** rpm del generador después de la caja.\n"
                "**Fórmula:** rpm_gen = rpm_rotor · G.\n"
                "**Parámetros:** G (calculado o manual en ‘Tren de potencia’). Cambia al mover TSR o G."
            )
        ),
        "λ_efectiva": st.column_config.NumberColumn(
            "λ_efectiva",
            help=(
                "**Descripción:** TSR que realmente alcanza el rotor.\n"
                "**Fórmula:** λ = ω_rot·R / v = U_tip / v.\n"
                "**Nota:** ≈ λ_ctrl en MPPT; en rpm fija varía con v y cae en Región 3."
            )
        ),
        "U_tip (m/s)": st.column_config.NumberColumn(
            "U_tip (m/s)",
            help=(
                "**Descripción:** velocidad de punta (criterio acústico/estructural).\n"
                "**Fórmula:** U_tip = λ_efectiva · v.\n"
                "**Observa:** responde al slider TSR y al rango de viento."
            )
        ),
        "Cp(λ_efectiva)": st.column_config.NumberColumn(
            "Cp(λ_efectiva)",
            help=(
                "**Descripción:** Cp teórico evaluado en la TSR efectiva.\n"
                "**Modelo:** curva Cp(λ) definida por el perfil seleccionado.\n"
                "**Uso:** identifica desviaciones del MPPT antes de pérdidas mecánicas."
            )
        ),
        "Cp_aero_equiv": st.column_config.NumberColumn(
            "Cp_aero_equiv",
            help=(
                "**Descripción:** Cp práctico a la salida del rotor.\n"
                "**Fórmula:** Cp = P_aero /(½·ρ·A·v³).\n"
                "**Inputs:** ρ y geometría (D·H) configurados en el sidebar."
            )
        ),
        "Cp_shaft_equiv": st.column_config.NumberColumn(
            "Cp_shaft_equiv",
            help=(
                "**Descripción:** Cp después de rodamientos+caja (llega al eje rápido).\n"
                "**Fórmula:** Cp_shaft = P_mec_gen /(½·ρ·A·v³)."
            )
        ),
        "Cp_el_equiv": st.column_config.NumberColumn(
            "Cp_el_equiv",
            help=(
                "**Descripción:** Cp a la salida eléctrica útil (tras electrónica+clipping).\n"
                "**Fórmula:** Cp_el = P_out /(½·ρ·A·v³)."
            )
        ),
        "Re (mid-span)": st.column_config.NumberColumn(
            "Re (mid-span)",
            help=(
                "**Descripción:** Reynolds en la sección media de pala.\n"
                "**Fórmula:** Re = ρ·U_tip·c / μ.\n"
                "**Parámetros:** ρ, μ y cuerda se editan en el panel lateral."
            )
        ),
        "P_aero (kW)": st.column_config.NumberColumn(
            "P_aero (kW)",
            help=(
                "**Descripción:** potencia aerodinámica capturada por el rotor.\n"
                "**Fórmula:** ½·ρ·A·v³·Cp(λ_efectiva)."
            )
        ),
        "P_mec_gen (kW)": st.column_config.NumberColumn(
            "P_mec_gen (kW)",
            help=(
                "**Descripción:** potencia que llega al eje del generador.\n"
                "**Fórmula:** P_mec = P_aero · η_mec (rodamientos · caja)."
            )
        ),
        "P_gen_curve (kW)": st.column_config.NumberColumn(
            "P_gen_curve (kW)",
            help=(
                "**Descripción:** potencia según la curva del generador (datasheet/CSV).\n"
                "**Fórmula:** Interpolación P(rpm_gen).\n"
                "**Uso:** valida si el MPPT exige más de lo que el generador puede entregar."
            )
        ),
        "η_gen (curve)": st.column_config.NumberColumn(
            "η_gen (curve)",
            help=(
                "**Descripción:** eficiencia instantánea del generador.\n"
                "**Fórmula:** η = P_el_gen / P_mec_gen.\n"
                "**Referencia:** compara con η_gen_max configurada para detectar saturaciones."
            )
        ),
        "V_LL (V)": st.column_config.NumberColumn(
            "V_LL (V)",
            help=(
                "**Descripción:** tensión línea-línea tomada de la curva cargada.\n"
                "**Fórmula:** V = interp_V(rpm_gen).\n"
                "**Nota:** útil para verificar compatibilidad con la electrónica existente."
            )
        ),
        "V_LL (Ke) [V]": st.column_config.NumberColumn(
            "V_LL (Ke) [V]",
            help=(
                "**Descripción:** estimación basada en la constante Ke del generador.\n"
                "**Fórmula:** V = Ke · ω_gen.\n"
                "**Uso:** comparar con la curva real (columna anterior) y detectar desvíos."
            )
        ),
        "f_e (Hz)": st.column_config.NumberColumn(
            "f_e (Hz)",
            help=(
                "**Descripción:** frecuencia eléctrica trifásica.\n"
                "**Fórmula:** f = (poles/2)·rpm_gen/60.\n"
                "**Parámetros:** número de polos definido en el panel ‘Tren de potencia’."
            )
        ),
        "f_1P (Hz)": st.column_config.NumberColumn(
            "f_1P (Hz)",
            help=(
                "**Descripción:** frecuencia de paso del rotor (1 vuelta por segundo).\n"
                "**Fórmula:** f_1P = rpm_rotor / 60.\n"
                "**Uso:** comparar con modos estructurales y evitar resonancias."
            )
        ),
        "f_3P (Hz)": st.column_config.NumberColumn(
            "f_3P (Hz)",
            help=(
                "**Descripción:** frecuencia 3P (una por pala en rotor de 3 palas).\n"
                "**Fórmula:** f_3P = 3 · f_1P."
            )
        ),
        "T_rotor (N·m)": st.column_config.NumberColumn(
            "T_rotor (N·m)",
            help=(
                "**Descripción:** torque aerodinámico del eje lento.\n"
                "**Fórmula:** T = P_aero / ω_rot.\n"
                "**Nota:** revisa límites IEC configurados en el panel de diseño."
            )
        ),
        "T_gen (N·m)": st.column_config.NumberColumn(
            "T_gen (N·m)",
            help=(
                "**Descripción:** torque visto por el generador.\n"
                "**Fórmula:** T_gen = T_rotor / G.\n"
                "**Control:** modifícalo cambiando G o la ley de par (TSR)."
            )
        ),
        "F_cen/pala (kN)": st.column_config.NumberColumn(
            "F_cen/pala (kN)",
            help=(
                "**Descripción:** fuerza centrípeta por pala para cada bin.\n"
                "**Fórmula:** F = m_pala · R · ω².\n"
                "**Uso:** comparar con límites estructurales de la raíz/struts."
            )
        ),
        "a_cen (g)": st.column_config.NumberColumn(
            "a_cen (g)",
            help=(
                "**Descripción:** aceleración radial equivalente en g.\n"
                "**Fórmula:** a = R·ω² / g.\n"
                "**Referencia:** chequea contra el límite configurado en el panel IEC."
            )
        ),
        "M_base (kN·m)": st.column_config.NumberColumn(
            "M_base (kN·m)",
            help=(
                "**Descripción:** momento flector estimado en la raíz de cada pala.\n"
                "**Fórmula:** M ≈ (T_rotor/N) + F_cen·brazo.\n"
                "**Uso:** dimensionamiento de pala, struts y base de torre."
            )
        ),
        "M_por_strut (kN·m)": st.column_config.NumberColumn(
            "M_por_strut (kN·m)",
            help=(
                "**Descripción:** momento flector que recibe cada strut/brazo.\n"
                "**Fórmula:** M_strut = M_base / Nº struts.\n"
                "**Nota:** ajusta el parámetro ‘N° de struts por pala’ en el panel lateral."
            )
        ),
        "sigma_root (MPa)": st.column_config.NumberColumn(
            "sigma_root (MPa)",
            help=(
                "**Descripción:** tensión aproximada en la raíz de la pala.\n"
                "**Fórmula:** σ = M_root / W_root.\n"
                "**Inputs:** W_root y σ_y se editan en ‘Propiedades estructurales avanzadas’."
            )
        ),
        "sigma_strut (MPa)": st.column_config.NumberColumn(
            "sigma_strut (MPa)",
            help=(
                "**Descripción:** tensión axial estimada en struts.\n"
                "**Fórmula:** σ = F_strut / A_strut, con F_strut ≈ M_strut / brazo.\n"
                "**Inputs:** Área efectiva y brazo se editan en el panel lateral."
            )
        ),
        "margen_root (%)": st.column_config.NumberColumn(
            "margen_root (%)",
            help=(
                "**Descripción:** margen de seguridad en la raíz.\n"
                "**Fórmula:** (σ_admisible/FS − σ_root) / (σ_admisible/FS).\n"
                "**Criterio:** negativo indica sobreesfuerzo."
            )
        ),
        "margen_strut (%)": st.column_config.NumberColumn(
            "margen_strut (%)",
            help=(
                "**Descripción:** margen de seguridad en struts.\n"
                "**Fórmula:** (σ_admisible/FS − σ_strut) / (σ_admisible/FS).\n"
                "**Criterio:** negativo indica sobreesfuerzo."
            )
        ),
        "P_el (kW)": st.column_config.NumberColumn(
            "P_el (kW)",
            help=(
                "**Descripción:** potencia AC tras la electrónica (sin clipping nominal).\n"
                "**Fórmula:** P_el = P_el_gen · η_elec.\n"
                "**Parámetros:** η_elec se define en el panel (rectificador+inversor)."
            )
        ),
        "P_out (clip) kW": st.column_config.NumberColumn(
            "P_out (clip) kW",
            help=(
                "**Descripción:** potencia útil limitada por P_nom o inversor.\n"
                "**Fórmula:** P_out = min(P_el, P_nom).\n"
                "**Nota:** cambia al modificar P_nom en el panel lateral."
            )
        ),
        "I_est (A)": st.column_config.NumberColumn(
            "I_est (A)",
            help=(
                "**Descripción:** corriente trifásica antes de la electrónica/clipping.\n"
                "**Fórmula:** I = max(P_el_gen /(√3·V_LL·PF), T_gen/Kt).\n"
                "**Parámetros:** PF configurable; Kt proviene del panel del generador."
            )
        ),
        "Duty_DC (%)": st.column_config.NumberColumn(
            "Duty DC (%)",
            help=(
                "**Descripción:** utilización del bus DC estimada.\n"
                "**Fórmula:** Duty = P_el_gen /(V_dc_nom · I_dc_nom).\n"
                "**Uso:** intenta mantenerlo < 100% para evitar saturación térmica del bus."
            )
        ),
        "Lw (dB)": st.column_config.NumberColumn(
            "Lw (dB)",
            help=(
                "**Descripción:** nivel de potencia sonora del rotor.\n"
                "**Modelo:** Lw = Lw_ref + 10·n·log10(U_tip/U_tip_ref).\n"
                "**Inputs:** Lw_ref y exponente n se definen en el expander de ruido."
            )
        ),
        "Lp_obs (dB)": st.column_config.NumberColumn(
            "Lp_obs (dB)",
            help=(
                "**Descripción:** nivel estimado en el receptor configurado.\n"
                "**Fórmula:** Lp = Lw − 20·log10(r_obs) − 11.\n"
                "**Supuesto:** propagación en campo libre a la distancia r_obs (panel de ruido)."
            )
        ),
        "P_loss_mec (kW)": st.column_config.NumberColumn(
            "P_loss_mec (kW)",
            help=(
                "**Descripción:** pérdidas entre rotor y eje rápido (rodamientos + caja).\n"
                "**Cálculo:** P_loss_mec = P_aero − P_mec_gen.\n"
                "**Acción:** reduce cargas o mejora la lubricación si esta banda domina."
            )
        ),
        "P_loss_gen (kW)": st.column_config.NumberColumn(
            "P_loss_gen (kW)",
            help=(
                "**Descripción:** pérdidas internas del generador (cobre, hierro, ventilación).\n"
                "**Cálculo:** P_loss_gen = P_mec_gen − P_el_gen."
            )
        ),
        "P_loss_elec (kW)": st.column_config.NumberColumn(
            "P_loss_elec (kW)",
            help=(
                "**Descripción:** pérdidas en electrónica de potencia.\n"
                "**Cálculo:** P_loss_elec = P_el_gen − P_el.\n"
                "**Nota:** depende de la η_elec escogida."
            )
        ),
        "P_loss_clip (kW)": st.column_config.NumberColumn(
            "P_loss_clip (kW)",
            help=(
                "**Descripción:** energía recortada por límites nominales.\n"
                "**Cálculo:** max(0, P_el − P_out).\n"
                "**Sugerencia:** si domina temprano, aumenta P_nom o ajusta TSR/MPPT."
            )
        ),
        "Región IEC": st.column_config.Column(
            "Región IEC",
            help=(
                "**Descripción:** etiqueta IEC automática del bin de viento.\n"
                f"• Pre cut-in: v < v_cut-in ({v_cut_in:.1f} m/s)\n"
                f"• {reg2_label}: {v_cut_in:.1f} ≤ v < v_rated ({v_rated:.1f} m/s)\n"
                f"• Potencia limitada: {v_rated:.1f} ≤ v ≤ v_cut-out ({v_cut_out:.1f} m/s)\n"
                f"• Sobre cut-out: v > v_cut-out ({v_cut_out:.1f} m/s)\n"
                "**Nota:** cambia automáticamente si modificas v_cut-in/rated/cut-out en el sidebar."
            ),
            width="small",
        ),
    }

visible_cols = list(df_view.columns) if hasattr(df_view, "columns") else []
column_config_filtered = {
    key: value for key, value in column_config_map.items() if key in visible_cols
}

st.dataframe(
    df_to_render,
    use_container_width=True,
    height=480,
    column_config=column_config_filtered,
)



# --- Botón para descargar CSV de la tabla ---


st.download_button(
    f"📥 Descargar CSV – vista: {mod_sel}",
    data=df_view.to_csv(index=False).encode("utf-8"),
    file_name=f"vawt_resultados_{mod_sel.replace(' ', '_')}.csv",
    mime="text/csv",
    key="csv_tabla_resultados"
)



# ====== DISEÑO PARA FÓRMULAS DE CADA COLUMNA ======
st.markdown(
    """
<style>
.formula-bullet {
    font-size: 0.9rem;
    margin: 0.15rem 0;
}
.formula-bullet b {
    font-weight: 600;
}
.formula-inline {
    font-family: "SF Mono", "JetBrains Mono", Menlo, monospace;
    font-size: 0.9rem;
}
</style>
""",
    unsafe_allow_html=True,
)


# Bloque – Aerodinámica
section_header("🌀 Aerodinámica y comportamiento del perfil")

# =========================================================
# Gráfico – Polar Lift–Drag del perfil seleccionado
# =========================================================
st.subheader("🌀 Polar Lift–Drag del perfil seleccionado")
question_prompt("¿En qué intervalo de ángulos de ataque quieres operar la pala para equilibrar sustentación y arrastre según el perfil seleccionado?")

df_polar = build_lift_drag_polar(t_rel=t_rel, symmetric=is_symmetric)

with st.expander("Cargar polar Lift-Drag (CSV o pegado)", expanded=False):
    use_custom_polar = st.checkbox(
        "Usar datos cargados",
        value=False,
        help="Si se activa, se reemplaza la polar simplificada por tus datos.",
    )
    st.caption("Columnas esperadas: alpha_deg, Cl, Cd (ClCd opcional).")
    polar_file = st.file_uploader(
        "Subir polar CSV",
        type=["csv"],
        accept_multiple_files=False,
    )
    st.caption("O ingresa datos manualmente:")
    default_rows = pd.DataFrame(
        {
            "alpha_deg": list(range(-10, 21, 5)),
            "Cl": [-0.90, -0.45, 0.00, 0.52, 0.90, 1.10, 1.20],
            "Cd": [0.020, 0.015, 0.012, 0.017, 0.025, 0.030, 0.035],
        }
    )
    polar_table = st.data_editor(
        st.session_state.get("polar_table", default_rows),
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "alpha_deg": st.column_config.NumberColumn(
                "alpha_deg",
                min_value=-10.0,
                max_value=20.0,
                step=0.1,
            ),
        },
        key="polar_table",
    )

if use_custom_polar:
    df_custom = None
    if polar_file is not None:
        df_custom = pd.read_csv(polar_file)
    elif polar_table is not None and not polar_table.empty:
        df_custom = polar_table.copy()

    if df_custom is not None and not df_custom.empty:
        cols = {c.lower().strip(): c for c in df_custom.columns}

        def pick_col(keys):
            for key in keys:
                for col_lower, col_name in cols.items():
                    if key in col_lower:
                        return col_name
            return None

        alpha_col = pick_col(["alpha_deg", "alpha", "aoa", "angle"])
        cl_col = pick_col(["cl"])
        cd_col = pick_col(["cd"])
        clcd_col = pick_col(["cl/cd", "clcd", "cl_cd"])

        if alpha_col is None or cl_col is None or cd_col is None:
            st.error("CSV incompleto: se requieren columnas de alpha_deg (o alpha), Cl y Cd.")
        else:
            df_polar = pd.DataFrame(
                {
                    "alpha_deg": df_custom[alpha_col].astype(float),
                    "Cl": df_custom[cl_col].astype(float),
                    "Cd": df_custom[cd_col].astype(float),
                }
            )
            if clcd_col is not None:
                df_polar["ClCd"] = df_custom[clcd_col].astype(float)
            else:
                df_polar["ClCd"] = np.divide(
                    df_polar["Cl"].values,
                    df_polar["Cd"].values,
                    out=np.zeros_like(df_polar["Cl"].values),
                    where=(df_polar["Cd"].values != 0),
                )

fig_polar = make_subplots(specs=[[{"secondary_y": True}]])

# Cl y Cd en eje izquierdo
fig_polar.add_trace(
    go.Scatter(
        x=df_polar["alpha_deg"],
        y=df_polar["Cl"],
        mode="lines",
        name="Cl(α)",
    ),
    secondary_y=False,
)

fig_polar.add_trace(
    go.Scatter(
        x=df_polar["alpha_deg"],
        y=df_polar["Cd"],
        mode="lines",
        name="Cd(α)",
    ),
    secondary_y=False,
)

# Cl/Cd en eje derecho
fig_polar.add_trace(
    go.Scatter(
        x=df_polar["alpha_deg"],
        y=df_polar["ClCd"],
        mode="lines",
        name="Cl/Cd(α)",
        line=dict(dash="dot"),
    ),
    secondary_y=True,
)

fig_polar.update_xaxes(
    title_text="Ángulo de ataque α [°]",
    zeroline=False,
    showgrid=True,
)

fig_polar.update_yaxes(
    title_text="Cl, Cd",
    secondary_y=False,
    showgrid=True,
    gridcolor="rgba(148,163,184,0.35)",
)
fig_polar.update_yaxes(
    title_text="Cl/Cd",
    secondary_y=True,
    showgrid=False,
)

fig_polar.update_layout(
    legend_title="Magnitudes",
    margin=dict(l=60, r=60, t=40, b=40),
    hovermode="x unified",
    hoverlabel=dict(
        bgcolor="white",
        font_size=12,
        font_color="black",
    ),
    plot_bgcolor="white",
)
st.plotly_chart(fig_polar, use_container_width=True)

# Comentario técnico
st.markdown("""
<div class="comment-box">
  <div class="comment-title">✈️ Lectura rápida de la polar</div>
  <p>
  Este gráfico muestra la respuesta aerodinámica genérica del perfil seleccionado:
  <ul>
    <li><b>Cl(α)</b> crece casi linealmente hasta la zona de <em>stall</em>, donde comienza a saturarse.</li>
    <li><b>Cd(α)</b> aumenta de forma cuadrática con Cl, reflejando el drag inducido y de perfil.</li>
    <li><b>Cl/Cd(α)</b> indica la <strong>eficiencia aerodinámica</strong>; el máximo local se asocia al rango de ángulos
        de ataque más conveniente para operación quasi-estacionaria del rotor.</li>
  </ul>
  En el contexto del VAWT, esta polar sirve como referencia para entender el rango de α en el que el perfil
  trabaja durante el giro, y cómo cambios en espesor o tipo (simétrico/asimétrico) afectan sustentación y pérdidas.
  </p>
</div>
""", unsafe_allow_html=True)

# =========================================================
# Gráfico – Ciclo de ángulo de ataque α(θ) y efecto del pitch
# =========================================================
st.subheader("🧭 Ciclo de ángulo de ataque α(θ) – efecto del pitch")
question_prompt("¿Qué desplazamiento de pitch necesitas validar para mantener α dentro del sweet spot durante todo el ciclo azimutal?")

theta_deg = np.linspace(0, 360, 721)  # 0.5° resolución
lam_used = float(tsr_ref)             # TSR de referencia (setpoint o resultante)

# Variantes de pitch (centro en slider actual)
pitch_variants = [
    float(np.clip(pitch_deg - 2.0, -10.0, 10.0)),
    float(np.clip(pitch_deg,       -10.0, 10.0)),
    float(np.clip(pitch_deg + 2.0, -10.0, 10.0)),
]

# Sweet spot de referencia (puedes ajustarlo)
alpha_opt_ref = 5.0
alpha_band = 2.0  # ±2°

fig_alpha = go.Figure()

# Banda del sweet spot
fig_alpha.add_hrect(
    y0=alpha_opt_ref - alpha_band,
    y1=alpha_opt_ref + alpha_band,
    fillcolor="rgba(34,197,94,0.10)",
    line_width=0,
    layer="below",
    annotation_text=f"Sweet spot ≈ {alpha_opt_ref:.0f}° ± {alpha_band:.0f}°",
    annotation_position="top right",
    annotation_yshift=12,
)

# Líneas α(θ) para cada pitch
for p in pitch_variants:
    alpha_deg = alpha_cycle_deg(theta_deg, lam=lam_used, pitch_deg=p)

    # Métricas rápidas (opcionales)
    a_min, a_max = float(alpha_deg.min()), float(alpha_deg.max())
    a_mean = float(alpha_deg.mean())

    fig_alpha.add_trace(
        go.Scatter(
            x=theta_deg,
            y=alpha_deg,
            mode="lines",
            name=f"pitch = {p:+.1f}°  (min={a_min:.1f}, mean={a_mean:.1f}, max={a_max:.1f})",
            hovertemplate="θ = %{x:.1f}°<br>α = %{y:.2f}°<extra></extra>",
        )
    )

# Líneas guía
fig_alpha.add_hline(
    y=alpha_opt_ref,
    line_dash="dot",
    annotation_text="α_opt ref",
    annotation_position="bottom right",
    annotation_yshift=-12,
)

fig_alpha.update_xaxes(
    title_text="Posición azimutal θ [°]",
    range=[0, 360],
    showgrid=True,
)
fig_alpha.update_yaxes(
    title_text="Ángulo de ataque α [°]",
    showgrid=True,
    gridcolor="rgba(148,163,184,0.35)",
    zeroline=False,
)

fig_alpha.update_layout(
    margin=dict(l=60, r=20, t=40, b=40),
    hovermode="x unified",
    plot_bgcolor="white",
    legend_title=f"TSR usado: λ = {lam_used:.2f}",
)

st.plotly_chart(fig_alpha, use_container_width=True)

st.markdown("""
<div class="comment-box">
  <div class="comment-title">🔍 Interpretación</div>
  <p>
  Este gráfico muestra cómo el ángulo de ataque <b>α</b> varía durante una vuelta completa del rotor (θ).
  Cambiar el <b>pitch</b> en ±2° desplaza toda la curva α(θ) hacia arriba o hacia abajo, sin cambiar su forma.
  La banda verde marca un rango de referencia para un “sweet spot” aerodinámico alrededor de α≈5°.
  </p>
</div>
""", unsafe_allow_html=True)


# Bloque – Operación y control
section_header("⚙️ Operación y control del rotor")

# =========================================================
# Gráfico 1 – rpm rotor / rpm generador (ancho completo)
# =========================================================
st.subheader("⚙️ rpm rotor / rpm generador")
question_prompt("¿Las transiciones entre regiones de control mantienen rpm_rotor y rpm_gen dentro de los límites que exige tu especificación de tren de potencia?")

# Datos ordenados + región de operación
df_rpm_plot = df.sort_values("v (m/s)").copy()
v_vals = df_rpm_plot["v (m/s)"].values

region = np.where(
    v_vals < v_cut_in, "Parado",
    np.where(
        v_vals <= v_rated,
        "MPPT (λ≈const)" if control_mode == "MPPT (λ constante)" else "RPM fija",
        np.where(v_vals <= v_cut_out, "Potencia limitada", "Parado"),
    ),
)

G_inst = np.divide(
    df_rpm_plot["rpm_gen"].values,
    np.maximum(df_rpm_plot["rpm_rotor"].values, 1e-6)
)

custom = np.stack([
    df_rpm_plot["rpm_rotor"].values,
    df_rpm_plot["rpm_gen"].values,
    G_inst,
    df_rpm_plot["λ_efectiva"].values,
    region
], axis=-1)

fig_r = go.Figure()

# Rotor
fig_r.add_trace(
    go.Scatter(
        x=df_rpm_plot["v (m/s)"],
        y=df_rpm_plot["rpm_rotor"],
        mode="lines+markers",
        name="Rotor (rpm)",
        customdata=custom,
        hovertemplate=(
            "v = %{x:.1f} m/s<br>"
            "rpm_rotor = %{y:.1f} rpm<br>"
            "rpm_gen = %{customdata[1]:.1f} rpm<br>"
            "G = %{customdata[2]:.2f}<br>"
            "λ_efectiva = %{customdata[3]:.2f}<br>"
            "Región = %{customdata[4]}<extra></extra>"
        ),
    )
)

# Generador
fig_r.add_trace(
    go.Scatter(
        x=df_rpm_plot["v (m/s)"],
        y=df_rpm_plot["rpm_gen"],
        mode="lines+markers",
        name="Generador (rpm)",
        customdata=custom,
        hovertemplate=(
            "v = %{x:.1f} m/s<br>"
            "rpm_gen = %{y:.1f} rpm<br>"
            "rpm_rotor = %{customdata[0]:.1f} rpm<br>"
            "G = %{customdata[2]:.2f}<br>"
            "λ_efectiva = %{customdata[3]:.2f}<br>"
            "Región = %{customdata[4]}<extra></extra>"
        ),
    )
)

# Zonas sombreadas por región de control
fig_r.add_vrect(
    x0=float(v_cut_in), x1=float(v_rated),
    fillcolor="rgba(34,197,94,0.06)",
    line_width=0,
    layer="below",
    annotation_text="Región MPPT" if control_mode == "MPPT (λ constante)" else "Región rpm fija",
    annotation_position="top left",
)

fig_r.add_vrect(
    x0=float(v_rated), x1=float(v_cut_out),
    fillcolor="rgba(148,163,184,0.06)",
    line_width=0,
    layer="below",
    annotation_text="Potencia limitada",
    annotation_position="top right",
)

# Líneas verticales
for v_mark, label in [
    (v_cut_in,  "v_cut-in"),
    (v_rated,   "v_rated"),
    (v_cut_out, "v_cut-out"),
]:
    fig_r.add_vline(
        x=float(v_mark),
        line_dash="dot",
        line_color="rgba(148,163,184,0.6)",
        annotation_text=label,
        annotation_position="top",
    )

# Líneas horizontales de rpm nominales
fig_r.add_hline(
    y=float(rpm_rotor_rated),
    line_dash="dot",
    line_color="#22c55e",
    annotation_text="rpm_rotor_rated",
    annotation_position="bottom left",
)
fig_r.add_hline(
    y=float(rpm_gen_rated),
    line_dash="dot",
    line_color="#eab308",
    annotation_text="rpm_gen_rated",
    annotation_position="bottom right",
)

# Estilo de ejes y layout
fig_r.update_xaxes(
    title_text="v (m/s)",
    showgrid=False,
    zeroline=False,
)

fig_r.update_yaxes(
    title_text="rpm",
    showgrid=True,
    gridcolor="rgba(148,163,184,0.35)",
    zeroline=False,
)

fig_r.update_layout(
    legend_title="Magnitud",
    margin=dict(l=60, r=20, t=40, b=40),
    plot_bgcolor="white",
    hovermode="x unified",  # 👈 tooltip unificado
    hoverlabel=dict(
        bgcolor="white",
        font_size=12,
        font_color="black",
    ),
)

st.plotly_chart(fig_r, use_container_width=True)

# 📝 Interpretación técnica
st.markdown("""
<div class="comment-box">
  <div class="comment-title">🔍 Interpretación técnica</div>
  <p>
    Entre <em>v_cut-in</em> y <em>v_rated</em> la curva azul debe crecer linealmente; si alcanza
    <strong>rpm_rotor_rated</strong> antes de <em>v_rated</em>, reduce la λ objetivo o la relación <strong>G</strong>
    para evitar sobrevelocidad. Si queda por debajo, estás perdiendo Cp: sube ligeramente TSR o disminuye pérdidas mecánicas.
  </p>
  <p>
    La curva naranja (generador) debe alcanzar <strong>rpm_gen_rated</strong> justo cuando inicia la región de potencia
    limitada. Si la sobrepasa, limita el MPPT o baja G; si nunca llega, la caja está muy corta y el generador opera lejos
    de su punto óptimo. Usa este gráfico para alinear control, multiplicadora y curva del generador.
  </p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# Gráfico – λ_efectiva, U_tip y Frecuencia eléctrica
# =========================================================
st.subheader("🚀 λ_efectiva, U_tip y Frecuencia eléctrica")
question_prompt("¿En qué punto la combinación de λ, U_tip y f_e empieza a chocar con restricciones acústicas o de electrónica que debamos ajustar?")

# Selector de eje x
x_axis_mode_u = st.radio(
    "Eje x",
    ("v (m/s)", "rpm"),
    horizontal=True,
    key="x_axis_lambda",
    help="Elige si quieres ver las curvas contra velocidad de viento o contra rpm del rotor.",
)

if x_axis_mode_u == "v (m/s)":
    x_col_u = "v (m/s)"
else:
    x_col_u = "rpm_rotor"

df_u = df.sort_values(x_col_u).copy()

fig_u = px.line(
    df_u,
    x=x_col_u,
    y=["λ_efectiva", "U_tip (m/s)", "f_e (Hz)"],
    markers=True,
)

fig_u.update_layout(
    xaxis_title="v (m/s)" if x_axis_mode_u == "v (m/s)" else "rpm",
    yaxis_title="λ / U_tip [m/s] / f_e [Hz]",
    legend_title="Variable",
    hovermode="x unified",
    plot_bgcolor="white",
    margin=dict(l=40, r=40, t=40, b=40),
)

# Fondo con solo líneas horizontales suaves
fig_u.update_xaxes(
    showgrid=False,
    zeroline=False,
)
fig_u.update_yaxes(
    showgrid=True,
    gridcolor="rgba(148,163,184,0.35)",
    zeroline=False,
)

# Líneas de referencia en x
if x_axis_mode_u == "v (m/s)":
    for x, label in [
        (v_cut_in, "v_cut-in"),
        (v_rated, "v_rated"),
        (v_cut_out, "v_cut-out"),
    ]:
        if x is not None:
            fig_u.add_vline(
                x=float(x),
                line_dash="dot",
                line_color="rgba(148,163,184,0.9)",
                annotation_text=label,
                annotation_position="top",
                annotation_font_size=11,
                annotation_font_color="rgba(107,114,128,1)",
            )
else:
    for x_val, label, color in [
        (rpm_rotor_rated, "rpm_rotor_rated", "rgba(148,163,184,0.8)"),
        (rpm_gen_rated,   "rpm_gen_rated",   "rgba(239,68,68,0.9)"),
    ]:
        try:
            fig_u.add_vline(
                x=float(x_val),
                line_dash="dot",
                line_color=color,
                annotation_text=label,
                annotation_position="top",
                annotation_font_size=11,
                annotation_font_color="rgba(107,114,128,1)",
            )
        except Exception:
            continue

# Región sombreada entre v_rated y v_cut-out (frecuencia / punta de pala limitadas)
if x_axis_mode_u == "v (m/s)":
    if (v_rated is not None) and (v_cut_out is not None):
        fig_u.add_vrect(
            x0=float(v_rated),
            x1=float(v_cut_out),
            fillcolor="rgba(148,163,184,0.10)",
            layer="below",
            line_width=0,
            annotation_text="Región potencia limitada",
            annotation_position="top left",
            annotation_font_size=11,
            annotation_font_color="rgba(107,114,128,1)",
        )

st.plotly_chart(fig_u, use_container_width=True)

st.markdown("""
<div class="comment-box">
  <div class="comment-title">🔍 Interpretación técnica</div>
  <p>
    Usa la curva de <strong>λ_efectiva</strong> para comprobar que en MPPT te mantienes dentro de la banda azul alrededor
    de λ<sub>opt</sub>; si cae antes de <em>v_rated</em> es señal de que el control está entregando menos par del necesario
    o de que la caja no sigue el setpoint.
  </p>
  <p>
    <strong>U_tip</strong> y <strong>f<sub>e</sub></strong> marcan límites acústicos y eléctricos: si U_tip supera el valor
    permitido antes de <em>v_rated</em>, reduce TSR o activa pitch para limitarla; si f<sub>e</sub> sale del rango de la
    electrónica, reevalúa el número de polos o el setpoint de G. En Región 3 deberían aplanarse.
  </p>
</div>
""", unsafe_allow_html=True)


# ==========================================================
# Curva Cp(λ)
# ==========================================================

st.subheader("🧩 Cp(λ) – Promedio, upwind y downwind")
question_prompt("¿Qué tan cerca quieres que el TSR objetivo permanezca del λ_opt estimado para cumplir la meta de Cp del proyecto?")

df_cp = cp_curve_for_plot(cp_params)

fig_cp = px.line(
    df_cp,
    x="λ",
    y=["Cp_prom", "Cp_upwind", "Cp_downwind"],
    markers=True,
)

fig_cp.update_layout(
    xaxis_title="λ",
    yaxis_title="Cp",
    legend_title="Componente",
    hovermode="x unified",           # 🔹 tooltip unificado en x
    plot_bgcolor="white",
    margin=dict(l=40, r=40, t=40, b=40),
)

# Fondo con solo líneas horizontales suaves
fig_cp.update_xaxes(
    showgrid=False,
    zeroline=False,
)
fig_cp.update_yaxes(
    showgrid=True,
    gridcolor="rgba(148,163,184,0.35)",
    zeroline=False,
)

lam_opt = float(cp_params["lam_opt"])
CP_BETZ = 16.0 / 27.0

# --- Línea vertical: TSR objetivo ---
tsr_line_label = "TSR objetivo" if control_mode == "MPPT (λ constante)" else "TSR @ v_rated"
fig_cp.add_vline(
    x=float(tsr_ref),
    line_dash="dot",
    line_color="rgba(249,115,22,0.9)",  # naranja
    annotation_text=tsr_line_label,
    annotation_position="top left",
    annotation_yshift=-60,
)

# --- Línea vertical: λ_opt del modelo ---
fig_cp.add_vline(
    x=lam_opt,
    line_dash="dash",
    line_color="rgba(34,197,94,0.9)",  # verde
    annotation_text="λ_opt",
    annotation_position="top right",
)

# --- Banda recomendada alrededor de λ_opt (banda MPPT) ---
band_half = 0.20 * lam_opt  # ±20% de λ_opt
x0_band = lam_opt - band_half
x1_band = lam_opt + band_half

fig_cp.add_vrect(
    x0=x0_band,
    x1=x1_band,
    fillcolor="rgba(59,130,246,0.08)",
    line_width=0,
    layer="below",
    annotation_text="Banda MPPT recomendada",
    annotation_position="top",
    annotation_yshift=16,
)

# --- Límite de Betz ---
fig_cp.add_hline(
    y=CP_BETZ,
    line_dash="dot",
    line_color="rgba(234,179,8,0.9)",
    annotation_text="Límite de Betz (0,593)",
    annotation_position="bottom right",
)

st.plotly_chart(fig_cp, use_container_width=True)

st.markdown("""
<div class="comment-box">
  <div class="comment-title">🔍 Interpretación técnica</div>
  <p>
    El máximo de la curva promedio define el Cp alcanzable con la geometría actual; si tu TSR objetivo (línea naranja)
    se separa más de ±0.2 de <strong>λ_opt</strong>, perderás más de 5 % de rendimiento incluso antes de considerar pérdidas
    mecánicas, por lo que conviene alinear el control o modificar la solidez.
  </p>
  <p>
    La diferencia entre las curvas upwind/downwind te dice cuánta energía estás perdiendo en la mitad lee: si la curva
    downwind cae demasiado, considera añadir helicoidal, end-plates o ajustar pitch para balancear cargas y acercarte al
    promedio. Opera siempre dentro de la banda azul para mantenerte a menos de 3 % del Cp máximo.
  </p>
</div>
""", unsafe_allow_html=True)


# Bloque – Potencia y eficiencia
section_header("📈 Potencia y eficiencia global")

    # =====================================================================
# POTENCIAS VS VIENTO – DOS MODOS
# =====================================================================
st.subheader("Potencia vs Viento")
question_prompt("¿En qué intervalo de vientos necesitas verificar que P_out siga la curva nominal sin clipping excesivo?")

# Selector tipo "pill" (horizontal) para el dominio de potencia
dominio_pot = st.radio(
    "Dominio de potencia",
    options=[
        "Potencias vs viento (recomendada)",
        "Potencia vs rpm generador",
    ],
    index=0,
    horizontal=True,
)

# =====================================================================
# MODO 1: POTENCIAS VS VIENTO
# =====================================================================
if dominio_pot == "Potencias vs viento (recomendada)":

    pot_norm = st.checkbox(
        "Mostrar potencias normalizadas (p.u.)",
        value=False,
        key="pot_norm_pu",
    )

    x_axis_mode_p = st.radio(
        "Eje x",
        ("v (m/s)", "rpm"),
        horizontal=True,
        key="x_axis_pot",
        help="Elige si quieres ver las potencias contra velocidad de viento o contra rpm del generador.",
    )

    y_cols_P = [
        "P_aero (kW)",
        "P_mec_gen (kW)",
        "P_out (clip) kW",
    ]

    if pot_norm and P_nom_kW > 0:
        y_label = "Potencia [p.u. de P_nom]"
        hline_y = 1.0
    else:
        y_label = "Potencia [kW]"
        hline_y = P_nom_kW

    if x_axis_mode_p == "v (m/s)":
        dfP = df.sort_values("v (m/s)").copy()
        if pot_norm and P_nom_kW > 0:
            for col in y_cols_P:
                dfP[col] = dfP[col] / P_nom_kW

        # FIGURA: POTENCIAS VS VIENTO
        figP = px.line(
            dfP,
            x="v (m/s)",
            y=y_cols_P,
            markers=True,
        )

        figP.update_layout(
            xaxis_title="v (m/s)",
            yaxis_title=y_label,
            legend_title="Etapa",
            hovermode="x unified",
            plot_bgcolor="white",
            margin=dict(l=40, r=40, t=40, b=40),
        )
    else:
        mask_reg2_p = (v_grid >= v_cut_in) & (v_grid <= v_rated)
        dfP_rot = df.loc[mask_reg2_p].sort_values("rpm_rotor").copy()
        dfP_gen = df.loc[mask_reg2_p].sort_values("rpm_gen").copy()

        P_aero_vals = dfP_rot["P_aero (kW)"].values
        P_mec_vals = dfP_gen["P_mec_gen (kW)"].values
        P_out_vals = dfP_gen["P_out (clip) kW"].values

        if pot_norm and P_nom_kW > 0:
            P_aero_vals = P_aero_vals / P_nom_kW
            P_mec_vals = P_mec_vals / P_nom_kW
            P_out_vals = P_out_vals / P_nom_kW

        figP = go.Figure()
        figP.add_trace(
            go.Scatter(
                x=dfP_rot["rpm_rotor"],
                y=P_aero_vals,
                mode="lines+markers",
                name="P_aero (kW)",
                hovertemplate="rpm_rotor = %{x:.1f} rpm<br>P_aero = %{y:.2f}<extra></extra>",
            )
        )
        figP.add_trace(
            go.Scatter(
                x=dfP_gen["rpm_gen"],
                y=P_mec_vals,
                mode="lines+markers",
                name="P_mec_gen (kW)",
                hovertemplate="rpm_gen = %{x:.1f} rpm<br>P_mec_gen = %{y:.2f}<extra></extra>",
            )
        )
        figP.add_trace(
            go.Scatter(
                x=dfP_gen["rpm_gen"],
                y=P_out_vals,
                mode="lines+markers",
                name="P_out (clip) kW",
                hovertemplate="rpm_gen = %{x:.1f} rpm<br>P_out = %{y:.2f}<extra></extra>",
            )
        )

        figP.update_layout(
            xaxis_title="rpm (rotor/gen)",
            yaxis_title=y_label,
            legend_title="Etapa",
            hovermode="x unified",
            plot_bgcolor="white",
            margin=dict(l=40, r=40, t=40, b=40),
        )

    # Fondo con sólo líneas horizontales suaves
    figP.update_xaxes(
        showgrid=False,
        zeroline=False,
    )
    figP.update_yaxes(
        showgrid=True,
        gridcolor="rgba(148,163,184,0.35)",
        zeroline=False,
    )

    # Línea horizontal de potencia nominal (o 1.0 p.u.)
    if P_nom_kW > 0:
        figP.add_hline(
            y=float(hline_y),
            line_dash="dot",
            line_color="rgba(234,179,8,0.9)",
            annotation_text="P_nom",
            annotation_position="bottom right",
            annotation_font_size=11,
            annotation_font_color="rgba(107,114,128,1)",
        )

    if x_axis_mode_p == "v (m/s)":
        # Líneas verticales v_cut-in, v_rated, v_cut-out
        for x_val, label in [
            (v_cut_in, "v_cut-in"),
            (v_rated, "v_rated"),
            (v_cut_out, "v_cut-out"),
        ]:
            if x_val is not None:
                figP.add_vline(
                    x=float(x_val),
                    line_dash="dot",
                    line_color="rgba(148,163,184,0.9)",
                    annotation_text=label,
                    annotation_position="top",
                    annotation_font_size=11,
                    annotation_font_color="rgba(107,114,128,1)",
                )

        # Región sombreada entre v_rated y v_cut-out (potencia limitada)
        if (v_rated is not None) and (v_cut_out is not None):
            figP.add_vrect(
                x0=float(v_rated),
                x1=float(v_cut_out),
                fillcolor="rgba(148,163,184,0.10)",
                layer="below",
                line_width=0,
                annotation_text="Región potencia limitada",
                annotation_position="top left",
                annotation_font_size=11,
                annotation_font_color="rgba(107,114,128,1)",
            )
    else:
        # rpm nominales de rotor y generador como referencia
        try:
            figP.add_vline(
                x=float(rpm_rotor_rated),
                line_dash="dot",
                line_color="rgba(148,163,184,0.8)",
                annotation_text="rpm_rotor_rated",
                annotation_position="top",
                annotation_font_size=11,
                annotation_font_color="rgba(107,114,128,1)",
            )
        except Exception:
            pass
        try:
            figP.add_vline(
                x=float(rpm_gen_rated),
                line_dash="dot",
                line_color="rgba(148,163,184,0.8)",
                annotation_text="rpm_gen_rated",
                annotation_position="top",
                annotation_font_size=11,
                annotation_font_color="rgba(107,114,128,1)",
            )
        except Exception:
            pass

    st.plotly_chart(figP, use_container_width=True)

    # INTERPRETACIÓN TÉCNICA – MODO VIENTO
    st.markdown(
        """
<div class="comment-box">
  <div class="comment-title">🔍 Interpretación técnica</div>
  <p>
    En Región 2, <strong>P_aero</strong> y <strong>P_mec_gen</strong> deberían crecer casi en paralelo;
    si divergen más de unos kW, hay pérdidas mecánicas excesivas o el MPPT no mantiene λ constante.
    Ajusta rodamientos/caja o recalibra el control hasta que la separación sea mínima.
  </p>
  <p>
    En Región 3 controla el gap entre <strong>P_out</strong> y <strong>P_nom</strong>: si el clipping aparece
    muy antes de <em>v_rated</em>, la máquina está sobredimensionada o el generador limita demasiado pronto;
    si nunca clippea, desaprovechas la capacidad del generador. Ajusta P_nom, G o la lógica de derating según el caso.
  </p>
</div>
""",
        unsafe_allow_html=True,
    )

# =====================================================================
# MODO 2: CURVA DEL GENERADOR VS RPM
# =====================================================================
else:

    pot_norm_gen = st.checkbox(
        "Mostrar potencia del generador normalizada (p.u.)",
        value=False,
        key="pot_norm_pu_gen",
    )

    dfG = df.sort_values("rpm_gen").copy()

    y_col = "P_gen_curve (kW)"
    if pot_norm_gen and P_nom_kW > 0:
        dfG[y_col] = dfG[y_col] / P_nom_kW
        y_label = "Potencia generador [p.u. de P_nom]"
        hline_y = 1.0
    else:
        y_label = "Potencia generador [kW]"
        hline_y = P_nom_kW

    # FIGURA: POTENCIA GENERADOR VS RPM
    figG = px.line(
        dfG,
        x="rpm_gen",
        y=y_col,
        markers=True,
    )

    figG.update_layout(
        xaxis_title="rpm generador",
        yaxis_title=y_label,
        legend_title="Variable",
        hovermode="x unified",
        plot_bgcolor="white",
        margin=dict(l=40, r=40, t=40, b=40),
    )

    # Fondo con solo líneas horizontales
    figG.update_xaxes(
        showgrid=False,
        zeroline=False,
    )
    figG.update_yaxes(
        showgrid=True,
        gridcolor="rgba(148,163,184,0.35)",
        zeroline=False,
    )

    # Línea horizontal P_nom (o 1.0 p.u.)
    if P_nom_kW > 0:
        figG.add_hline(
            y=float(hline_y),
            line_dash="dot",
            line_color="rgba(234,179,8,0.9)",
            annotation_text="P_nom",
            annotation_position="bottom right",
            annotation_font_size=11,
            annotation_font_color="rgba(107,114,128,1)",
        )

    # Línea vertical en rpm nominal del generador (si la tienes definida)
    try:
        if rpm_gen_rated is not None:
            figG.add_vline(
                x=float(rpm_gen_rated),
                line_dash="dot",
                line_color="rgba(148,163,184,0.9)",
                annotation_text="rpm_gen_rated",
                annotation_position="top",
                annotation_font_size=11,
                annotation_font_color="rgba(107,114,128,1)",
            )
    except NameError:
        # Si rpm_gen_rated no existe, simplemente no se dibuja la línea
        pass

    st.plotly_chart(figG, use_container_width=True)

    # INTERPRETACIÓN TÉCNICA – MODO GENERADOR
    st.markdown(
        """
<div class="comment-box">
  <div class="comment-title">🔍 Interpretación técnica</div>
  <p>
    Esta vista se centra en el dominio eléctrico: la curva
    <strong>P_gen_curve</strong> muestra cómo crece la potencia del
    generador en función de sus rpm. La pendiente en la zona de bajas rpm
    permite verificar el ajuste entre par, flujo magnético y pérdidas
    internas del generador.
  </p>
  <p>
    El punto <strong>rpm_gen_rated</strong> marca el régimen nominal del
    generador: a partir de allí la potencia se aproxima a
    <em>P_nom</em> y el control debe limitar par o corriente para evitar
    sobrecargas térmicas. Comparar esta curva con
    <strong>P_out</strong> permite validar que la electrónica de potencia
    y la ley de control aprovechan adecuadamente la capacidad del
    generador sin exceder sus límites.
  </p>
</div>
""",
        unsafe_allow_html=True,
    )


# =====================================================================
# CP EQUIVALENTE POR ETAPA
# =====================================================================


# Cp equivalente por etapa
st.subheader("📉 Cp equivalente por etapa")
question_prompt("¿Qué etapa del tren (rotor, eje o salida eléctrica) debería optimizarse primero según la caída de Cp que ves frente al viento?")

# Selector de eje x
x_axis_mode_cp = st.radio(
    "Eje x",
    ("v (m/s)", "rpm"),
    horizontal=True,
    key="x_axis_cp_eq",
    help="Elige si quieres ver Cp equivalente contra velocidad de viento o contra rpm (rotor/gen).",
)

# --- Cálculo de eficiencias locales a partir de los Cp equivalentes ---
Cp_a = df["Cp_aero_equiv"].values
Cp_s = df["Cp_shaft_equiv"].values
Cp_e = df["Cp_el_equiv"].values
eps  = 1e-9

eta_mec_loc  = np.divide(Cp_s, np.maximum(Cp_a, eps))
eta_el_loc   = np.divide(Cp_e, np.maximum(Cp_s, eps))
eta_tot_loc  = np.divide(Cp_e, np.maximum(Cp_a, eps))

df_cp_eq = df.copy()
df_cp_eq["η_mec"]   = eta_mec_loc
df_cp_eq["η_el"]    = eta_el_loc
df_cp_eq["η_total"] = eta_tot_loc

fig_cp_eq = go.Figure()

if x_axis_mode_cp == "v (m/s)":
    df_cp_eq = df_cp_eq.sort_values("v (m/s)").copy()

    # customdata para mostrar eficiencias en el hover
    custom = np.stack(
        [df_cp_eq["η_mec"].values, df_cp_eq["η_el"].values, df_cp_eq["η_total"].values],
        axis=-1,
    )

    # --- Curvas de Cp equivalente por etapa ---
    series = [
        ("Cp_aero_equiv",  "Rotor – Cp_aero"),
        ("Cp_shaft_equiv", "Eje generador – Cp_shaft"),
        ("Cp_el_equiv",    "Salida eléctrica – Cp_el"),
    ]

    for col, name in series:
        fig_cp_eq.add_trace(
            go.Scatter(
                x=df_cp_eq["v (m/s)"],
                y=df_cp_eq[col],
                mode="lines+markers",
                name=name,
                customdata=custom,
                hovertemplate=(
                    "v = %{x:.1f} m/s<br>"
                    "Cp_equiv = %{y:.3f}<br>"
                    "η_mec = %{customdata[0]:.3f}<br>"
                    "η_el = %{customdata[1]:.3f}<br>"
                    "η_total = %{customdata[2]:.3f}<extra></extra>"
                ),
            )
        )
else:
    mask_reg2_cp = (v_grid >= v_cut_in) & (v_grid <= v_rated)
    df_cp_rot = df_cp_eq.loc[mask_reg2_cp].sort_values("rpm_rotor").copy()
    df_cp_gen = df_cp_eq.loc[mask_reg2_cp].sort_values("rpm_gen").copy()

    custom_rot = np.stack(
        [df_cp_rot["η_mec"].values, df_cp_rot["η_el"].values, df_cp_rot["η_total"].values],
        axis=-1,
    )
    custom_gen = np.stack(
        [df_cp_gen["η_mec"].values, df_cp_gen["η_el"].values, df_cp_gen["η_total"].values],
        axis=-1,
    )

    fig_cp_eq.add_trace(
        go.Scatter(
            x=df_cp_rot["rpm_rotor"],
            y=df_cp_rot["Cp_aero_equiv"],
            mode="lines+markers",
            name="Rotor – Cp_aero",
            customdata=custom_rot,
            hovertemplate=(
                "rpm_rotor = %{x:.1f} rpm<br>"
                "Cp_equiv = %{y:.3f}<br>"
                "η_mec = %{customdata[0]:.3f}<br>"
                "η_el = %{customdata[1]:.3f}<br>"
                "η_total = %{customdata[2]:.3f}<extra></extra>"
            ),
        )
    )
    fig_cp_eq.add_trace(
        go.Scatter(
            x=df_cp_gen["rpm_gen"],
            y=df_cp_gen["Cp_shaft_equiv"],
            mode="lines+markers",
            name="Eje generador – Cp_shaft",
            customdata=custom_gen,
            hovertemplate=(
                "rpm_gen = %{x:.1f} rpm<br>"
                "Cp_equiv = %{y:.3f}<br>"
                "η_mec = %{customdata[0]:.3f}<br>"
                "η_el = %{customdata[1]:.3f}<br>"
                "η_total = %{customdata[2]:.3f}<extra></extra>"
            ),
        )
    )
    fig_cp_eq.add_trace(
        go.Scatter(
            x=df_cp_gen["rpm_gen"],
            y=df_cp_gen["Cp_el_equiv"],
            mode="lines+markers",
            name="Salida eléctrica – Cp_el",
            customdata=custom_gen,
            hovertemplate=(
                "rpm_gen = %{x:.1f} rpm<br>"
                "Cp_equiv = %{y:.3f}<br>"
                "η_mec = %{customdata[0]:.3f}<br>"
                "η_el = %{customdata[1]:.3f}<br>"
                "η_total = %{customdata[2]:.3f}<extra></extra>"
            ),
        )
    )

# --- Línea horizontal: límite de Betz ---
CP_BETZ = 16.0 / 27.0
fig_cp_eq.add_hline(
    y=CP_BETZ,
    line_dash="dot",
    line_color="rgba(234,179,8,0.9)",
    annotation_text="Límite de Betz",
    annotation_position="top left",
)

if x_axis_mode_cp == "v (m/s)":
    # --- Líneas verticales: v_cut-in / v_rated / v_cut-out ---
    for x_val, label in [
        (v_cut_in,  "v_cut-in"),
        (v_rated,   "v_rated"),
        (v_cut_out, "v_cut-out"),
    ]:
        fig_cp_eq.add_vline(
            x=float(x_val),
            line_dash="dot",
            line_color="rgba(148,163,184,0.8)",
            annotation_text=label,
            annotation_position="top",
        )

    # --- Región sombreada: operación nominal (potencia constante) ---
    fig_cp_eq.add_vrect(
        x0=float(v_rated),
        x1=float(v_cut_out),
        fillcolor="rgba(148,163,184,0.15)",
        line_width=0,
        layer="below",
        annotation_text="Región potencia limitada",
        annotation_position="top right",
    )
else:
    for x_val, label in [
        (rpm_rotor_rated, "rpm_rotor_rated"),
        (rpm_gen_rated,   "rpm_gen_rated"),
    ]:
        try:
            fig_cp_eq.add_vline(
                x=float(x_val),
                line_dash="dot",
                line_color="rgba(148,163,184,0.8)",
                annotation_text=label,
                annotation_position="top",
            )
        except Exception:
            continue

# --- Estilo de ejes ---
fig_cp_eq.update_xaxes(
    title_text="v (m/s)" if x_axis_mode_cp == "v (m/s)" else "rpm (rotor/gen)",
    showgrid=False,
    zeroline=False,
)

fig_cp_eq.update_yaxes(
    title_text="Cp equivalente",
    showgrid=True,
    gridcolor="rgba(148,163,184,0.35)",
    zeroline=False,
)

# --- Layout global + hover unificado ---
fig_cp_eq.update_layout(
    legend_title="Etapa",
    margin=dict(l=60, r=20, t=40, b=40),
    plot_bgcolor="white",
    hovermode="x unified",  # 🔍 tooltip unificado en X
    hoverlabel=dict(
        bgcolor="white",
        font_size=12,
        font_color="black",
    ),
)
st.plotly_chart(fig_cp_eq, use_container_width=True)
st.markdown("""
<div class="comment-box">
  <div class="comment-title">🔍 Interpretación técnica (Cp equivalente por etapa)</div>
  <p>
    Usa la caída entre <strong>Cp_aero</strong>, <strong>Cp_shaft</strong> y <strong>Cp_el</strong> para decidir dónde actuar primero:
    si <strong>Cp_aero</strong> cae antes de <em>v_rated</em>, el control MPPT está soltando TSR y conviene retocar el setpoint o subir la solidez;
    si la brecha <strong>Cp_aero → Cp_shaft</strong> crece con el viento, revisa rodamientos y caja porque los pares pico están disparando pérdidas.
  </p>
  <p>
    En Región 3 la diferencia <strong>Cp_shaft → Cp_el</strong> revela si estás limitando por rendimiento del generador o por clipping electrónico:
    una caída temprana implica sobredimensionar el generador o redistribuir P_nom; si la caída ocurre solo en la curva eléctrica, ajusta límites de inversor.
    Mantén las tres curvas dentro de una pendiente gradual; cualquier quiebre te dice exactamente qué etapa no soporta el escalamiento del piloto.
  </p>
</div>
""", unsafe_allow_html=True)




# =========================================================
# PÉRDIDAS POR ETAPA (MECÁNICA, GENERADOR, ELECTRÓNICA, CLIPPING)
# =========================================================
st.subheader("🔍 Pérdidas por etapa (mecánica, generador, electrónica, clipping)")
question_prompt("¿Qué componente quieres atacar primero para reducir pérdidas cuando pases de la región MPPT a la potencia limitada?")

# Selector de eje x
x_axis_mode_loss = st.radio(
    "Eje x",
    ("v (m/s)", "rpm"),
    horizontal=True,
    key="x_axis_loss",
    help="Elige si quieres ver pérdidas contra velocidad de viento o contra rpm del rotor.",
)

if x_axis_mode_loss == "v (m/s)":
    x_col_loss = "v (m/s)"
else:
    x_col_loss = "rpm_rotor"

dfL = df.sort_values(x_col_loss).copy()

# --- detectar columnas de pérdidas por patrón, sin depender del nombre exacto ---
loss_cols = [
    c for c in dfL.columns
    if any(pat in c for pat in ["Pérdida", "Perdida", "loss", "Loss"])
]

# opcional: excluir una columna de pérdida total si la tuvieras
loss_cols = [c for c in loss_cols if "total" not in c.lower()]

if len(loss_cols) == 0:
    st.warning("No se encontraron columnas de pérdidas en el DataFrame. Revisa los nombres de columnas.")
else:
    fig_loss = px.area(
        dfL,
        x=x_col_loss,
        y=loss_cols,
    )

    fig_loss.update_layout(
        xaxis_title="v (m/s)" if x_axis_mode_loss == "v (m/s)" else "rpm",
        yaxis_title="Pérdidas [kW]",
        legend_title="Etapa",
        hovermode="x unified",
        plot_bgcolor="white",
        margin=dict(l=40, r=40, t=40, b=40),
    )

    # Fondo con solo líneas horizontales suaves
    fig_loss.update_xaxes(showgrid=False, zeroline=False)
    fig_loss.update_yaxes(
        showgrid=True,
        gridcolor="rgba(148,163,184,0.35)",
        zeroline=False,
    )

    if x_axis_mode_loss == "v (m/s)":
        # Líneas verticales v_cut-in / v_rated / v_cut-out
        for x_val, label in [
            (v_cut_in, "v_cut-in"),
            (v_rated,  "v_rated"),
            (v_cut_out,"v_cut-out"),
        ]:
            if x_val is not None:
                fig_loss.add_vline(
                    x=float(x_val),
                    line_dash="dot",
                    line_color="rgba(148,163,184,0.9)",
                    annotation_text=label,
                    annotation_position="top",
                    annotation_font_size=11,
                    annotation_font_color="rgba(107,114,128,1)",
                )

        # Región potencia limitada
        if (v_rated is not None) and (v_cut_out is not None):
            fig_loss.add_vrect(
                x0=float(v_rated),
                x1=float(v_cut_out),
                fillcolor="rgba(148,163,184,0.10)",
                layer="below",
                line_width=0,
                annotation_text="Región potencia limitada",
                annotation_position="top left",
                annotation_font_size=11,
                annotation_font_color="rgba(107,114,128,1)",
            )
    else:
        try:
            fig_loss.add_vline(
                x=float(rpm_rotor_rated),
                line_dash="dot",
                line_color="rgba(148,163,184,0.9)",
                annotation_text="rpm_rotor_rated",
                annotation_position="top",
                annotation_font_size=11,
                annotation_font_color="rgba(107,114,128,1)",
            )
        except Exception:
            pass
    st.plotly_chart(fig_loss, use_container_width=True)

    # ===========================
    # INTERPRETACIÓN TÉCNICA
    # ===========================
    st.markdown(
        """
<div class="comment-box">
  <div class="comment-title">🔍 Interpretación técnica</div>
  <p>
    Prioriza la barra que más crece en la región sombreada: si las pérdidas mecánicas dominan antes de <em>v_rated</em>,
    reduce cargas en rodamientos (menor G o mejores sellos); si el generador se dispara después, necesitas mejor
    ventilación o un modelo con menor R<sub>s</sub>.
  </p>
  <p>
    Cuando el área de <em>clipping</em> supera al resto ya no ganas nada subiendo Cp: toca subir P_nom o suavizar el perfil MPPT.
    Mantén la contribución de cada banda por debajo del 10–15&nbsp;% de la potencia útil para asegurar que el piloto llegue
    competitivo al escalado de 80 kW.
  </p>
</div>
""",
        unsafe_allow_html=True,
    )

# ==========================================================
# Eficiencias por etapa
# ==========================================================
st.subheader("📈 Eficiencias: mecánica, generador y global")
question_prompt("¿Cuál es la eficiencia mínima aceptable en cada etapa antes de considerar rediseño o cambio de proveedor?")

# Selector de eje x
x_axis_mode_eff = st.radio(
    "Eje x",
    ("v (m/s)", "rpm"),
    horizontal=True,
    key="x_axis_eff",
    help="Elige si quieres ver eficiencias contra velocidad de viento o contra rpm (rotor/gen).",
)

# --- Vectores base (en W) ---
v_axis      = v_grid                      # o df["v (m/s)"].values
P_aero      = P_aero_W                    # Potencia aerodinámica
P_mec       = P_mec_gen_W                 # Potencia mecánica en eje generador
P_el_before = P_el_gen_W                  # Potencia eléctrica antes de electrónica
P_out       = P_el_ac_clip                # Potencia de salida tras electrónica + clipping

eta_mec_pct = 100 * np.divide(
    P_mec, P_aero,
    out=np.zeros_like(P_aero),
    where=(P_aero > 0)
)
eta_gen_pct = 100 * np.divide(
    P_el_before, P_mec,
    out=np.zeros_like(P_mec),
    where=(P_mec > 0)
)
eta_tot_pct = 100 * np.divide(
    P_out, P_aero,
    out=np.zeros_like(P_aero),
    where=(P_aero > 0)
)

figE = go.Figure()

if x_axis_mode_eff == "v (m/s)":
    eff_df = pd.DataFrame({
        "v (m/s)":      v_axis,
        "η_mec [%]":   np.round(eta_mec_pct, 1),
        "η_gen [%]":   np.round(eta_gen_pct, 1),
        "η_total [%]": np.round(eta_tot_pct, 1),
    })

    figE = px.line(
        eff_df,
        x="v (m/s)",
        y=["η_mec [%]", "η_gen [%]", "η_total [%]"],
        markers=True,
    )

    figE.update_traces(
        line=dict(width=2.4),
        marker=dict(size=7),
        hovertemplate=(
            "v = %{x:.1f} m/s<br>"
            "%{y:.1f} %<extra>%{fullData.name}</extra>"
        ),
    )
else:
    mask_reg2_eff = (v_grid >= v_cut_in) & (v_grid <= v_rated)
    eff_rot_df = pd.DataFrame({
        "rpm_rotor": rpm_rotor[mask_reg2_eff],
        "η_mec [%]": np.round(eta_mec_pct[mask_reg2_eff], 1),
    }).sort_values("rpm_rotor")

    eff_gen_df = pd.DataFrame({
        "rpm_gen":   rpm_gen[mask_reg2_eff],
        "η_gen [%]": np.round(eta_gen_pct[mask_reg2_eff], 1),
        "η_total [%]": np.round(eta_tot_pct[mask_reg2_eff], 1),
    }).sort_values("rpm_gen")

    figE.add_trace(
        go.Scatter(
            x=eff_rot_df["rpm_rotor"],
            y=eff_rot_df["η_mec [%]"],
            mode="lines+markers",
            name="η_mec [%]",
            hovertemplate="rpm_rotor = %{x:.1f} rpm<br>%{y:.1f} %<extra></extra>",
        )
    )
    figE.add_trace(
        go.Scatter(
            x=eff_gen_df["rpm_gen"],
            y=eff_gen_df["η_gen [%]"],
            mode="lines+markers",
            name="η_gen [%]",
            hovertemplate="rpm_gen = %{x:.1f} rpm<br>%{y:.1f} %<extra></extra>",
        )
    )
    figE.add_trace(
        go.Scatter(
            x=eff_gen_df["rpm_gen"],
            y=eff_gen_df["η_total [%]"],
            mode="lines+markers",
            name="η_total [%]",
            hovertemplate="rpm_gen = %{x:.1f} rpm<br>%{y:.1f} %<extra></extra>",
        )
    )

# Layout general + hover unificado
figE.update_layout(
    xaxis_title="v (m/s)" if x_axis_mode_eff == "v (m/s)" else "rpm (rotor/gen)",
    yaxis_title="Eficiencia [%]",
    legend_title="Etapa",
    hovermode="x unified",         # 👈 cuadro único con las 3 eficiencias
    plot_bgcolor="white",
    margin=dict(l=50, r=20, t=40, b=40),
    hoverlabel=dict(
        bgcolor="white",
        font_size=12,
        font_color="black",
    ),
)

# Fondo con solo grilla horizontal
figE.update_xaxes(showgrid=False, zeroline=False)
figE.update_yaxes(
    showgrid=True,
    gridcolor="rgba(148,163,184,0.35)",
    zeroline=False,
)

if x_axis_mode_eff == "v (m/s)":
    # --- Líneas verticales: cut-in / rated / cut-out ---
    for x_val, label in [
        (v_cut_in,  "v_cut-in"),
        (v_rated,   "v_rated"),
        (v_cut_out, "v_cut-out"),
    ]:
        figE.add_vline(
            x=float(x_val),
            line_dash="dot",
            line_color="rgba(148,163,184,0.8)",
            annotation_text=label,
            annotation_position="top",
            annotation_font_size=11,
            annotation_font_color="rgba(107,114,128,1)",
        )

    # --- Región sombreada: potencia limitada (IEC 61400-2 para <200 kW) ---
    figE.add_vrect(
        x0=float(v_rated),
        x1=float(v_cut_out),
        fillcolor="rgba(148,163,184,0.10)",
        line_width=0,
        layer="below",
        annotation_text="Región potencia constante / IEC 61400-2",
        annotation_position="top right",
        annotation_font_size=11,
        annotation_font_color="rgba(107,114,128,1)",
    )
else:
    for x_val, label in [
        (rpm_rotor_rated, "rpm_rotor_rated"),
        (rpm_gen_rated,   "rpm_gen_rated"),
    ]:
        try:
            figE.add_vline(
                x=float(x_val),
                line_dash="dot",
                line_color="rgba(148,163,184,0.8)",
                annotation_text=label,
                annotation_position="top",
                annotation_font_size=11,
                annotation_font_color="rgba(107,114,128,1)",
            )
        except Exception:
            continue
st.plotly_chart(figE, use_container_width=True)

st.markdown("""
<div class="comment-box">
  <div class="comment-title">🔍 Interpretación técnica</div>
  <p>
    Fija umbrales claros: <strong>η_mec</strong> &gt; 92&nbsp;% en la banda MPPT asegura que la caja soporta el par objetivo;
    si cae más de 3 puntos frente a <strong>v_rated</strong>, alarga G o mejora lubricación.
  </p>
  <p>
    Mantén <strong>η_gen</strong> plana en 94–97&nbsp;%; si encuentras un diente al entrar en Región 3 es síntoma de saturación del generador o desfase de PF.
    <strong>η_total</strong> debe caer solo cuando active el clipping: si baja antes, revisa la cadena completa (TSR, pérdidas o electrónica).
  </p>
</div>
""", unsafe_allow_html=True)

st.caption(
    "η_total = P_out / P_aero. Si la curva de 'Pérdida por clipping' domina desde cierta v, "
    "estás en región de potencia constante; considera redimensionar G/TSR o estrategia de control."
)


# Bloque – Tren mecánico y cargas
section_header("🛠️ Tren mecánico y cargas")

# =========================================================
# Gráfico – Momento flector en raíz de pala
# =========================================================
st.subheader("🧱 Momento flector en unión pala–struts")
question_prompt("¿En qué bins el momento flector supera tu límite FEM y cómo influye el brazo efectivo o la masa de pala?")

# Selector de eje x
x_axis_mode_moment = st.radio(
    "Eje x",
    ("v (m/s)", "rpm"),
    horizontal=True,
    key="x_axis_moment",
    help="Elige si quieres ver el momento contra velocidad de viento o contra rpm del rotor.",
)

if x_axis_mode_moment == "v (m/s)":
    x_col_moment = "v (m/s)"
    hover_x_moment = "v = %{x:.1f} m/s"
else:
    x_col_moment = "rpm_rotor"
    hover_x_moment = "rpm = %{x:.0f} rpm"

df_moment = df.sort_values(x_col_moment).copy()

fig_mbase = go.Figure()
fig_mbase.add_trace(
    go.Scatter(
        x=df_moment[x_col_moment],
        y=df_moment["M_base (kN·m)"],
        mode="lines+markers",
        name="M_base (kN·m)",
        line=dict(width=2.8),
        marker=dict(size=7),
        hovertemplate=hover_x_moment + "<br>M_base = %{y:.1f} kN·m<extra></extra>",
    )
)

if M_base_max_iec > 0:
    fig_mbase.add_hline(
        y=float(M_base_max_iec),
        line_dash="dot",
        line_color="rgba(239,68,68,0.9)",
        annotation_text=f"Límite IEC = {M_base_max_iec:.0f} kN·m",
        annotation_position="bottom right",
    )

fig_mbase.update_layout(
    xaxis_title="v (m/s)" if x_axis_mode_moment == "v (m/s)" else "rpm",
    yaxis_title="Momento flector raíz [kN·m]",
    hovermode="x unified",
    plot_bgcolor="white",
    margin=dict(l=60, r=20, t=40, b=40),
    legend_title="Magnitud",
)
fig_mbase.update_xaxes(showgrid=False, zeroline=False)
fig_mbase.update_yaxes(
    showgrid=True,
    gridcolor="rgba(148,163,184,0.35)",
    zeroline=False,
)

if x_axis_mode_moment == "v (m/s)":
    for x_val, label in [
        (v_cut_in, "v_cut-in"),
        (v_rated, "v_rated"),
        (v_cut_out, "v_cut-out"),
    ]:
        fig_mbase.add_vline(
            x=float(x_val),
            line_dash="dot",
            line_color="rgba(148,163,184,0.6)",
            annotation_text=label,
            annotation_position="top",
            annotation_font_size=11,
            annotation_font_color="rgba(107,114,128,1)",
        )
else:
    try:
        fig_mbase.add_vline(
            x=float(rpm_rotor_rated),
            line_dash="dot",
            line_color="rgba(148,163,184,0.6)",
            annotation_text="rpm_rotor_rated",
            annotation_position="top",
            annotation_font_size=11,
            annotation_font_color="rgba(107,114,128,1)",
        )
    except Exception:
        pass

st.plotly_chart(fig_mbase, use_container_width=True)

st.markdown("""
<div class="comment-box">
  <div class="comment-title">🔍 Lectura rápida (M_base)</div>
  <p>
    La curva muestra cómo el momento flector combinado (torque aerodinámico por pala + efecto centrífugo) escala con el viento.
    Si cruza la línea IEC antes de <em>v_rated</em>, necesitas reducir masa, brazo efectivo o ajustar la ley de rpm para aliviar la raíz.
  </p>
</div>
""", unsafe_allow_html=True)

# ==========================================================
# Tensiones en struts (selector de gráficos)
# ==========================================================
st.subheader("🧩 Tensiones en struts")
question_prompt("¿El área efectiva de strut aporta suficiente margen frente al esfuerzo axial estimado?")

# Selector de eje x
x_axis_mode_strut = st.radio(
    "Eje x",
    ("v (m/s)", "rpm"),
    horizontal=True,
    key="x_axis_strut",
    help="Elige si quieres ver las tensiones contra velocidad de viento o contra rpm del rotor.",
)

if x_axis_mode_strut == "v (m/s)":
    x_strut = v_grid
    xaxis_title_strut = "v (m/s)"
    hover_x_strut = "v = %{x:.1f} m/s"
    title_suffix_strut = "viento"
else:
    x_strut = rpm_rotor
    xaxis_title_strut = "rpm"
    hover_x_strut = "rpm = %{x:.0f} rpm"
    title_suffix_strut = "rpm"

show_both_strut = st.checkbox("Mostrar ambos gráficos (tensión + margen)", value=True)

fig_strut = go.Figure()
fig_strut.add_trace(
    go.Scatter(
        x=x_strut,
        y=sigma_strut_MPa,
        mode="lines+markers",
        name="σ_strut (MPa)",
        hovertemplate=hover_x_strut + "<br>σ_strut = %{y:.1f} MPa<extra></extra>",
    )
)
if allow_strut_MPa > 0:
    fig_strut.add_hline(
        y=float(allow_strut_MPa),
        line_dash="dot",
        line_color="rgba(239,68,68,0.9)",
        annotation_text=f"σ_admisible/FS = {allow_strut_MPa:.1f} MPa",
        annotation_position="top right",
    )
fig_strut.update_layout(
    title=f"Tensión axial en struts vs {title_suffix_strut}",
    xaxis_title=xaxis_title_strut,
    yaxis_title="σ_strut (MPa)",
    hovermode="x unified",
    plot_bgcolor="white",
    margin=dict(l=50, r=20, t=50, b=40),
)
fig_strut.update_xaxes(showgrid=False, zeroline=False)
fig_strut.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.35)", zeroline=False)
if x_axis_mode_strut == "v (m/s)":
    for x_val, label in [
        (v_cut_in, "v_cut-in"),
        (v_rated, "v_rated"),
        (v_cut_out, "v_cut-out"),
    ]:
        fig_strut.add_vline(
            x=float(x_val),
            line_dash="dot",
            line_color="rgba(148,163,184,0.6)",
            annotation_text=label,
            annotation_position="top",
            annotation_font_size=11,
            annotation_font_color="rgba(107,114,128,1)",
        )
else:
    try:
        fig_strut.add_vline(
            x=float(rpm_rotor_rated),
            line_dash="dot",
            line_color="rgba(148,163,184,0.6)",
            annotation_text="rpm_rotor_rated",
            annotation_position="top",
            annotation_font_size=11,
            annotation_font_color="rgba(107,114,128,1)",
        )
    except Exception:
        pass
st.plotly_chart(fig_strut, use_container_width=True)

if show_both_strut:
    fig_strut_margin = go.Figure()
    fig_strut_margin.add_trace(
        go.Scatter(
            x=x_strut,
            y=margin_strut * 100.0,
            mode="lines+markers",
            name="Margen σ_strut (%)",
            hovertemplate=hover_x_strut + "<br>margen = %{y:.1f}%<extra></extra>",
        )
    )
    fig_strut_margin.add_hline(
        y=0.0,
        line_dash="dot",
        line_color="rgba(239,68,68,0.9)",
        annotation_text="Margen = 0%",
        annotation_position="top right",
    )
    fig_strut_margin.update_layout(
        title=f"Margen en struts vs {title_suffix_strut}",
        xaxis_title=xaxis_title_strut,
        yaxis_title="Margen (%)",
        hovermode="x unified",
        plot_bgcolor="white",
        margin=dict(l=50, r=20, t=50, b=40),
    )
    fig_strut_margin.update_xaxes(showgrid=False, zeroline=False)
    fig_strut_margin.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.35)", zeroline=False)
    if x_axis_mode_strut == "v (m/s)":
        for x_val, label in [
            (v_cut_in, "v_cut-in"),
            (v_rated, "v_rated"),
            (v_cut_out, "v_cut-out"),
        ]:
            fig_strut_margin.add_vline(
                x=float(x_val),
                line_dash="dot",
                line_color="rgba(148,163,184,0.6)",
                annotation_text=label,
                annotation_position="top",
                annotation_font_size=11,
                annotation_font_color="rgba(107,114,128,1)",
            )
    else:
        try:
            fig_strut_margin.add_vline(
                x=float(rpm_rotor_rated),
                line_dash="dot",
                line_color="rgba(148,163,184,0.6)",
                annotation_text="rpm_rotor_rated",
                annotation_position="top",
                annotation_font_size=11,
                annotation_font_color="rgba(107,114,128,1)",
            )
        except Exception:
            pass
    st.plotly_chart(fig_strut_margin, use_container_width=True)

st.markdown(f"""
<div class="comment-box">
  <div class="comment-title">🔍 Interpretación técnica (struts)</div>
  <p>
    Si la curva de <strong>σ_strut</strong> supera la línea de admisible/FS antes de <em>v_rated</em>,
    necesitas aumentar el área efectiva, reducir el brazo efectivo o bajar TSR para aliviar la carga.
    Revisa el margen: valores negativos indican sobre-esfuerzo.
  </p>
  <p>
    En Región 3 (potencia limitada) el control debería estabilizar la pendiente; si sigue creciendo rápido,
    revisa la ley de rpm o el setpoint de par del generador.
  </p>
</div>
""", unsafe_allow_html=True)


# ==========================================================
# Torque (rotor y generador)
# ==========================================================
st.subheader("🧲 Torque (rotor y generador) ")
question_prompt("¿En qué bin de viento el par se acerca más a tus límites estructurales IEC y requiere estrategias de mitigación?")

# Datos importantes del generador (ficha técnica)
T_gen_nom = GDG_RATED_T_Nm   # 3460 N·m
I_nom     = GDG_RATED_I
T_gen_safe = T_gen_nom * 1.10  # umbral “zona amarilla”

# Ordenar por viento
dfT = df.sort_values("v (m/s)").copy()

# Selector de eje x
x_axis_mode = st.radio(
    "Eje x",
    ("v (m/s)", "rpm"),
    horizontal=True,
    key="x_axis_torque",
    help="Elige si quieres ver el par contra velocidad de viento o contra rpm.",
)

if x_axis_mode == "v (m/s)":
    x_col = "v (m/s)"
    # Pasar a formato largo para usar px.line
    dfT_long = dfT.melt(
        id_vars=[x_col],
        value_vars=["T_rotor (N·m)", "T_gen (N·m)"],
        var_name="Variable",
        value_name="T [N·m]",
    )
else:
    x_col = "rpm"
    dfT_long = pd.DataFrame(
        {
            x_col: np.concatenate([dfT["rpm_rotor"].values, dfT["rpm_gen"].values]),
            "T [N·m]": np.concatenate([dfT["T_rotor (N·m)"].values, dfT["T_gen (N·m)"].values]),
            "Variable": (["T_rotor (N·m)"] * len(dfT)) + (["T_gen (N·m)"] * len(dfT)),
        }
    )

# Mapa más legible de nombres
dfT_long["Variable"] = dfT_long["Variable"].map({
    "T_rotor (N·m)": "T_rotor (N·m)",
    "T_gen (N·m)":   "T_gen (N·m)",
})

# FIGURA BASE
figT = px.line(
    dfT_long,
    x=x_col,
    y="T [N·m]",
    color="Variable",
    markers=True,
)

# Estilo general coherente con el resto
figT.update_layout(
    xaxis_title="v (m/s)" if x_axis_mode == "v (m/s)" else "rpm",
    yaxis_title="Par [N·m]",
    legend_title="Variable",
    hovermode="x unified",
    plot_bgcolor="white",
    margin=dict(l=40, r=40, t=40, b=40),
)

figT.update_xaxes(
    showgrid=False,
    zeroline=False,
)
figT.update_yaxes(
    showgrid=True,
    gridcolor="rgba(148,163,184,0.35)",
    zeroline=False,
)

# Hover más técnico
hover_x = "v = %{x:.1f} m/s" if x_axis_mode == "v (m/s)" else "rpm = %{x:.0f} rpm"
figT.update_traces(
    hovertemplate=(
        f"{hover_x}<br>"
        "%{fullData.name} = %{y:,.0f} N·m<extra></extra>"
    )
)

# ----------------------------------------------------------
# CAPAS IEC / LÍMITES
# ----------------------------------------------------------

# Línea horizontal: torque nominal del generador
figT.add_hline(
    y=float(T_gen_nom),
    line_dash="dot",
    line_color="rgba(234,179,8,0.95)",
    annotation_text=f"T_nom gen ({T_gen_nom:.0f} N·m)",
    annotation_position="bottom right",
    annotation_font_size=11,
    annotation_font_color="rgba(107,114,128,1)",
)


# Región “safe” de par generador (0 – T_nom) en color muy suave
figT.add_hrect(
    y0=0.0,
    y1=float(T_gen_nom),
    fillcolor="rgba(34,197,94,0.05)",
    line_width=0,
    layer="below",
)

# Región de sobre-torque generador (T_nom – T_gen_safe)
figT.add_hrect(
    y0=float(T_gen_nom),
    y1=float(max(dfT["T_gen (N·m)"].max(), T_gen_safe)),
    fillcolor="rgba(239,68,68,0.06)",
    line_width=0,
    layer="below",
)

# Límite IEC de par rotor (si está definido en el sidebar)
try:
    if T_rotor_max_iec > 0:
        figT.add_hline(
            y=float(T_rotor_max_iec),
            line_dash="dash",
            line_color="rgba(239,68,68,0.9)",
            annotation_text="Límite IEC T_rotor",
            annotation_position="top right",
            annotation_font_size=11,
            annotation_font_color="rgba(127,29,29,1)",
        )
except NameError:
    pass

if x_axis_mode == "v (m/s)":
    # v_rated, v_cut-out y v_shutdown IEC
    for x_val, label in [
        (v_rated,        "v_rated"),
        (v_cut_out,      "v_cut-out"),
        ("_shutdown_",   "v_shutdown IEC"),
    ]:
        try:
            if label == "v_shutdown IEC":
                x_draw = float(v_shutdown_iec)
            else:
                x_draw = float(x_val)

            figT.add_vline(
                x=x_draw,
                line_dash="dot" if label != "v_shutdown IEC" else "dash",
                line_color="rgba(148,163,184,0.8)" if label != "v_shutdown IEC" else "rgba(239,68,68,0.9)",
                annotation_text=label,
                annotation_position="top",
                annotation_font_size=11,
                annotation_font_color="rgba(107,114,128,1)",
            )
        except Exception:
            # Si alguna no está definida, simplemente no se dibuja
            continue
else:
    # rpm nominales de rotor y generador
    for x_val, label, color in [
        (rpm_rotor_rated, "rpm_rotor_rated", "rgba(148,163,184,0.8)"),
        (rpm_gen_rated,   "rpm_gen_rated",   "rgba(239,68,68,0.9)"),
    ]:
        try:
            figT.add_vline(
                x=float(x_val),
                line_dash="dot",
                line_color=color,
                annotation_text=label,
                annotation_position="top",
                annotation_font_size=11,
                annotation_font_color="rgba(107,114,128,1)",
            )
        except Exception:
            continue
st.plotly_chart(figT, use_container_width=True)

st.markdown("""
<div class="comment-box">
  <div class="comment-title">🔍 Interpretación técnica (Par)</div>
  <p>
    Usa el cruce entre <strong>T_rotor</strong> y <strong>T_gen</strong> con las franjas verde/roja para definir tu política de protección:
    si más del 5&nbsp;% del rango operativo cae en la zona roja, reduce la pendiente MPPT o baja G para descargar la caja.
  </p>
  <p>
    Valida también la línea IEC: si <strong>T_rotor</strong> toca el límite antes de <em>v_shutdown</em>, necesitas reforzar la estructura
    o subir el setpoint de shutdown. Esta lectura define el dimensionamiento definitivo de rodamientos y carcasas.
  </p>
</div>
""", unsafe_allow_html=True)

# =========================================================
# Módulo 4 – Envolvente T–rpm del generador (mapa operativo)
# =========================================================
# =========================================================
# Módulo 4 – Envolvente T–rpm del generador (mapa operativo)
# =========================================================
st.subheader("📐 Envolvente T–rpm del generador")
question_prompt("¿Permitirías sobrepasos breves de par fuera de la zona verde antes de activar la protección?")

# Datos base desde la simulación
rpm_gen_arr = df["rpm_gen"].values
T_gen_arr   = df["T_gen (N·m)"].values

# Punto nominal de ficha
rpm_nom_gen = GDG_RATED_RPM
T_nom_gen   = GDG_RATED_T_Nm

# Límites “sugeridos” para zonas de operación
rpm_safe_max    = 1.05 * rpm_nom_gen   # 105% de rpm_nom
T_safe_max      = 1.00 * T_nom_gen     # 100% de T_nom
rpm_warning_max = 1.15 * rpm_nom_gen   # 115%
T_warning_max   = 1.20 * T_nom_gen     # 120%

# ==========================
# FIGURA BASE
# ==========================
fig_env = go.Figure()

# Curva de operación simulada
fig_env.add_trace(
    go.Scatter(
        x=rpm_gen_arr,
        y=T_gen_arr,
        mode="lines+markers",
        name="Operación simulada",
        hovertemplate=(
            "rpm_gen = %{x:.0f} rpm<br>"
            "T_gen = %{y:,.0f} N·m<extra></extra>"
        ),
    )
)

# Punto nominal del generador
fig_env.add_trace(
    go.Scatter(
        x=[rpm_nom_gen],
        y=[T_nom_gen],
        mode="markers+text",
        name="Punto nominal generador",
        marker=dict(size=10, symbol="x"),
        text=["Nominal"],
        textposition="top right",
        hovertemplate=(
            "rpm_nom = %{x:.0f} rpm<br>"
            "T_nom = %{y:,.0f} N·m<extra></extra>"
        ),
    )
)

# ==========================
# ZONAS DE OPERACIÓN
# ==========================
# 1) Zona segura (verde)
fig_env.add_shape(
    type="rect",
    x0=0, y0=0,
    x1=rpm_safe_max, y1=T_safe_max,
    fillcolor="rgba(34,197,94,0.10)",
    line=dict(width=0),
    layer="below",
)

# 2) Zona de advertencia (amarillo)
fig_env.add_shape(
    type="rect",
    x0=0, y0=T_safe_max,
    x1=rpm_warning_max, y1=T_warning_max,
    fillcolor="rgba(234,179,8,0.10)",
    line=dict(width=0),
    layer="below",
)
fig_env.add_shape(
    type="rect",
    x0=rpm_safe_max, y0=0,
    x1=rpm_warning_max, y1=T_safe_max,
    fillcolor="rgba(234,179,8,0.05)",
    line=dict(width=0),
    layer="below",
)

# 3) Zona fuera de envolvente (rojo)
fig_env.add_shape(
    type="rect",
    x0=0,
    y0=T_warning_max,
    x1=max(rpm_gen_arr.max(), rpm_warning_max * 1.05),
    y1=max(T_gen_arr.max(), T_warning_max * 1.05),
    fillcolor="rgba(239,68,68,0.08)",
    line=dict(width=0),
    layer="below",
)
fig_env.add_shape(
    type="rect",
    x0=rpm_warning_max,
    y0=0,
    x1=max(rpm_gen_arr.max(), rpm_warning_max * 1.05),
    y1=max(T_gen_arr.max(), T_warning_max * 1.05),
    fillcolor="rgba(239,68,68,0.04)",
    line=dict(width=0),
    layer="below",
)

# ==========================
# LÍNEAS GUIA NOMINALES
# ==========================
fig_env.add_vline(
    x=float(rpm_nom_gen),
    line_dash="dot",
    line_color="rgba(148,163,184,0.9)",
    annotation_text="rpm_nom gen",
    annotation_position="top left",
    annotation_font_size=11,
    annotation_font_color="rgba(107,114,128,1)",
)
fig_env.add_hline(
    y=float(T_nom_gen),
    line_dash="dot",
    line_color="rgba(148,163,184,0.9)",
    annotation_text="T_nom gen",
    annotation_position="bottom right",
    annotation_font_size=11,
    annotation_font_color="rgba(107,114,128,1)",
)

# Estilo global coherente con el resto del dashboard
fig_env.update_layout(
    xaxis_title="rpm_gen [rpm]",
    yaxis_title="T_gen [N·m]",
    legend_title="Referencia",
    
    # 🔥 Hover unificado (este es el cuadro único con todos los valores)
    hovermode="x unified",

    plot_bgcolor="white",
    margin=dict(l=60, r=20, t=40, b=40),

    # Opcional: hace que el cuadro flotante sea más legible
    hoverlabel=dict(
        bgcolor="white",
        font_size=12,
        font_color="black",
    ),
)


fig_env.update_xaxes(
    showgrid=False,
    zeroline=False,
)
fig_env.update_yaxes(
    showgrid=True,
    gridcolor="rgba(148,163,184,0.35)",
    zeroline=False,
)
st.plotly_chart(fig_env, use_container_width=True)

st.markdown("""
<div class="comment-box">
  <div class="comment-title">🔍 Interpretación técnica (envolvente T–rpm)</div>
  <p>
    Mantén la nube azul confinada en la caja verde; cualquier excursión sostenida en amarillo implica que el control permite
    sobre-torque o sobrevelocidad repetitiva y debes endurecer los límites de par.
  </p>
  <p>
    Si algunos puntos brincan a la zona roja, decide si aceptas esos transientes o si necesitas disparar frenado/shutdown antes:
    cruces frecuentes significan que la ley MPPT y el dimensionamiento de G no escalan a un generador comercial.
  </p>
</div>
""", unsafe_allow_html=True)


# Bloque – Sistema eléctrico y vibraciones
section_header("🔌 Sistema eléctrico y vibraciones")

# ==========================================================
# Corriente estimada vs velocidad de viento (con IEC)
# ==========================================================
# ==========================================================
# Corriente estimada vs velocidad de viento (con hover x-unified)
# ==========================================================
st.subheader("🔌 Corriente estimada vs velocidad de viento")
question_prompt("¿Qué corriente pico estás dispuesto a tolerar antes de redimensionar cables, breaker o control de par?")

# Selector de eje x
x_axis_mode_I = st.radio(
    "Eje x",
    ("v (m/s)", "rpm"),
    horizontal=True,
    key="x_axis_current",
    help="Elige si quieres ver la corriente contra velocidad de viento o contra rpm del rotor.",
)

if x_axis_mode_I == "v (m/s)":
    x_col_I = "v (m/s)"
    hover_x_I = "v = %{x:.1f} m/s"
else:
    x_col_I = "rpm_rotor"
    hover_x_I = "rpm = %{x:.0f} rpm"

# Ordenamos por el eje seleccionado para que la curva quede limpia
if x_axis_mode_I == "v (m/s)":
    dfI = df.sort_values(x_col_I).copy()
else:
    mask_reg2_I = (v_grid >= v_cut_in) & (v_grid <= v_rated)
    dfI = df.loc[mask_reg2_I].sort_values(x_col_I).copy()

figI = px.line(
    dfI,
    x=x_col_I,
    y="I_est (A)",
    markers=True,
)

# Estilo de traza + tooltip
figI.update_traces(
    line=dict(width=2.6),
    marker=dict(size=7),
    hovertemplate=(
        f"{hover_x_I}<br>"
        "I_est = %{y:.1f} A<extra></extra>"
    ),
    name="I_est (A)",
    showlegend=False,
)

# Layout general + hover unificado
figI.update_layout(
    xaxis_title="v (m/s)" if x_axis_mode_I == "v (m/s)" else "rpm",
    yaxis_title="Corriente trifásica estimada [A]",
    legend_title="",
    hovermode="x unified",          # 🔥 cuadro único al mover el cursor
    plot_bgcolor="white",
    margin=dict(l=50, r=20, t=40, b=40),
    hoverlabel=dict(
        bgcolor="white",
        font_size=12,
        font_color="black",
    ),
)

# Fondo con solo grilla horizontal suave
figI.update_xaxes(
    showgrid=False,
    zeroline=False,
)
figI.update_yaxes(
    showgrid=True,
    gridcolor="rgba(148,163,184,0.35)",
    zeroline=False,
)

if x_axis_mode_I == "v (m/s)":
    # ---- Líneas verticales: v_rated y v_cut-out ----
    figI.add_vline(
        x=float(v_rated),
        line_dash="dot",
        line_color="rgba(148,163,184,0.8)",
        annotation_text="v_rated",
        annotation_position="top",
    )

    figI.add_vline(
        x=float(v_cut_out),
        line_dash="dot",
        line_color="rgba(148,163,184,0.8)",
        annotation_text="v_cut-out",
        annotation_position="top",
    )
else:
    try:
        figI.add_vline(
            x=float(rpm_rotor_rated),
            line_dash="dot",
            line_color="rgba(148,163,184,0.8)",
            annotation_text="rpm_rotor_rated",
            annotation_position="top",
        )
    except Exception:
        pass

# ---- Línea horizontal: corriente nominal del generador ----
figI.add_hline(
    y=float(GDG_RATED_I),
    line_dash="dot",
    line_color="rgba(234,179,8,0.95)",
    annotation_text=f"I_nom gen ({GDG_RATED_I:.0f} A)",
    annotation_position="bottom right",
)

# ---- Franja IEC: zona de sobrecorriente (> I_nom) ----
I_max = float(dfI["I_est (A)"].max())
if I_max > GDG_RATED_I:
    figI.add_hrect(
        y0=float(GDG_RATED_I),
        y1=I_max,
        fillcolor="rgba(239,68,68,0.10)",
        line_width=0,
        layer="below",
        annotation_text="Zona sobre I_nom (IEC 61400-2 / protección térmica)",
        annotation_position="top left",
        annotation_font_size=11,
        annotation_font_color="rgba(107,114,128,1)",
    )
st.plotly_chart(figI, use_container_width=True)


st.markdown("""
<div class="comment-box">
  <div class="comment-title">🔍 Interpretación técnica (Corriente)</div>
  <p>
    Revisa dónde la curva azul cruza <strong>I_nom</strong>: si lo hace antes de <em>v_rated</em>, el MPPT está pidiendo
    más par del que soporta el estator y necesitas bajar TSR o ajustar el factor de potencia del inversor.
  </p>
  <p>
    La franja roja determina tu dimensionamiento eléctrico: corrientes planas dentro de esa banda implican que estás
    operando en derating permanente, así que o subes calibre y breaker o reduces el setpoint de potencia.
    Solo tolera picos cortos entre <em>v_rated</em> y <em>v_cut-out</em>; cualquier meseta constante exige redimensionar el tren eléctrico.
  </p>
</div>
""", unsafe_allow_html=True)

# ==========================================================
# Frecuencias 1P / 3P del rotor
# ==========================================================
st.subheader("📡 Frecuencias 1P / 3P del rotor")
question_prompt("¿Alguna de las frecuencias 1P o 3P coincide con modos estructurales que tengamos que evitar en el diseño final?")

# Selector de eje x
x_axis_mode_freq = st.radio(
    "Eje x",
    ("v (m/s)", "rpm"),
    horizontal=True,
    key="x_axis_freq",
    help="Elige si quieres ver las frecuencias contra velocidad de viento o contra rpm del rotor.",
)

if x_axis_mode_freq == "v (m/s)":
    x_col_freq = "v (m/s)"
    hover_x_freq = "v = %{x:.1f} m/s"
else:
    x_col_freq = "rpm_rotor"
    hover_x_freq = "rpm = %{x:.0f} rpm"

# Ordenamos por el eje seleccionado y preparamos info extra para el hover
df_freq = df.sort_values(x_col_freq).copy()
custom = np.stack(
    [df_freq["rpm_rotor"].values, df_freq["λ_efectiva"].values],
    axis=-1
)

figF = go.Figure()

series_freq = [
    ("f_1P (Hz)", "f_1P (Hz) – paso de pala"),
    ("f_3P (Hz)", "f_3P (Hz) – cargas 3P"),
]

for col, name in series_freq:
    figF.add_trace(
        go.Scatter(
            x=df_freq[x_col_freq],
            y=df_freq[col],
            mode="lines+markers",
            name=name,
            customdata=custom,
            line=dict(width=2.4),
            marker=dict(size=7),
            hovertemplate=(
                f"{hover_x_freq}<br>"
                "f = %{y:.3f} Hz<br>"
                "rpm_rotor = %{customdata[0]:.1f} rpm<br>"
                "λ_efectiva = %{customdata[1]:.2f}"
                "<extra></extra>"
            ),
        )
    )

if x_axis_mode_freq == "v (m/s)":
    # Líneas verticales: cut-in / rated / cut-out
    for x_val, label in [
        (v_cut_in,  "v_cut-in"),
        (v_rated,   "v_rated"),
        (v_cut_out, "v_cut-out"),
    ]:
        figF.add_vline(
            x=float(x_val),
            line_dash="dot",
            line_color="rgba(148,163,184,0.8)",
            annotation_text=label,
            annotation_position="top",
            annotation_font_size=11,
            annotation_font_color="rgba(107,114,128,1)",
        )
else:
    try:
        figF.add_vline(
            x=float(rpm_rotor_rated),
            line_dash="dot",
            line_color="rgba(148,163,184,0.8)",
            annotation_text="rpm_rotor_rated",
            annotation_position="top",
            annotation_font_size=11,
            annotation_font_color="rgba(107,114,128,1)",
        )
    except Exception:
        pass

# Banda típica de modos propios torre/fundación
f_min_modo = 0.2   # Hz  (ajusta según cálculo estructural real)
f_max_modo = 1.0   # Hz
figF.add_hrect(
    y0=f_min_modo,
    y1=f_max_modo,
    fillcolor="rgba(96,165,250,0.10)",
    line_width=0,
    layer="below",
    annotation_text="Banda típica modo 1 torre/fundación",
    annotation_position="top left",
    annotation_font_size=11,
    annotation_font_color="rgba(107,114,128,1)",
)

figF.update_layout(
    xaxis_title="v (m/s)" if x_axis_mode_freq == "v (m/s)" else "rpm",
    yaxis_title="Frecuencia [Hz]",
    legend_title="Componente",
    hovermode="x unified",          # 👈 cuadro único con las dos curvas
    plot_bgcolor="white",
    margin=dict(l=60, r=20, t=40, b=40),
    hoverlabel=dict(
        bgcolor="white",
        font_size=12,
        font_color="black",
    ),
)

# Fondo con solo grilla horizontal
figF.update_xaxes(showgrid=False, zeroline=False)
figF.update_yaxes(
    showgrid=True,
    gridcolor="rgba(148,163,184,0.35)",
    zeroline=False,
)
st.plotly_chart(figF, use_container_width=True)

st.markdown("""
<div class="comment-box">
  <div class="comment-title">🔍 Interpretación técnica (1P / 3P)</div>
  <p>
    Mantén 1P fuera del modo fundamental de torre: si cruza la banda azul en el rango de operación,
    incrementa la rigidez (torre más corta, fundación más rígida) o desplaza TSR para alejar la frecuencia.
  </p>
  <p>
    3P suele excitar aletas y mástil; si coincide con un modo, añade amortiguamiento o reconsidera número de palas/helicalidad.
    Esta lectura decide dónde ubicar soportes de struts y qué filtros de vibración necesitas antes de fabricar el piloto.
  </p>
</div>
""", unsafe_allow_html=True)

# ==========================================================
# Ruido (si aplica)
# ==========================================================

if use_noise:
    st.subheader("🔈 Ruido estimado vs velocidad de viento")
    question_prompt("¿Cumples con los límites acústicos del sitio a la distancia crítica o necesitas estrategias de reducción de U_tip?")

    # Selector de eje x
    x_axis_mode_noise = st.radio(
        "Eje x",
        ("v (m/s)", "rpm"),
        horizontal=True,
        key="x_axis_noise",
        help="Elige si quieres ver el ruido contra velocidad de viento o contra rpm del rotor.",
    )

    if x_axis_mode_noise == "v (m/s)":
        x_col_noise = "v (m/s)"
    else:
        x_col_noise = "rpm_rotor"

    df_noise = df.sort_values(x_col_noise).copy()

    # --- Curva principal ---
    figNoise = px.line(
        df_noise,
        x=x_col_noise,
        y=["Lw (dB)", "Lp_obs (dB)"],
        markers=True,
    )

    # --- Hover unificado y estilo principal ---
    figNoise.update_layout(
        xaxis_title="v (m/s)" if x_axis_mode_noise == "v (m/s)" else "rpm",
        yaxis_title="Nivel sonoro [dB]",
        legend_title="Magnitud",
        hovermode="x unified",          # 🔥 Tooltip unificado
        plot_bgcolor="white",
        margin=dict(l=50, r=20, t=40, b=40),
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_color="black",
        ),
    )

    # --- Estilo ejes (solo horizontal grid) ---
    figNoise.update_xaxes(showgrid=False, zeroline=False)
    figNoise.update_yaxes(
        showgrid=True,
        gridcolor="rgba(148,163,184,0.35)",
        zeroline=False,
    )

    if x_axis_mode_noise == "v (m/s)":
        # --- Líneas verticales: cut-in / rated / cut-out ---
        for x_val, label in [
            (v_cut_in,  "v_cut-in"),
            (v_rated,   "v_rated"),
            (v_cut_out, "v_cut-out"),
        ]:
            figNoise.add_vline(
                x=float(x_val),
                line_dash="dot",
                line_color="rgba(148,163,184,0.85)",
                annotation_text=label,
                annotation_position="top",
                annotation_font_size=11,
                annotation_font_color="rgba(107,114,128,1)",
            )
    else:
        try:
            figNoise.add_vline(
                x=float(rpm_rotor_rated),
                line_dash="dot",
                line_color="rgba(148,163,184,0.85)",
                annotation_text="rpm_rotor_rated",
                annotation_position="top",
                annotation_font_size=11,
                annotation_font_color="rgba(107,114,128,1)",
            )
        except Exception:
            pass

    # --- Línea horizontal: nivel objetivo en receptor ---
    Lp_obj = 45.0
    figNoise.add_hline(
        y=Lp_obj,
        line_dash="dot",
        line_color="rgba(34,197,94,0.9)",
        annotation_text=f"Nivel objetivo receptor ≈ {Lp_obj:.0f} dB",
        annotation_position="bottom right",
        annotation_font_size=11,
        annotation_font_color="rgba(107,114,128,1)",
    )

    # --- Franja donde se supera el nivel objetivo ---
    Lp_max = float(np.nanmax(df["Lp_obs (dB)"].values))
    if Lp_max > Lp_obj:
        figNoise.add_hrect(
            y0=Lp_obj,
            y1=Lp_max,
            fillcolor="rgba(239,68,68,0.10)",
            line_width=0,
            layer="below",
            annotation_text="Zona > nivel objetivo en receptor",
            annotation_position="top left",
            annotation_font_size=11,
            annotation_font_color="rgba(107,114,128,1)",
        )

    # --- Mostrar gráfico ---
    st.plotly_chart(figNoise, use_container_width=True)

    # --- Interpretación técnica ---
    st.markdown(f"""
    <div class="comment-box">
      <div class="comment-title">🔍 Interpretación técnica (ruido)</div>
      <p>
        Usa la curva <strong>Lp_obs</strong> para validar tu escenario de cumplimiento: si cruza los {Lp_obj:.0f} dB
        antes de <em>v_rated</em> necesitas bajar <strong>U_tip</strong> (reducir TSR o rpm) o incrementar la distancia al receptor.
      </p>
      <p>
        El gradiente de la curva te dice qué tan sensible es el ruido a variaciones de TSR: pendientes muy altas sugieren
        implementar control acústico (pitch o derating nocturno). Mantén la franja roja vacía para evitar trámites adicionales.
      </p>
    </div>
    """, unsafe_allow_html=True)



# Bloque – Recurso y energía anual
section_header("🌬️ Recurso y energía anual")

# =========================================================
# WEIBULL – SIEMPRE ACTIVO
# =========================================================

# Título ANTES de mostrar AEP y CF
st.subheader("🌬️ Distribución de viento vs curva de potencia")
question_prompt("¿Qué combinación de parámetros Weibull explica mejor tu sitio y justifica cambios en P_nom o v_rated?")

# Generación del vector Weibull
v_w_max = max(v_cut_out, v_max, 20.0)
v_w = np.linspace(0.01, v_w_max, 400)

# Potencia respetando cut-in / cut-out
P_curve_W = df["P_out (clip) kW"].values * 1000.0
P_curve_W[v_grid < v_cut_in] = 0.0
P_curve_W[v_grid > v_cut_out] = 0.0

P_interp_W = np.interp(
    v_w,
    v_grid,
    P_curve_W,
    left=0.0,
    right=0.0
)

# Weibull PDF
pdf_w = weibull_pdf(v_w, k_w, c_w)

# AEP y CF
AEP_kWh, P_mean_W = aep_from_weibull(v_w, P_interp_W, k_w, c_w)
CF = P_mean_W / (P_nom_kW * 1000.0)

colW1, colW2 = st.columns(2)
colW1.metric("AEP [kWh/año]", f"{AEP_kWh:,.0f}")
colW2.metric("Factor de Planta [%]", f"{CF*100:.1f}")

# Dataframe para gráfico técnico
df_weib = pd.DataFrame({
    "v (m/s)":      v_w,
    "f_W(v)":       pdf_w,
    "P_out (kW)":   P_interp_W / 1000.0,
    "P·f_W (kW·prob)": (P_interp_W / 1000.0) * pdf_w,
})

# Gráfico
figW = make_subplots(specs=[[{"secondary_y": True}]])

# Distribución Weibull
figW.add_trace(
    go.Scatter(
        x=df_weib["v (m/s)"],
        y=df_weib["f_W(v)"],
        mode="lines",
        name="Weibull f(v)",
        hovertemplate=(
            "v = %{x:.2f} m/s<br>"
            "f_W(v) = %{y:.3f} 1/(m/s)"
            "<extra></extra>"
        ),
    ),
    secondary_y=False,
)

# Curva de potencia
figW.add_trace(
    go.Scatter(
        x=df_weib["v (m/s)"],
        y=df_weib["P_out (kW)"],
        mode="lines",
        name="P_out (kW)",
        hovertemplate=(
            "v = %{x:.2f} m/s<br>"
            "P_out = %{y:.2f} kW"
            "<extra></extra>"
        ),
    ),
    secondary_y=True,
)

# Contribución al AEP
figW.add_trace(
    go.Scatter(
        x=df_weib["v (m/s)"],
        y=df_weib["P·f_W (kW·prob)"],
        mode="lines",
        name="P_out · f(v)",
        line=dict(dash="dot"),
        hovertemplate=(
            "v = %{x:.2f} m/s<br>"
            "P_out·f(v) = %{y:.3f} kW·prob"
            "<extra></extra>"
        ),
    ),
    secondary_y=True,
)

# Ejes
figW.update_xaxes(
    title_text="Velocidad de viento v [m/s]",
    showgrid=False,
    zeroline=False,
)

figW.update_yaxes(
    title_text="f_W(v) [1/(m/s)]",
    secondary_y=False,
    showgrid=True,
    gridcolor="rgba(148,163,184,0.35)",
    zeroline=False,
)

figW.update_yaxes(
    title_text="Potencia / Contribución [kW]",
    secondary_y=True,
    showgrid=False,
    zeroline=False,
)

# Estilo global y hover unificado
figW.update_layout(
    hovermode="x unified",          # 🔥 tooltip unificado en X
    plot_bgcolor="white",
    legend_title_text="",
    margin=dict(l=60, r=20, t=40, b=40),
    hoverlabel=dict(
        bgcolor="white",
        font_size=12,
        font_color="black",
    ),
)
st.plotly_chart(figW, use_container_width=True)


# Comentario técnico
st.markdown("""
<div class="comment-box">
  <div class="comment-title">🔍 Interpretación técnica (Weibull)</div>
  <p>
    Ajusta <strong>k</strong> y <strong>c</strong> hasta que el pico de <strong>f(v)</strong> coincida con el tramo plano de tu curva de potencia;
    si el máximo cae en la región clippeada estás perdiendo AEP por dimensionamiento de P_nom.
  </p>
  <p>
    Usa la curva punteada <strong>P·f(v)</strong> para elegir dónde invertir esfuerzos: si el área se concentra antes de 8 m/s,
    vale más bajar v_cut-in o reforzar Cp que subir potencia nominal. Maximiza el área bajo esa curva antes de cerrar especificaciones.
  </p>
</div>
""", unsafe_allow_html=True)

# KPIs globales para alertas / escenarios (solo región operativa IEC)
# =========================================================
op_mask = None
if (v_cut_in is not None) and (v_cut_out is not None):
    op_mask = (df["v (m/s)"] >= v_cut_in) & (df["v (m/s)"] <= v_cut_out)

df_alert = df.loc[op_mask] if op_mask is not None else df
if df_alert.empty:
    df_alert = df.copy()

sigma_root_sel = sigma_root_MPa
sigma_strut_sel = sigma_strut_MPa
if op_mask is not None and sigma_root_MPa.size == len(op_mask):
    sigma_root_sel = sigma_root_MPa[op_mask.values]
    sigma_strut_sel = sigma_strut_MPa[op_mask.values]

max_T_gen   = float(df_alert["T_gen (N·m)"].max())
max_T_rotor = float(df_alert["T_rotor (N·m)"].max())
max_I_est   = float(df_alert["I_est (A)"].max())
max_rpm_rot = float(df_alert["rpm_rotor"].max())
max_P_out   = float(df_alert["P_out (clip) kW"].max())
max_g_pala  = float(np.nanmax(df_alert["a_cen (g)"].values)) if "a_cen (g)" in df_alert.columns else np.nan
max_M_base  = float(np.nanmax(df_alert["M_base (kN·m)"].values)) if "M_base (kN·m)" in df_alert.columns else np.nan
max_sigma_root = float(np.nanmax(sigma_root_sel)) if sigma_root_sel.size else np.nan
max_sigma_strut = float(np.nanmax(sigma_strut_sel)) if sigma_strut_sel.size else np.nan

if T_gen_max > 0:
    margen_Tgen_iec = (T_gen_max - max_T_gen) / T_gen_max
else:
    margen_Tgen_iec = np.nan

try:
    if T_rotor_max_iec > 0:
        margen_Trot_iec = (T_rotor_max_iec - max_T_rotor) / T_rotor_max_iec
    else:
        margen_Trot_iec = np.nan
except NameError:
    margen_Trot_iec = np.nan

if GDG_RATED_I > 0:
    margen_I = (GDG_RATED_I - max_I_est) / GDG_RATED_I
else:
    margen_I = np.nan

if P_nom_kW > 0:
    margen_P = (P_nom_kW - max_P_out) / P_nom_kW
else:
    margen_P = np.nan

if g_max_pala_iec > 0:
    margen_g = (g_max_pala_iec - max_g_pala) / g_max_pala_iec
else:
    margen_g = np.nan

if M_base_max_iec > 0:
    margen_M = (M_base_max_iec - max_M_base) / M_base_max_iec
else:
    margen_M = np.nan

if allow_root_MPa > 0:
    margen_sigma_root = (allow_root_MPa - max_sigma_root) / allow_root_MPa
else:
    margen_sigma_root = np.nan

if allow_strut_MPa > 0:
    margen_sigma_strut = (allow_strut_MPa - max_sigma_strut) / allow_strut_MPa
else:
    margen_sigma_strut = np.nan


# Bloque – Electrónica y red
section_header("⚡ Electrónica y red – calidad de energía")

pf_margin = pf_setpoint - pf_min_grid
thd_margin = thd_req_pct - thd_cap_pct
lvrt_time_margin = lvrt_cap_time_ms - lvrt_req_time_ms
lvrt_voltage_margin = lvrt_req_voltage_pu - lvrt_cap_voltage_pu
inv_thermal_margin = (I_inv_thermal_A - max_I_inv) / I_inv_thermal_A if I_inv_thermal_A > 0 else np.nan
dc_util_max = float(np.nanmax(dc_util_series)) if dc_util_series.size else np.nan
dc_margin = 1.0 - dc_util_max if np.isfinite(dc_util_max) else np.nan

st.subheader("⚡ Electrónica y red – calidad de energía")
question_prompt("¿PF, THD y LVRT cumplen la normativa y queda margen térmico/energético en el bus DC?")

col_e1, col_e2, col_e3, col_e4, col_e5 = st.columns(5)
with col_e1:
    accent = "red" if pf_margin < 0 else "orange" if pf_margin < 0.02 else "green"
    kpi_card(
        "PF operativo",
        f"{pf_setpoint:.2f}",
        f"Margen vs red: {pf_margin:+.2f}",
        accent=accent
    )
with col_e2:
    accent = "red" if thd_margin < 0 else "orange" if thd_margin < 1 else "green"
    kpi_card(
        "THD estimada",
        f"{thd_cap_pct:.1f} %",
        f"Límite {thd_req_pct:.1f}% · margen {thd_margin:+.1f}%",
        accent=accent
    )
with col_e3:
    lvrt_ok = lvrt_time_margin >= 0 and lvrt_voltage_margin >= 0
    accent = "green" if lvrt_ok else "red"
    kpi_card(
        "LVRT capacidad",
        f"{lvrt_cap_time_ms:.0f} ms / {lvrt_cap_voltage_pu:.2f} pu",
        f"Δt {lvrt_time_margin:+.0f} ms · ΔV {lvrt_voltage_margin:+.2f} pu",
        accent=accent
    )
with col_e4:
    accent = "red" if inv_thermal_margin < 0 else "orange" if inv_thermal_margin < 0.1 else "green"
    kpi_card(
        "Margen térmico inversor",
        f"{(inv_thermal_margin*100):+.1f} %" if np.isfinite(inv_thermal_margin) else "N/A",
        f"I_max simulado ≈ {max_I_inv:.1f} A",
        accent=accent
    )
with col_e5:
    accent = "red" if dc_margin < 0 else "orange" if dc_margin < 0.05 else "green"
    kpi_card(
        "Duty bus DC",
        f"{dc_util_max*100:.1f} %" if np.isfinite(dc_util_max) else "N/A",
        f"Margen {dc_margin*100:+.1f} %",
        accent=accent
    )

# Selector de eje x
x_axis_mode_dc = st.radio(
    "Eje x",
    ("v (m/s)", "rpm"),
    horizontal=True,
    key="x_axis_dc",
    help="Elige si quieres ver la utilización del bus DC contra velocidad de viento o contra rpm del rotor.",
)

if x_axis_mode_dc == "v (m/s)":
    x_dc = v_grid
    hover_x_dc = "v = %{x:.2f} m/s"
    title_dc = "Utilización del bus DC vs viento"
    xaxis_title_dc = "v (m/s)"
else:
    x_dc = rpm_rotor
    hover_x_dc = "rpm = %{x:.0f} rpm"
    title_dc = "Utilización del bus DC vs rpm"
    xaxis_title_dc = "rpm"

fig_dc = go.Figure()
fig_dc.add_trace(go.Scatter(
    x=x_dc,
    y=dc_util_series * 100.0,
    mode="lines+markers",
    name="Duty DC [%]",
    hovertemplate=hover_x_dc + "<br>Duty = %{y:.1f}%<extra></extra>"
))
fig_dc.add_hline(
    y=100,
    line_dash="dash",
    line_color="#dc2626",
    annotation_text="Capacidad Vdc·Idc",
    annotation_position="top left"
)
if x_axis_mode_dc == "v (m/s)":
    for x_line, label in [
        (v_cut_in, "v_cut-in"),
        (v_rated, "v_rated"),
        (v_cut_out, "v_cut-out"),
    ]:
        fig_dc.add_vline(
            x=x_line,
            line_dash="dot",
            line_color="rgba(71,85,105,0.6)",
            annotation_text=label,
            annotation_position="top"
        )
else:
    try:
        fig_dc.add_vline(
            x=float(rpm_rotor_rated),
            line_dash="dot",
            line_color="rgba(71,85,105,0.6)",
            annotation_text="rpm_rotor_rated",
            annotation_position="top"
        )
    except Exception:
        pass
fig_dc.update_layout(
    title=title_dc,
    xaxis_title=xaxis_title_dc,
    yaxis_title="Duty DC [%] (|P| / Vdc·Idc)",
    template="plotly_white",
    hovermode="x unified"
)
st.plotly_chart(fig_dc, use_container_width=True)

st.markdown("""
<div class="comment-box">
  <div class="comment-title">🔌 Interpretación técnica (electrónica)</div>
  <p>
    Mantén el PF por encima del mínimo exigido para evitar penalizaciones de la distribuidora y verifica que la THD calculada
    por tus filtros se mantenga por debajo del límite normativo. Si alguno de los márgenes se vuelve negativo, deberás
    recalibrar la electrónica (filtros, control de reactive o hardware).
  </p>
  <p>
    El bloque LVRT compara tu capacidad frente al requisito del código de red, mientras que el duty del bus DC te indica qué
    tan cerca estás de saturar V_dc·I_dc_nom. Usa estos indicadores para dimensionar disipadores, capacitores y protecciones.
  </p>
</div>
""", unsafe_allow_html=True)

# =========================================================
# Módulo 3 – Alertas de diseño / operación (IEC-style)
# =========================================================
st.markdown('<div id="alertas"></div>', unsafe_allow_html=True)
st.subheader("🚨 Alertas de diseño / operación")
question_prompt("¿Cuál de los márgenes (par, corriente, potencia o rpm) se acerca más al cero y requiere acciones inmediatas?")
st.caption("Evaluación en región operativa IEC (v_cut_in–v_cut_out).")

flag_entries = []

def add_flag(message, suggestion=None):
    flag_entries.append({
        "message": message,
        "suggestion": suggestion,
    })

if T_gen_max > 0 and max_T_gen > 1.05 * T_gen_max:
    f_geom_tmax = (T_gen_max / max_T_gen) ** (1/3) if max_T_gen > 0 else 1.0
    suggestion = f"Reduce D/H por f ≈ {f_geom_tmax:.2f} o baja G para llevar el par por debajo de {T_gen_max:,.0f} N·m." if np.isfinite(f_geom_tmax) else None
    add_flag(
        f"⚠️ El par máximo en el generador ({max_T_gen:,.0f} N·m) excede el límite de diseño "
        f"configurado T_gen_max = {T_gen_max:,.0f} N·m (IEC / criterio estructural).",
        suggestion
    )

try:
    if T_rotor_max_iec > 0 and max_T_rotor > 1.02 * T_rotor_max_iec:
        f_rotor_lim = (T_rotor_max_iec / max_T_rotor) ** (1/3) if max_T_rotor > 0 else 1.0
        suggestion = (
            f"Escala D/H por f ≈ {f_rotor_lim:.2f} → D {D*f_rotor_lim:.2f} m, H {H*f_rotor_lim:.2f} m o aligera palas."
            if np.isfinite(f_rotor_lim) else None
        )
        add_flag(
            f"⚠️ El par máximo en el rotor ({max_T_rotor:,.0f} N·m) supera el límite IEC configurado "
            f"T_rotor_max_iec = {T_rotor_max_iec:,.0f} N·m. Requiere revisión estructural.",
            suggestion
        )
except NameError:
    pass

if GDG_RATED_I > 0 and max_I_est > 1.05 * GDG_RATED_I:
    P_target = P_nom_kW * (GDG_RATED_I / max_I_est) if P_nom_kW > 0 else None
    suggestion = (
        f"Reduce P_nom a ≈ {P_target:.1f} kW o incrementa la especificación de I_nom ≥ {max_I_est:.1f} A."
        if P_target and np.isfinite(P_target) else
        f"Sube el generador/inversor a I_nom ≥ {max_I_est:.1f} A."
    )
    add_flag(
        f"⚠️ La corriente máxima estimada ({max_I_est:,.1f} A) supera en más de un 5% "
        f"la corriente nominal de la máquina ({GDG_RATED_I:.1f} A). "
        "Revisa el dimensionamiento de cables, protecciones y el setpoint de potencia.",
        suggestion
    )

try:
    if rpm_rotor_max_iec > 0 and max_rpm_rot > 1.02 * rpm_rotor_max_iec:
        rpm_target = rpm_rotor_rated * (rpm_rotor_max_iec / max_rpm_rot) if rpm_rotor_rated > 0 else rpm_rotor_max_iec
        suggestion = None
        if np.isfinite(rpm_target):
            if control_mode == "MPPT (λ constante)":
                lambda_target = lam_opt_ctrl * (rpm_rotor_max_iec / max_rpm_rot) if lam_opt_ctrl else None
                if lambda_target and np.isfinite(lambda_target):
                    suggestion = (
                        f"Ajusta λ_control a ≤ {lambda_target:.2f} (actual {lam_opt_ctrl:.2f}) "
                        f"para sostener rpm ≤ {rpm_rotor_max_iec:.1f}."
                    )
                else:
                    suggestion = f"Reduce λ_control para sostener rpm ≤ {rpm_rotor_max_iec:.1f}."
            else:
                suggestion = f"Configura rpm_rotor_rated ≤ {rpm_target:.1f} rpm en el control de velocidad."
        add_flag(
            f"⚠️ La rpm máxima del rotor ({max_rpm_rot:.1f} rpm) excede el límite IEC configurado "
            f"rpm_rotor_max_iec = {rpm_rotor_max_iec:.1f} rpm. Ajusta el control de velocidad / shutdown.",
            suggestion
        )
except NameError:
    pass

if g_max_pala_iec > 0 and max_g_pala > 1.02 * g_max_pala_iec:
    rpm_target = rpm_rotor_rated * np.sqrt(g_max_pala_iec / max_g_pala) if rpm_rotor_rated > 0 else rpm_rotor_rated
    suggestion = (
        f"Baja rpm_rotor_rated a ≈ {rpm_target:.1f} rpm o reduce la masa efectiva de cada pala."
        if np.isfinite(rpm_target) else "Reduce rpm controlada o aligera las palas para disminuir g_radial."
    )
    add_flag(
        f"⚠️ La aceleración radial máxima ({max_g_pala:.1f} g) supera el límite configurado "
        f"({g_max_pala_iec:.1f} g). Revisa rpm_rated, masa de pala o control de velocidad.",
        suggestion
    )

if M_base_max_iec > 0 and max_M_base > 1.02 * M_base_max_iec:
    lever_target = lever_arm_pala * (M_base_max_iec / max_M_base)
    suggestion = (
        f"Reubica struts para un brazo efectivo ≈ {lever_target:.2f} m (actual {lever_arm_pala:.2f} m)."
        if np.isfinite(lever_target) else "Disminuye el brazo efectivo de struts o el momento aplicado."
    )
    add_flag(
        f"⚠️ El momento flector máximo en la unión pala–struts ({max_M_base:.0f} kN·m) excede el límite definido "
        f"({M_base_max_iec:.0f} kN·m). Ajusta el brazo efectivo, masa de pala o estrategia de control.",
        suggestion
    )

if allow_root_MPa > 0 and max_sigma_root > 1.02 * allow_root_MPa:
    W_target = section_modulus_root * (max_sigma_root / allow_root_MPa) if section_modulus_root > 0 else None
    suggestion = (
        f"Incrementa W en raíz a ≈ {W_target:.3f} m³ o reduce M_base con menor masa/TSR."
        if W_target and np.isfinite(W_target) else "Aumenta W en raíz o reduce M_base (masa/TSR)."
    )
    add_flag(
        f"⚠️ La tensión en raíz ({max_sigma_root:.1f} MPa) supera el admisible "
        f"({allow_root_MPa:.1f} MPa, FS={safety_target:.2f}).",
        suggestion
    )

if allow_strut_MPa > 0 and max_sigma_strut > 1.02 * allow_strut_MPa:
    area_target_cm2 = strut_area_cm2 * (max_sigma_strut / allow_strut_MPa) if strut_area_cm2 > 0 else None
    suggestion = (
        f"Aumenta área efectiva de strut a ≈ {area_target_cm2:.1f} cm² o reduce M_base."
        if area_target_cm2 and np.isfinite(area_target_cm2) else "Aumenta área de strut o reduce M_base."
    )
    add_flag(
        f"⚠️ La tensión en struts ({max_sigma_strut:.1f} MPa) supera el admisible "
        f"({allow_strut_MPa:.1f} MPa, FS={safety_target:.2f}).",
        suggestion
    )

if P_nom_kW > 0 and max_P_out > 1.02 * P_nom_kW:
    suggestion = (
        f"Sube el límite de inversor a ≥ {max_P_out:.1f} kW o recorta TSR para que P_out máx ≤ {P_nom_kW:.1f} kW."
        if np.isfinite(max_P_out) else None
    )
    add_flag(
        f"⚠️ La potencia máxima de salida ({max_P_out:.1f} kW) supera en más de un 2% "
        f"la potencia nominal del sistema ({P_nom_kW:.1f} kW). Revisa el clipping y los límites del inversor.",
        suggestion
    )

margin_cards_data = [
    {
        "label": "Margen T_gen vs IEC",
        "value": margen_Tgen_iec,
        "help": "(T_gen_max - T_max) / T_gen_max. Valores negativos indican sobrecarga."
    },
    {
        "label": "Margen T_rotor vs IEC",
        "value": margen_Trot_iec,
        "help": "(T_rotor_max_iec - T_rotor_max) / T_rotor_max_iec."
    },
    {
        "label": "Margen I_est vs I_nom",
        "value": margen_I,
        "help": "(I_nom - I_max_est) / I_nom."
    },
    {
        "label": "Margen P_out vs P_nom",
        "value": margen_P,
        "help": "(P_nom - P_max_out) / P_nom."
    },
    {
        "label": "Margen g_radial vs IEC",
        "value": margen_g,
        "help": "(g_max - g_max_medida) / g_max."
    },
    {
        "label": "Margen M_base vs IEC",
        "value": margen_M,
        "help": "(M_límite - M_max) / M_límite."
    },
    {
        "label": "Margen sigma_root vs FS",
        "value": margen_sigma_root,
        "help": "(sigma_allow - sigma_root_max) / sigma_allow."
    },
    {
        "label": "Margen sigma_strut vs FS",
        "value": margen_sigma_strut,
        "help": "(sigma_allow - sigma_strut_max) / sigma_allow."
    },
]

def margin_status(val: float) -> str:
    if not np.isfinite(val):
        return "neutral"
    if val < 0:
        return "danger"
    if val < 0.1:
        return "warn"
    return "ok"

cols_margins = st.columns(len(margin_cards_data))
for col, card in zip(cols_margins, margin_cards_data):
    val = card["value"]
    val_text = f"{val*100:.1f} %" if np.isfinite(val) else "N/A"
    help_txt = escape(card["help"])
    col.markdown(
        f"""
        <div class="margin-card margin-{margin_status(val)}">
            <div class="margin-card__title">
                <span>{escape(card["label"])}</span>
                <span class="margin-card__badge" title="{help_txt}">?</span>
            </div>
            <div class="margin-card__value">{val_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Tabla resumida de uso de límites
alert_rows = [
    {
        "Indicador": "T_gen (N·m)",
        "Máximo": max_T_gen,
        "Límite": T_gen_max if T_gen_max > 0 else np.nan,
        "Uso_%": (max_T_gen / T_gen_max * 100) if T_gen_max > 0 else np.nan,
    },
    {
        "Indicador": "T_rotor (N·m)",
        "Máximo": max_T_rotor,
        "Límite": T_rotor_max_iec if 'T_rotor_max_iec' in locals() and T_rotor_max_iec > 0 else np.nan,
        "Uso_%": (max_T_rotor / T_rotor_max_iec * 100) if 'T_rotor_max_iec' in locals() and T_rotor_max_iec > 0 else np.nan,
    },
    {
        "Indicador": "I_est (A)",
        "Máximo": max_I_est,
        "Límite": GDG_RATED_I if GDG_RATED_I > 0 else np.nan,
        "Uso_%": (max_I_est / GDG_RATED_I * 100) if GDG_RATED_I > 0 else np.nan,
    },
    {
        "Indicador": "P_out (kW)",
        "Máximo": max_P_out,
        "Límite": P_nom_kW if P_nom_kW > 0 else np.nan,
        "Uso_%": (max_P_out / P_nom_kW * 100) if P_nom_kW > 0 else np.nan,
    },
    {
        "Indicador": "a_cen (g)",
        "Máximo": max_g_pala,
        "Límite": g_max_pala_iec if g_max_pala_iec > 0 else np.nan,
        "Uso_%": (max_g_pala / g_max_pala_iec * 100) if g_max_pala_iec > 0 else np.nan,
    },
    {
        "Indicador": "M_base (kN·m)",
        "Máximo": max_M_base,
        "Límite": M_base_max_iec if M_base_max_iec > 0 else np.nan,
        "Uso_%": (max_M_base / M_base_max_iec * 100) if M_base_max_iec > 0 else np.nan,
    },
]

df_alerts = pd.DataFrame(alert_rows)
df_alerts["Margen_%"] = 100 - df_alerts["Uso_%"]

def classify_usage(u):
    if not np.isfinite(u):
        return "Sin dato"
    if u < 90:
        return "Seguro"
    if u < 100:
        return "Atento"
    return "Crítico"

df_alerts["Estado"] = df_alerts["Uso_%"].apply(classify_usage)

st.dataframe(
    df_alerts.style.format({
        "Máximo": "{:,.1f}",
        "Límite": "{:,.1f}",
        "Uso_%": "{:,.1f}",
        "Margen_%": "{:,.1f}",
    }).apply(
        lambda row: ["background-color: #dcfce7" if row.Estado == "Seguro"
                     else "background-color: #fef9c3" if row.Estado == "Atento"
                     else "background-color: #fee2e2" if row.Estado == "Crítico"
                     else ""]
            * len(row),
        axis=1
    ),
    use_container_width=True,
)

if flag_entries:
    section_header("Estado de diseño / operación", level=3)
    for entry in flag_entries:
        st.warning(entry["message"])
        if entry.get("suggestion"):
            st.caption(f"Valor sugerido: {entry['suggestion']}")
else:
    st.success("✅ Dentro de los límites configurados: sin alertas críticas para el rango de viento analizado.")

st.markdown("""
<div class="comment-box">
  <div class="comment-title">🔍 Interpretación técnica (alertas)</div>
  <p>
    La tabla resume el método: se toma el máximo simulado vs el límite configurado (IEC o ficha) y se calcula el % de uso.
    Usa estos márgenes como semáforo para liberar o no el diseño:
  </p>
  <ul>
    <li><strong>Margen T_gen</strong> &lt; 0 → baja G o sube la especificación del generador antes de fabricar.</li>
    <li><strong>Margen T_rotor</strong> próximo a 0 → refuerza palas/struts o reduce TSR en vientos altos.</li>
    <li><strong>Margen I_est</strong> negativo → revisa cables, breaker y electrónica o ajusta PF.</li>
    <li><strong>Margen P_out</strong> negativo → el inversor clippea demasiado; necesitas otro nivel de potencia.</li>
  </ul>
  <p>
    No cierres el diseño mientras algún margen sea negativo; prioriza corregirlos en ese orden porque definen esfuerzos,
    cumplimiento IEC y disipación térmica.
  </p>
</div>
""", unsafe_allow_html=True)

# =========================================================
# Resumen IEC 61400-2 – tabla operativa
# =========================================================
st.subheader("📋 Resumen IEC 61400-2 – operación por bin de viento")
question_prompt("¿Qué filas de la tabla IEC necesitas documentar para tu expediente de certificación antes de avanzar a la siguiente iteración?")

region_iec = df["v (m/s)"].apply(region_tag)

flag_t_rotor = np.full(len(df), "OK", dtype=object)
flag_t_gen = np.full(len(df), "OK", dtype=object)
flag_I = np.full(len(df), "OK", dtype=object)

if "T_rotor_max_iec" in locals() and T_rotor_max_iec > 0:
    flag_t_rotor = np.where(df["T_rotor (N·m)"] > T_rotor_max_iec, "⚠️", "OK")

if T_gen_max > 0:
    flag_t_gen = np.where(df["T_gen (N·m)"] > T_gen_max, "⚠️", "OK")
elif GDG_RATED_T_Nm > 0:
    flag_t_gen = np.where(df["T_gen (N·m)"] > GDG_RATED_T_Nm, "⚠️", "OK")

if GDG_RATED_I > 0:
    flag_I = np.where(df["I_est (A)"] > GDG_RATED_I, "⚠️", "OK")

df_iec = pd.DataFrame({
    "v (m/s)": df["v (m/s)"],
    "Región IEC": region_iec,
    "rpm_rotor": df["rpm_rotor"],
    "rpm_gen": df["rpm_gen"],
    "λ_efectiva": df["λ_efectiva"],
    "T_rotor (N·m)": df["T_rotor (N·m)"],
    "T_rotor estado": flag_t_rotor,
    "T_gen (N·m)": df["T_gen (N·m)"],
    "T_gen estado": flag_t_gen,
    "a_cen (g)": df["a_cen (g)"],
    "a_cen estado": np.where(df["a_cen (g)"] > g_max_pala_iec, "⚠️", "OK") if g_max_pala_iec > 0 else "OK",
    "M_base (kN·m)": df["M_base (kN·m)"],
    "M_base estado": np.where(df["M_base (kN·m)"] > M_base_max_iec, "⚠️", "OK") if M_base_max_iec > 0 else "OK",
    "M_por_strut (kN·m)": df["M_por_strut (kN·m)"],
    "P_mec eje (kW)": df["P_mec_gen (kW)"],
    "P_out (kW)": df["P_out (clip) kW"],
    "I_est (A)": df["I_est (A)"],
    "I estado": flag_I,
})

st.dataframe(df_iec, use_container_width=True)

st.download_button(
    "📥 Descargar tabla IEC 61400-2 (CSV)",
    data=df_iec.to_csv(index=False).encode("utf-8"),
    file_name="IEC61400_2_resumen_operativo.csv",
    mime="text/csv"
)


# 📄 Nota técnica (IEC 61400-2) – cierre
# =========================================================
st.markdown("""
---

### 📄 Nota técnica (IEC 61400-2)

Esta es la **tabla de operación del prototipo conforme a IEC 61400-2**:  
para cada *bin* de viento se documentan:

- **Región IEC** (pre cut-in / MPPT / potencia limitada / sobre cut-out),
- **rpm del rotor y del generador** y su **TSR (λ)**,
- **Torque** en rotor y eje lento/rápido con banderas de cumplimiento,
- **Potencia mecánica en el eje y potencia eléctrica disponible**,  
- **Corriente trifásica estimada** en el generador con alerta frente a I_nom o límite configurado.

Este registro es requerido para **validación estructural, evaluación energética (AEP), chequeo de límites de diseño** y para la preparación de documentación técnica del piloto en conformidad con IEC 61400-2 e IEC 61400-12-1.
""")


# Escenarios de diseño y comparador
# =========================================================
st.subheader("🧬 Escenarios de diseño y comparación")
question_prompt("¿Qué métrica (AEP, CF, márgenes IEC) quieres optimizar cuando compares dos configuraciones?")

# Inicializar contenedor de escenarios
if "escenarios" not in st.session_state:
    st.session_state["escenarios"] = []

colE1, colE2 = st.columns([2, 1])

# Nombre sugerido según cantidad de escenarios guardados
default_name = (
    f"Escenario {len(st.session_state['escenarios']) + 1}"
    if st.session_state["escenarios"] == []
    else "Escenario actual"
)

with colE1:
    nombre_esc = st.text_input(
        "Nombre del escenario actual",
        value=default_name,
        help="Ej: Helicoidal_60_G6.8, Sin_helix_G7.2, etc."
    )

with colE2:
    if st.button("💾 Guardar escenario actual"):
        escenario = {
            "nombre": nombre_esc,

            # --- Generador seleccionado (ficha GDG) ---
            "gen_key": gen_key,
            "gen_label": GEN["label"],
            "gen_T_nom_Nm": float(GDG_RATED_T_Nm),
            "gen_I_nom_A": float(GDG_RATED_I),
            "gen_rpm_nom": float(GDG_RATED_RPM),

            # Inputs clave (para poder recordar qué se probó)
            "inputs": {
                "D [m]": D,
                "H [m]": H,
                "N palas": N,
                "cuerda [m]": c,
                "TSR ref": tsr_ref,
                "G": G,
                "η_mec": eta_mec,
                "η_elec": eta_elec,
                "perfil": airfoil_name,
                "tipo_perfil": tipo_perfil,
                "t_rel [%]": t_rel,
                "helical": helical,
                "endplates": endplates,
                "trips": trips,
            "struts_perf": struts_perf,
            "struts/pala": struts_per_blade,
                "v_cut_in": v_cut_in,
                "v_rated": v_rated,
                "v_cut_out": v_cut_out,
                "k_Weibull": k_w,
                "c_Weibull [m/s]": c_w,
            },

            # Curvas principales (vs viento)
            "v": df["v (m/s)"].values.tolist(),
            "P_out_kW": df["P_out (clip) kW"].values.tolist(),
            "Cp_el": df["Cp_el_equiv"].values.tolist(),
            "T_rotor": df["T_rotor (N·m)"].values.tolist(),

            # Curvas eléctricas para el generador
            "T_gen": df["T_gen (N·m)"].values.tolist(),
            "I_est": df["I_est (A)"].values.tolist(),
            "a_cen_g": df["a_cen (g)"].values.tolist(),
            "M_strut_kNm": df["M_por_strut (kN·m)"].values.tolist(),
            "M_base_kNm": df["M_base (kN·m)"].values.tolist(),

            # KPIs energéticos
            "AEP_kWh": float(AEP_kWh),
            "CF": float(CF),
            "P_nom_kW": float(P_nom_kW),

            # KPIs de esfuerzo y márgenes (desde módulo de alertas)
            "max_T_gen": float(max_T_gen),
            "max_T_rotor": float(max_T_rotor),
            "max_I_est": float(max_I_est),
            "max_M_base": float(max_M_base),
            "max_a_cen_g": float(max_g_pala),
            "margen_Tgen_iec": float(margen_Tgen_iec),
            "margen_Trot_iec": float(margen_Trot_iec),
            "margen_I": float(margen_I),
            "margen_P": float(margen_P),
            "margen_g": float(margen_g),
            "margen_M": float(margen_M),
        }

        st.session_state["escenarios"].append(escenario)
        st.success(f"Escenario '{nombre_esc}' guardado en memoria de la sesión.")

# Mostrar listado resumen de escenarios guardados
if st.session_state["escenarios"]:
    section_header("Escenarios guardados en sesión", level=3)
    for i, esc in enumerate(st.session_state["escenarios"], start=1):
        margen_tgen = esc.get("margen_Tgen_iec", esc.get("margen_Tgen_nom", np.nan))
        st.markdown(
            f"- **{i}. {esc['nombre']}** "
            f"({esc['gen_label']}, G={esc['inputs']['G']:.2f}) – "
            f"P_nom = {esc['P_nom_kW']:.1f} kW, "
            f"AEP = {esc['AEP_kWh']:,.0f} kWh/año, "
            f"CF = {esc['CF']*100:.1f} %, "
            f"margen T_gen = {margen_tgen*100:.1f} %"
        )

# =========================================================
# Comparador A vs B
# =========================================================
if len(st.session_state["escenarios"]) < 2:
    st.info("Guarda al menos **dos escenarios** para habilitar el comparador A vs B.")
else:
    section_header("⚖️ Comparar dos escenarios")

    nombres = [e["nombre"] for e in st.session_state["escenarios"]]

    colC1, colC2 = st.columns(2)
    with colC1:
        escA_name = st.selectbox("Escenario A", nombres, key="escA_sel")
    with colC2:
        # Por defecto el segundo de la lista si existe
        default_idx_B = 1 if len(nombres) > 1 else 0
        escB_name = st.selectbox("Escenario B", nombres, index=default_idx_B, key="escB_sel")

    # Recuperar escenarios seleccionados
    escA = next(e for e in st.session_state["escenarios"] if e["nombre"] == escA_name)
    escB = next(e for e in st.session_state["escenarios"] if e["nombre"] == escB_name)

    # --- v_cut / v_rated / v_out de referencia (escenario A) ---
    v_cut_in_A  = float(escA["inputs"]["v_cut_in"])
    v_rated_A   = float(escA["inputs"]["v_rated"])
    v_cut_out_A = float(escA["inputs"]["v_cut_out"])

    # Helper para dar el mismo estilo a todos los gráficos del comparador
    def style_fig_comparador(fig, x_label, y_label,
                             v_cut_in, v_rated, v_cut_out,
                             region_label="Región potencia limitada"):
        fig.update_layout(
            xaxis_title=x_label,
            yaxis_title=y_label,
            legend_title="Escenario",
            hovermode="x unified",
            plot_bgcolor="white",
            margin=dict(l=60, r=20, t=40, b=40),
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_color="black",
            ),
        )
        fig.update_xaxes(showgrid=False, zeroline=False)
        fig.update_yaxes(
            showgrid=True,
            gridcolor="rgba(148,163,184,0.35)",
            zeroline=False,
        )

        # Líneas verticales
        for x_val, label in [
            (v_cut_in,  "v_cut-in"),
            (v_rated,   "v_rated"),
            (v_cut_out, "v_cut-out"),
        ]:
            fig.add_vline(
                x=float(x_val),
                line_dash="dot",
                line_color="rgba(148,163,184,0.8)",
                annotation_text=label,
                annotation_position="top",
                annotation_font_size=11,
                annotation_font_color="rgba(107,114,128,1)",
            )

        # Región sombreada entre v_rated y v_cut-out
        fig.add_vrect(
            x0=float(v_rated),
            x1=float(v_cut_out),
            fillcolor="rgba(148,163,184,0.10)",
            line_width=0,
            layer="below",
            annotation_text=region_label,
            annotation_position="top right",
            annotation_font_size=11,
            annotation_font_color="rgba(107,114,128,1)",
        )
        return fig

    # --- Resumen de generador para cada escenario ---
    colG1, colG2 = st.columns(2)
    with colG1:
        st.markdown(f"""
        **{escA_name}**  
        - Generador: **{escA['gen_label']}**  
        - P_nom gen ≈ {escA['P_nom_kW']:.1f} kW  
        - T_nom gen ≈ {escA['gen_T_nom_Nm']:,.0f} N·m  
        - I_nom gen ≈ {escA['gen_I_nom_A']:.0f} A  
        - rpm_nom gen ≈ {escA['gen_rpm_nom']:.0f} rpm  
        """)
    with colG2:
        st.markdown(f"""
        **{escB_name}**  
        - Generador: **{escB['gen_label']}**  
        - P_nom gen ≈ {escB['P_nom_kW']:.1f} kW  
        - T_nom gen ≈ {escB['gen_T_nom_Nm']:,.0f} N·m  
        - I_nom gen ≈ {escB['gen_I_nom_A']:.0f} A  
        - rpm_nom gen ≈ {escB['gen_rpm_nom']:.0f} rpm  
        """)

    # Grid común de velocidades para comparar (interpolamos)
    vA = np.array(escA["v"])
    vB = np.array(escB["v"])
    v_min_common = max(vA.min(), vB.min())
    v_max_common = min(vA.max(), vB.max())

    if v_max_common <= v_min_common:
        st.warning(
            "Los rangos de viento de los escenarios A y B no se solapan de forma útil. "
            "Intenta usar el mismo rango v_min / v_max en ambos antes de comparar."
        )
    else:
        v_common = np.linspace(v_min_common, v_max_common, 80)

        P_A = np.interp(v_common, vA, np.array(escA["P_out_kW"]))
        P_B = np.interp(v_common, vB, np.array(escB["P_out_kW"]))

        Cp_A = np.interp(v_common, vA, np.array(escA["Cp_el"]))
        Cp_B = np.interp(v_common, vB, np.array(escB["Cp_el"]))

        T_A = np.interp(v_common, vA, np.array(escA["T_rotor"]))
        T_B = np.interp(v_common, vB, np.array(escB["T_rotor"]))

        # =======================
        # KPIs comparativos energéticos
        # =======================
        colK1, colK2, colK3 = st.columns(3)
        colK1.metric(
            f"AEP {escA_name}",
            f"{escA['AEP_kWh']:,.0f} kWh/año",
            help="Escenario A"
        )
        colK2.metric(
            f"AEP {escB_name}",
            f"{escB['AEP_kWh']:,.0f} kWh/año",
            help="Escenario B"
        )
        delta_AEP = escB["AEP_kWh"] - escA["AEP_kWh"]
        colK3.metric(
            "ΔAEP (B - A)",
            f"{delta_AEP:,.0f} kWh/año",
        )

        colK4, colK5, colK6 = st.columns(3)
        colK4.metric(
            f"CF {escA_name}",
            f"{escA['CF']*100:.1f} %",
        )
        colK5.metric(
            f"CF {escB_name}",
            f"{escB['CF']*100:.1f} %",
        )
        colK6.metric(
            "ΔCF (B - A)",
            f"{(escB['CF']-escA['CF'])*100:.1f} pts",
        )

        # =======================
        # Márgenes IEC / esfuerzo del generador
        # =======================
        section_header("Márgenes de diseño (par, corriente, potencia)", level=3)

        margen_tgen_A = escA.get("margen_Tgen_iec", escA.get("margen_Tgen_nom", np.nan))
        margen_tgen_B = escB.get("margen_Tgen_iec", escB.get("margen_Tgen_nom", np.nan))

        colM1, colM2, colM3 = st.columns(3)
        colM1.metric(
            f"Margen T_gen {escA_name}",
            f"{margen_tgen_A*100:.1f} %",
            help="(T_gen_max - T_max)/T_gen_max – A"
        )
        colM2.metric(
            f"Margen T_gen {escB_name}",
            f"{margen_tgen_B*100:.1f} %",
            help="(T_gen_max - T_max)/T_gen_max – B"
        )
        colM3.metric(
            "Δ margen T_gen (B - A)",
            f"{(margen_tgen_B-margen_tgen_A)*100:.1f} pts",
        )

        colM4, colM5, colM6 = st.columns(3)
        colM4.metric(
            f"Margen I_est {escA_name}",
            f"{escA['margen_I']*100:.1f} %",
            help="(I_nom - I_max)/I_nom – A"
        )
        colM5.metric(
            f"Margen I_est {escB_name}",
            f"{escB['margen_I']*100:.1f} %",
            help="(I_nom - I_max)/I_nom – B"
        )
        colM6.metric(
            "Δ margen I_est (B - A)",
            f"{(escB['margen_I']-escA['margen_I'])*100:.1f} pts",
        )

        # =======================
        # Gráfico 1: P_out(kW)
        # =======================
        section_header("Curva de potencia eléctrica P_out(kW) vs viento", level=3)

        df_comp_P = pd.DataFrame({
            "v (m/s)": v_common,
            f"P_out {escA_name} [kW]": P_A,
            f"P_out {escB_name} [kW]": P_B,
        })

        fig_comp_P = px.line(
            df_comp_P,
            x="v (m/s)",
            y=[f"P_out {escA_name} [kW]", f"P_out {escB_name} [kW]"],
            markers=True,
        )
        fig_comp_P = style_fig_comparador(
            fig_comp_P,
            x_label="v (m/s)",
            y_label="P_out [kW]",
            v_cut_in=v_cut_in_A,
            v_rated=v_rated_A,
            v_cut_out=v_cut_out_A,
        )
        st.plotly_chart(fig_comp_P, use_container_width=True)

        # =======================
        # Gráfico 2: Cp_el_equiv
        # =======================
        section_header("Cp_el_equiv (eficiencia global viento → eléctrica)", level=3)

        df_comp_Cp = pd.DataFrame({
            "v (m/s)": v_common,
            f"Cp_el {escA_name}": Cp_A,
            f"Cp_el {escB_name}": Cp_B,
        })

        fig_comp_Cp = px.line(
            df_comp_Cp,
            x="v (m/s)",
            y=[f"Cp_el {escA_name}", f"Cp_el {escB_name}"],
            markers=True,
        )
        fig_comp_Cp = style_fig_comparador(
            fig_comp_Cp,
            x_label="v (m/s)",
            y_label="Cp_el_equiv",
            v_cut_in=v_cut_in_A,
            v_rated=v_rated_A,
            v_cut_out=v_cut_out_A,
        )
        st.plotly_chart(fig_comp_Cp, use_container_width=True)

        # =======================
        # Gráfico 3: Torque rotor
        # =======================
        section_header("Torque en rotor (N·m) – impacto estructural", level=3)

        df_comp_T = pd.DataFrame({
            "v (m/s)": v_common,
            f"T_rotor {escA_name} [N·m]": T_A,
            f"T_rotor {escB_name} [N·m]": T_B,
        })

        fig_comp_T = px.line(
            df_comp_T,
            x="v (m/s)",
            y=[f"T_rotor {escA_name} [N·m]", f"T_rotor {escB_name} [N·m]"],
            markers=True,
        )
        fig_comp_T = style_fig_comparador(
            fig_comp_T,
            x_label="v (m/s)",
            y_label="T_rotor [N·m]",
            v_cut_in=v_cut_in_A,
            v_rated=v_rated_A,
            v_cut_out=v_cut_out_A,
        )
        st.plotly_chart(fig_comp_T, use_container_width=True)

        # =======================
        # Gráfico 4: Torque generador
        # =======================
        section_header("Torque en generador (N·m) – esfuerzo en el eje rápido", level=3)

        Tgen_A = np.array(escA["T_gen"])
        Tgen_B = np.array(escB["T_gen"])

        df_comp_Tg = pd.DataFrame({
            "v (m/s)": v_common,
            f"T_gen {escA_name} [N·m]": np.interp(v_common, vA, Tgen_A),
            f"T_gen {escB_name} [N·m]": np.interp(v_common, vB, Tgen_B),
        })

        fig_comp_Tg = px.line(
            df_comp_Tg,
            x="v (m/s)",
            y=[f"T_gen {escA_name} [N·m]", f"T_gen {escB_name} [N·m]"],
            markers=True,
        )
        fig_comp_Tg = style_fig_comparador(
            fig_comp_Tg,
            x_label="v (m/s)",
            y_label="T_gen [N·m]",
            v_cut_in=v_cut_in_A,
            v_rated=v_rated_A,
            v_cut_out=v_cut_out_A,
        )
        st.plotly_chart(fig_comp_Tg, use_container_width=True)

        # =======================
        # Gráfico 5: Corriente estimada
        # =======================
        section_header("Corriente estimada en generador (A)", level=3)

        I_A = np.array(escA["I_est"])
        I_B = np.array(escB["I_est"])

        df_comp_I = pd.DataFrame({
            "v (m/s)": v_common,
            f"I_est {escA_name} [A]": np.interp(v_common, vA, I_A),
            f"I_est {escB_name} [A]": np.interp(v_common, vB, I_B),
        })

        fig_comp_I = px.line(
            df_comp_I,
            x="v (m/s)",
            y=[f"I_est {escA_name} [A]", f"I_est {escB_name} [A]"],
            markers=True,
        )
        fig_comp_I = style_fig_comparador(
            fig_comp_I,
            x_label="v (m/s)",
            y_label="I_est [A]",
            v_cut_in=v_cut_in_A,
            v_rated=v_rated_A,
            v_cut_out=v_cut_out_A,
            region_label="Región potencia limitada / sobrecorriente",
        )
        st.plotly_chart(fig_comp_I, use_container_width=True)

        st.markdown(f"""
        <div class="comment-box">
          <div class="comment-title">🔍 Interpretación técnica (comparador A vs B)</div>
          <p>
            Usa el comparador para decidir qué escenario escala mejor:
          </p>
          <ul>
            <li>Si <strong>{escB_name}</strong> gana AEP pero empuja <strong>T_rotor</strong> o <strong>T_gen</strong> sobre IEC, solo vale la pena si aceptas reforzar la estructura.</li>
            <li>Compara <strong>Cp_el_equiv</strong>: si la mejora viene solo de subir P_nom sin subir Cp, terminarás clippeando antes.</li>
            <li>Revisa <strong>I_est</strong>: un escenario que llega menos tiempo a I_nom libera presupuesto en cables e inversor.</li>
            <li>La curva de <strong>P_out</strong> te dice en qué bins ganas energía; si la diferencia está solo en vientos raros, prioriza el escenario con menores cargas.</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)



# =========================================================

# Recomendaciones dinámicas
# =========================================================

# 1) Construimos la lista 'bullets' en función de los resultados
bullets = []

# Arranque / cut-in
if v_cut_in > 3.5:
    bullets.append(
        f"Arranque: v_cut-in = {v_cut_in:.1f} m/s es algo alta; evalúa bajar a 3–3.5 m/s "
        "con más solidez o apoyo Savonius/kick para mejorar energía en vientos bajos."
    )
else:
    bullets.append(
        f"Arranque: v_cut-in = {v_cut_in:.1f} m/s es adecuada para capturar energía en vientos bajos "
        "sin penalizar demasiado el par de arranque."
    )

# Solidez / Cp
if sig_conv < 0.22:
    bullets.append(
        f"Solidez: σ_conv ≈ {sig_conv:.2f} indica un rotor liviano; podrías subir ligeramente c o N "
        "para ganar Cp en rangos medios de viento."
    )
elif sig_conv > 0.30:
    bullets.append(
        f"Solidez: σ_conv ≈ {sig_conv:.2f} es alta; revisa cargas inerciales y par en arranque, "
        "porque el rotor puede volverse pesado para rpm bajas."
    )
else:
    bullets.append(
        f"Solidez: σ_conv ≈ {sig_conv:.2f} está en el rango 0.22–0.30, razonable para un VAWT de potencia."
    )

# Eficiencias
if eta_mec < 0.95:
    bullets.append(
        f"Eficiencia mecánica: η_mec ≈ {eta_mec:.3f}; conviene revisar pérdidas en rodamientos y caja "
        "porque podrías estar perdiendo varios puntos de rendimiento antes del generador."
    )
else:
    bullets.append(
        f"Eficiencia mecánica: η_mec ≈ {eta_mec:.3f} es buena para un tren de potencia con caja de engranajes."
    )

if eta_elec < 0.97:
    bullets.append(
        f"Eficiencia electrónica: η_elec ≈ {eta_elec:.3f}; considera equipos más eficientes o mejor ajuste de PF "
        "si el proyecto es muy sensible al LCOE."
    )

# Factor de planta / AEP
if CF < 0.20:
    bullets.append(
        f"Factor de planta: FP ≈ {CF*100:.1f}% es algo bajo; revisa ajuste entre Weibull del sitio, "
        "v_rated y potencia nominal para mejorar utilización anual."
    )
else:
    bullets.append(
        f"Factor de planta: FP ≈ {CF*100:.1f}% es razonable; el dimensionamiento entre viento del sitio y "
        "potencia nominal parece coherente."
    )
# Curvas respecto al viento / TSR / rpm (fundamento IEC)
bullets.append(
    "Curvas respecto al viento: una turbina no se diseña con rpm como entrada; "
    "las rpm son un resultado directo del TSR y de la velocidad del viento. "
    "Por norma internacional (IEC 61400-12-1 e IEC 61400-2), la potencia, el par, el Cp, "
    "las pérdidas y las rpm deben expresarse en función del viento, porque es la variable "
    "física primaria que gobierna el comportamiento del aerogenerador y la única referencia "
    "universal para comparar turbinas, validar rendimiento y certificar la curva de potencia."
)

# Si por alguna razón no se generó nada:
if not bullets:
    bullets.append(
        "Configuración del piloto consistente; se recomienda validar en sitio con mediciones "
        "de viento y curvas del generador antes de congelar diseño."
    )

# 2) Caja completa: recomendaciones + fórmulas
st.markdown("""
<div class="rec-wrapper">
  <div class="rec-header">
    <div class="rec-header-icon">🛠️</div>
    <div>
      <div class="rec-header-chip">Salida automática del modelo</div>
      <div class="rec-header-text-main">Recomendaciones para el piloto</div>
    </div>
  </div>
""", unsafe_allow_html=True)

# Recomendaciones (usamos la lista 'bullets')
for b in bullets:
    st.markdown(f"<div class='rec-item'>{b}</div>", unsafe_allow_html=True)

# === Caja de fórmulas en dos columnas (versión Streamlit) ===

# Cabecera con el mismo look de caja
st.markdown("""
<div class="formula-box">
    <div class="formula-title">🧮 Fórmulas clave</div>
</div>
""", unsafe_allow_html=True)

# Dos columnas reales de Streamlit
col1, col2 = st.columns(2)

# ----------- COLUMNA IZQUIERDA ----------
with col1:
    st.latex(r"\bullet\ \text{TSR: }\lambda = \dfrac{\omega R}{v} = \dfrac{U_{\text{tip}}}{v}")
    st.latex(r"\bullet\ \text{rpm (rotor): }\text{rpm} = \dfrac{30}{\pi R}\,\lambda\,v")
    st.latex(r"\bullet\ \text{Potencia aerodinámica: }P_a = \dfrac{1}{2}\rho A v^{3} C_p(\lambda)")
    st.latex(r"\bullet\ \text{Par: }T = \dfrac{P}{\omega}")

# ----------- COLUMNA DERECHA ----------
with col2:
    st.latex(r"\bullet\ \text{Frecuencia eléctrica: }f_e = \dfrac{P_{\text{polos}}}{2}\,\dfrac{\text{rpm}_{gen}}{60}")
    st.latex(r"\bullet\ \text{Corriente trifásica (aprox.): }I \approx \dfrac{P}{\sqrt{3}\,V_{LL}\,PF}")
    st.latex(r"\bullet\ \text{Reynolds pala: }Re \approx \dfrac{\rho\,U_{\text{tip}}\,c}{\mu}")



# =========================================================
# Descargar reporte técnico (PDF)
# =========================================================
st.subheader("📄 Descargar reporte técnico (PDF)")

kpi_summary = (
    f"Geometría evaluada: D = {D:.1f} m, H = {H:.1f} m, N = {N} palas. "
    f"TSR ref λ = {tsr_ref:.2f}, solidez σ_int = {sig_int:.2f} (σ_conv ≈ {sig_conv:.2f}). "
    f"Potencia nominal configurada: {P_nom_kW:.1f} kW; "
    f"relación de transmisión G = {G:.2f}; "
    f"η_mec ≈ {eta_mec:.3f}, η_elec ≈ {eta_elec:.3f}."
)

# --- Elegir qué figura de potencia mandar al PDF (según modo seleccionado) ---
if dominio_pot == "Potencias vs viento (recomendada)":
    fig_pot = figP
else:
    fig_pot = figG

# --- Selección priorizada de figuras para el reporte (ordenada) ---
figs_report = [
    ("rpm rotor / generador vs velocidad de viento", fig_r),
    ("Curva de potencia (según vista seleccionada)", fig_pot),
    ("Par en rotor / generador", figT),
    ("Momento flector en unión pala–struts", fig_mbase),
    ("Corriente estimada vs velocidad de viento", figI),
    ("Cp equivalente por etapa", fig_cp_eq),
    ("Pérdidas por etapa", fig_loss),
    ("Frecuencias 1P / 3P del rotor", figF),
    ("Curva Cp(λ) – promedio y componentes", fig_cp),
    ("🌬️ Distribución de viento vs curva de potencia", figW),
]


# -------------------------------------------------------
# Construcción diccionario de figuras
# -------------------------------------------------------
if use_noise:
    figs_report.append(("Ruido estimado vs velocidad de viento", figNoise))

# -------------------------------------------------------
# Botón para generar PDF
# -------------------------------------------------------
if st.button("Generar reporte PDF"):
    pdf_bytes = build_pdf_report(df_view, figs_report, kpi_summary)

    st.download_button(
        label="📥 Descargar reporte técnico (PDF)",
        data=pdf_bytes,
        file_name="reporte_tecnico_VAWT.pdf",
        mime="application/pdf",
        key="descargar_pdf_tecnico_vawt"   # 🔑 clave única
    )

# NUEVO: Calibración modelo vs datos piloto (SCADA)
# =========================================================
st.subheader("🧪 Calibración modelo vs datos piloto (SCADA)")
question_prompt("¿Qué métrica de ajuste (Bias, RMSE o R²) debe mejorar para aceptar que el modelo representa al piloto en campo?")

df_scada = st.session_state.get("df_scada_raw", None)
scada_map = st.session_state.get("scada_map", None)

if df_scada is None or scada_map is None:
    st.info(
        "Sube un CSV en el panel lateral (expander 'Datos piloto (SCADA)') "
        "para comparar el modelo con las mediciones del piloto."
    )
else:
    # Limpieza básica
    df_sc = df_scada.copy()

    v_col = scada_map["v"]
    P_col = scada_map["P"]

    # El modelo está en df con 'v (m/s)' y 'P_out (clip) kW'
    v_meas = df_sc[v_col].astype(float).values
    P_meas = df_sc[P_col].astype(float).values

    # Interpolamos la potencia modelo en las velocidades medidas
    P_model = np.interp(
        v_meas,
        df["v (m/s)"].values,
        df["P_out (clip) kW"].values,
        left=0.0,
        right=0.0,
    )

    # Cálculo de métricas de ajuste
    mask_valid = ~np.isnan(P_meas) & ~np.isnan(P_model)
    if mask_valid.sum() > 3:
        err = P_model[mask_valid] - P_meas[mask_valid]
        bias = np.mean(err)
        rmse = np.sqrt(np.mean(err**2))
        ss_res = np.sum((P_meas[mask_valid] - P_model[mask_valid])**2)
        ss_tot = np.sum((P_meas[mask_valid] - np.mean(P_meas[mask_valid]))**2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    else:
        bias = rmse = r2 = np.nan

    c1, c2, c3 = st.columns(3)
    c1.metric("Bias modelo - medida [kW]", f"{bias:,.2f}")
    c2.metric("RMSE [kW]", f"{rmse:,.2f}")
    c3.metric("R² ajuste", f"{r2:,.2f}")

    st.caption(
        "Bias > 0 indica que el modelo sobreestima la potencia respecto al piloto; "
        "Bias < 0 indica subestimación. RMSE resume el error típico por punto, y R² "
        "qué tan bien el modelo explica la variabilidad de las mediciones."
    )

    # ---------------- Gráfico 1: v vs Potencia ----------------
    section_header("Potencia eléctrica: modelo vs piloto", level=3)

    df_plotP = pd.DataFrame({
        "v (m/s)": v_meas,
        "P_meas (kW)": P_meas,
        "P_model (kW)": P_model,
    })

    fig_scada_P = px.scatter(
        df_plotP,
        x="v (m/s)",
        y="P_meas (kW)",
        opacity=0.7,
        labels={"P_meas (kW)": "Potencia medida [kW]"},
        title="Potencia medida vs modelo",
    )
    # Agregamos la curva modelo suavizada vs viento
    fig_scada_P.add_trace(
        go.Scatter(
            x=df["v (m/s)"],
            y=df["P_out (clip) kW"],
            mode="lines",
            name="P_model curva",
        )
    )
    fig_scada_P.update_layout(
        legend_title="Serie",
        xaxis_title="v (m/s)",
        yaxis_title="Potencia [kW]",
    )
    st.plotly_chart(fig_scada_P, use_container_width=True)

    # ---------------- Gráfico 2: rpm rotor ----------------
    rpm_rotor_col = scada_map.get("rpm_rotor", None)
    if rpm_rotor_col is not None:
        section_header("rpm rotor: modelo vs piloto", level=3)

        rpm_meas = df_sc[rpm_rotor_col].astype(float).values
        rpm_model = np.interp(
            v_meas,
            df["v (m/s)"].values,
            df["rpm_rotor"].values,
            left=0.0,
            right=0.0,
        )
        df_plotR = pd.DataFrame({
            "v (m/s)": v_meas,
            "rpm_meas": rpm_meas,
            "rpm_model": rpm_model,
        })

        fig_scada_R = px.scatter(
            df_plotR,
            x="v (m/s)",
            y="rpm_meas",
            opacity=0.7,
            labels={"rpm_meas": "rpm rotor medida"},
            title="rpm rotor medida vs modelo",
        )
        fig_scada_R.add_trace(
            go.Scatter(
                x=df["v (m/s)"],
                y=df["rpm_rotor"],
                mode="lines",
                name="rpm_rotor modelo",
            )
        )
        fig_scada_R.update_layout(
            xaxis_title="v (m/s)",
            yaxis_title="rpm rotor",
        )
        st.plotly_chart(fig_scada_R, use_container_width=True)

    # ---------------- Gráfico 3: corriente ----------------
    I_col = scada_map.get("I", None)
    if I_col is not None:
        section_header("Corriente: modelo vs piloto", level=3)

        I_meas = df_sc[I_col].astype(float).values
        I_model = np.interp(
            v_meas,
            df["v (m/s)"].values,
            df["I_est (A)"].values,
            left=0.0,
            right=0.0,
        )
        df_plotI = pd.DataFrame({
            "v (m/s)": v_meas,
            "I_meas (A)": I_meas,
            "I_model (A)": I_model,
        })

        fig_scada_I = px.scatter(
            df_plotI,
            x="v (m/s)",
            y="I_meas (A)",
            opacity=0.7,
            labels={"I_meas (A)": "Corriente medida [A]"},
            title="Corriente medida vs modelo",
        )
        fig_scada_I.add_trace(
            go.Scatter(
                x=df["v (m/s)"],
                y=df["I_est (A)"],
                mode="lines",
                name="I_model curva",
            )
        )
        fig_scada_I.update_layout(
            xaxis_title="v (m/s)",
            yaxis_title="Corriente [A]",
        )
        st.plotly_chart(fig_scada_I, use_container_width=True)

    st.markdown("""
    <div class="comment-box">
      <div class="comment-title">🔍 Interpretación técnica (calibración)</div>
      <p>
        Si el <strong>Bias</strong> es positivo estás sobrestimando potencia: ajusta Cp(λ) o pérdidas hasta que quede dentro de ±5&nbsp;%.
        Un <strong>RMSE</strong> alto en vientos medios indica que el MPPT o la curva del generador no replican al piloto; vuelve a calibrar G o la curva P–rpm.
      </p>
      <p>
        Exige <strong>R²</strong> &gt; 0.9 antes de liberar el modelo; valores menores significan que falta capturar algún mecanismo (clipping, turbulencia o histeresis).
        Usa los gráficos por variable para ver en qué bin se separa y corrige ese módulo antes de la siguiente campaña.
      </p>
    </div>
    """, unsafe_allow_html=True)

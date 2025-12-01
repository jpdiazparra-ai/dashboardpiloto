import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from math import pi, sqrt
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


# ====== ESTILO GLOBAL (comentarios + KPIs) ======


st.markdown("""
<style>

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

</style>
""", unsafe_allow_html=True)


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

    df_short = df_view.head(15)

    # Ajustar ancho de columnas
    page_width, _ = A4
    table_width = page_width - 2 * cm
    n_cols = len(df_short.columns)
    col_widths = [table_width / max(n_cols, 1)] * n_cols

    data = [list(df_short.columns)] + df_short.values.tolist()
    table = Table(data, colWidths=col_widths, repeatRows=1)

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
            "Muestra cómo crecen las rpm del rotor y del generador según la ley de control por regiones. "
            "Permite verificar TSR casi constante en región 2 y que el generador llega a su zona nominal sin sobrepasarla.",
        "Potencias: aero, mecánica, generador y salida":
            "Compara la potencia aerodinámica, mecánica, la curva nominal del generador y la potencia eléctrica con clipping. "
            "Ayuda a ver en qué rango dominan las pérdidas mecánicas, el generador o el límite nominal.",
        "Cp equivalente por etapa":
            "Traduce cada etapa a un Cp equivalente (aero, eje, eléctrico) para visualizar dónde se pierden más eficiencias "
            "entre rotor, tren mecánico, generador y electrónica.",
        "Pérdidas por etapa":
            "Área apilada que muestra las pérdidas mecánicas, del generador, de la electrónica y por clipping. "
            "Sirve para priorizar dónde actuar en rediseño o control.",
        "Par en rotor / generador":
            "Muestra el par en rotor y generador según el viento. Es clave para dimensionar ejes, rodamientos, caja y límites "
            "de T_gen_max, evitando sobrepar crítico.",
        "Corriente estimada vs velocidad de viento":
            "Permite dimensionar cables, protecciones e inversores y comprobar que no se superan las corrientes nominales.",
        "Frecuencias 1P / 3P del rotor":
            "Frecuencias asociadas al paso de palas y cargas periódicas principales, para comparar con modos propios de torre "
            "y cimentación y evitar resonancias.",
        "Curva Cp(λ) – promedio y componentes":
            "Curva Cp(λ) con componentes upwind/downwind. La comparación entre λ_opt y el TSR objetivo guía ajustes de "
            "geometría y control para operar cerca del máximo Cp.",
        "Ruido estimado vs velocidad de viento":
            "Nivel de potencia sonora y de presión en función de la velocidad de punta y la distancia de observador, útil "
            "para verificar cumplimiento de criterios acústicos."
    }

    # Gráficos
    for title, fig in figs_dict.items():
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
GDG_POWER_TABLE_80 = pd.DataFrame({
    "rpm":  [  0,  24,  48,  72,  96, 120, 144, 168, 192, 216, 240, 264],
    "P_kW": [  0,   2,   3,   7,  12,  19,  28,  38,  50,  64,  80,  97],
})

GDG_VOLT_TABLE_80 = pd.DataFrame({
    "rpm":  [  0,  24,  48,  72,  96, 120, 144, 168, 192, 216, 240, 264],
    "V_LL": [  0,  40,  80, 120, 160, 200, 240, 280, 320, 360, 400, 440],
})

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
GDG_POWER_TABLE_10 = pd.DataFrame({
    "rpm":  [0, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77],
    "P_kW": [0, 0.2, 0.4, 0.9, 1.5, 2.4, 3.5, 4.7, 6.2, 8.0, 10.0, 12.1],
})

GDG_VOLT_TABLE_10 = pd.DataFrame({
    "rpm":  [0, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77],
    "V_LL": [0, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440],
})

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
st.title("🧪 VAWT kW + Generador (aero → mecánico → eléctrico)")

with st.sidebar:

    # Geometría
    with st.expander("Geometría", expanded=False):
        D = st.number_input("Diámetro D [m]",  min_value=2.0, value=14.0, step=0.5)
        H = st.number_input("Altura H [m]",    min_value=2.0, value=14.0, step=0.5)
        N = st.number_input("Nº de palas N",   min_value=2,   value=3, step=1)
        c = st.number_input("Cuerda c [m]",    min_value=0.1, value=0.80, step=0.05)
    
    with st.expander("Operación / Control", expanded=False):

        # TSR óptimo para control MPPT
        lam_opt_ctrl = st.number_input(
            "TSR objetivo λ (control)",
            min_value=1.6,
            value=2.47,   # aquí defines tu λ_opt de operación
            step=0.05
        )
        tsr = lam_opt_ctrl  # este TSR se usa en las ecuaciones aero

        rho = st.number_input("Densidad aire ρ [kg/m³]", min_value=1.0, value=1.225, step=0.025)
        mu  = st.number_input(
            "Viscosidad dinámica μ [Pa·s]",
            min_value=1.0e-5, max_value=3.0e-5,
            value=1.8e-5, step=0.1e-5, format="%.6f"
        )
        v_cut_in  = st.number_input("v_cut-in [m/s]",  min_value=0.5, value=3.0, step=0.5)
        v_rated   = st.number_input("v_rated [m/s]",   min_value=v_cut_in + 0.5, value=12.0, step=0.5)
        v_cut_out = st.number_input("v_cut-out [m/s]", min_value=v_rated + 0.5, value=20.0, step=0.5)


    # Tweaks aerodinámicos
    with st.expander("Tweaks aerodinámicos", expanded=False):
        helical     = st.checkbox("Helicoidal 60–90°", True)
        endplates   = st.checkbox("End-plates / winglets", False)
        trips       = st.checkbox("Trips / micro-tabs", False)
        struts_perf = st.checkbox("Struts perfilados (0012)", False)

    # Perfil de pala / masa
    with st.expander("Perfil de pala / masa", expanded=False):
        airfoil_name = st.text_input("Perfil (ej: NACA 0018)", "NACA 0022")
        tipo_perfil  = st.selectbox("Tipo de perfil", ["Simétrico", "Asimétrico"])
        is_symmetric = (tipo_perfil == "Simétrico")
        t_rel = st.number_input("Espesor relativo e/c [%]", min_value=8.0, max_value=40.0, value=22.0, step=1.0)
        pitch_deg = st.number_input("Ángulo de calaje (pitch) [°]", min_value=-10.0, max_value=10.0, value=0.0, step=0.5)
        m_blade = st.number_input("Masa por pala [kg]", min_value=10.0, value=120.0, step=10.0)
        helix_angle_deg = st.number_input("Ángulo helicoidal pala [°]", min_value=0.0, max_value=90.0, value=60.0, step=5.0)
        use_H_for_span = st.checkbox("Usar H para longitud de pala", True)
        

        if use_H_for_span:
            helix_rad = np.deg2rad(helix_angle_deg)
            blade_span = H / max(np.cos(helix_rad), 1e-3)
            st.caption(f"Longitud de pala estimada ≈ {blade_span:.1f} m (helix {helix_angle_deg:.0f}°)")
        else:
            blade_span = st.number_input("Longitud de pala [m]", min_value=H*0.5, value=float(H), step=0.5)

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

        # rpm sugerida por aerodinámica
        rpm_sugerida = float(rpm_from_tsr(v_rated, D, tsr))
        st.caption(
            f"rpm rotor rated sugerida por diseño aerodinámico (TSR y v_rated): "
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


    # --- IEC 61400-2 – límites de diseño (expander separado, NO anidado) ---
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


        # Weibull (opcional)
    with st.expander("Weibull (opcional)", expanded=False):
        use_weibull = st.checkbox("Calcular AEP/FP con Weibull", False)
        k_w = st.number_input("k (forma)",  min_value=1.0, value=2.0, step=0.1)
        c_w = st.number_input("c (escala) [m/s]", min_value=2.0, value=7.5, step=0.5)

    # =========================================================
    # NUEVO: Datos piloto (SCADA) para calibración
    # =========================================================
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
# (lo igualamos al óptimo teórico para que λ_opt_ctrl = λ_opt_teo)
lambda_mppt = lambda_opt_teo


# Grid de vientos
v_grid = np.arange(v_min, v_max + 1e-9, 0.5 if v_max - v_min > 1 else 0.1)

# Ley de operación por regiones:
# En región MPPT usamos λ_mppt (igualado a λ_opt_teo para que el control sea óptimo).
rpm_tsr = rpm_from_tsr(v_grid, D, lambda_mppt)
rpm_rotor = np.zeros_like(v_grid)

mask_reg2 = (v_grid >= v_cut_in) & (v_grid <= v_rated)
rpm_rotor[mask_reg2] = rpm_tsr[mask_reg2]

# rpm nominal coherente con el λ_mppt utilizado
rpm_rated_val = rpm_from_tsr(v_rated, D, lambda_mppt)


rpm_rotor = rpm_rotor_mppt(
    v_array=v_grid,
    D=D,
    lam_opt=lam_opt_ctrl,
    v_cut_in=v_cut_in,
    v_rated=v_rated,
    v_cut_out=v_cut_out,
    rpm_rotor_rated=rpm_rotor_rated,
)

# Chequeo de consistencia entre rpm_rotor_rated y la ley MPPT en v_rated
rpm_rated_ctrl = float(np.interp(v_rated, v_grid, rpm_rotor))
if abs(rpm_rotor_rated - rpm_rated_ctrl) > 5:
    st.warning(
        f"⚠️ rpm_rotor_rated ({rpm_rotor_rated:.1f} rpm) difiere de la rpm MPPT @ v_rated "
        f"({rpm_rated_ctrl:.1f} rpm). Revisa consistencia entre diseño aerodinámico, λ_opt y control MPPT."
    )


rpm_gen   = rpm_rotor * G
omega_rot = 2 * pi * rpm_rotor / 60.0
omega_gen = 2 * pi * rpm_gen   / 60.0

# TSR efectiva λ(v) y U_tip
lambda_eff = np.zeros_like(v_grid, dtype=float)
mask_v = v_grid > 0
lambda_eff[mask_v] = (omega_rot[mask_v] * R) / v_grid[mask_v]
U_tip = lambda_eff * v_grid

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

# Modelo simplificado de generador:
# P_el_gen = min(P_mec * η_gen_max, P_gen_curve)
P_el_gen_W = np.minimum(P_mec_gen_W * eta_gen_max, P_gen_curve_W)

# Eficiencia instantánea del generador (para info)
eta_gen_curve = np.divide(
    P_el_gen_W,
    np.maximum(P_mec_gen_W, 1.0),
    out=np.zeros_like(P_el_gen_W),
    where=(P_mec_gen_W > 0)
)
eta_gen_curve = np.clip(eta_gen_curve, 0.0, eta_gen_max)

# Potencia eléctrica después de electrónica
P_el_ac = P_el_gen_W * eta_elec

# Clipping por potencia nominal
P_el_ac_clip = np.minimum(P_el_ac, P_nom_kW * 1000.0)

# Torques
T_rotor_Nm = np.divide(P_aero_W, np.maximum(omega_rot, 1e-6))
T_gen_Nm   = T_rotor_Nm / np.maximum(G, 1e-9)

# Límite por T_gen_max
if T_gen_max > 0:
    T_gen_allowed = np.minimum(T_gen_Nm, T_gen_max)
    P_limit_by_T  = T_gen_allowed * omega_gen
    P_el_ac_clip  = np.minimum(P_el_ac_clip, P_limit_by_T)

# Frecuencia eléctrica
p_pairs = poles_total / 2.0
f_e_Hz  = p_pairs * rpm_gen / 60.0

PF = 0.95

# Corriente estimada: limpiando zona de muy baja tensión
V_eff = np.maximum(V_LL_curve, 1.0)
I_A = np.where(
    V_LL_curve < 10.0,
    0.0,
    np.divide(
        P_el_ac_clip,
        np.sqrt(3) * V_eff * PF,
        out=np.zeros_like(P_el_ac_clip),
        where=(P_el_ac_clip > 0)
    )
)

V_LL_from_Ke = Ke_vsr_default * omega_gen

# Cp equivalente por etapa
P_out_W = P_el_ac_clip
Cp_aero = np.divide(
    P_aero_W,
    0.5 * rho * A * (v_grid ** 3),
    out=np.zeros_like(v_grid), where=(v_grid > 0)
)
Cp_shaft = np.divide(
    P_mec_gen_W,
    0.5 * rho * A * (v_grid ** 3),
    out=np.zeros_like(v_grid), where=(v_grid > 0)
)
Cp_el = np.divide(
    P_out_W,
    0.5 * rho * A * (v_grid ** 3),
    out=np.zeros_like(v_grid), where=(v_grid > 0)
)

# Reynolds en pala (aprox. con U_tip)
Re_mid = np.zeros_like(v_grid)
if mu > 0:
    Re_mid = rho * U_tip * c / mu

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
    "P_el (kW)":         np.round(P_el_ac / 1000.0, 2),
    "P_out (clip) kW":   np.round(P_el_ac_clip / 1000.0, 2),
    "I_est (A)":         np.round(I_A, 1),
    "Lw (dB)":           np.round(Lw_dB, 1),
    "Lp_obs (dB)":       np.round(Lp_dB, 1),
})
# =========================
# PÉRDIDAS POR ETAPA [W]
# =========================
P_loss_mec_W  = np.maximum(P_aero_W    - P_mec_gen_W, 0.0)
P_loss_gen_W  = np.maximum(P_mec_gen_W - P_el_gen_W,  0.0)
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

Re_8 = np.interp(8.0, v_grid, Re_mid) if (v_grid[0] <= 8.0 <= v_grid[-1]) else Re_mid[-1]
Re_max = Re_mid[-1] if len(Re_mid) > 0 else 0.0

st.markdown("## 📊 Panel técnico de KPIs")

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
        kpi_card("TSR objetivo λ", f"{tsr:.2f}", "Setpoint de control aerodinámico")

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
        kpi_card("F CEN. / PALA ≈ m·R·w²", f"{F_centripetal_per_blade/1000:.1f} kN", "Esfuerzo radial en raíz de pala (m: masa; R: radio; ω: velocidad angular)",)

    p7, p8 = st.columns(2)
    with p7:
        kpi_card("Re @ 8 m/s ≈ (ρ·U_tip·c)/u",f"{Re_8:,.0f}", "Régimen aerodinámico de diseño (ρ: densidad; U_tip: punta; c: cuerda; μ: viscosidad)",)
    with p8:
        kpi_card("Re @ v_max ≈ (ρ·U_tip,max·c)/u",f"{Re_max:,.0f}","Régimen aerodinámico límite operativo para alta velocidad",)

    st.caption(
        "Las propiedades de la pala permiten evaluar esfuerzos en uniones, ejes y rodamientos, "
        "además de la respuesta dinámica del rotor. Re indica el régimen aerodinámico del perfil."
    )
st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# Tabla de resultados + filtro tipo píldoras
# =========================================================

modulos_columnas = {
    "Rotor (aero + dinámica)": [
        "v (m/s)", "λ_efectiva", "U_tip (m/s)",
        "Re (mid-span)", "Cp(λ_efectiva)", "Cp_aero_equiv",
        "rpm_rotor", "T_rotor (N·m)", "f_1P (Hz)", "f_3P (Hz)"
    ],
    "Tren mecánico": [
        "v (m/s)", "P_aero (kW)", "P_mec_gen (kW)",
        "Cp_shaft_equiv"
    ],
    "Generador + eléctrico": [
        "v (m/s)", "rpm_gen", "P_gen_curve (kW)",
        "V_LL (V)", "V_LL (Ke) [V]", "f_e (Hz)",
        "η_gen (curve)", "T_gen (N·m)",
        "P_el (kW)", "P_out (clip) kW", "I_est (A)",
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
st.markdown("### Ver módulo")

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
    df_view = df
else:
    cols = [c for c in modulos_columnas.get(mod_sel, []) if c in df.columns]
    df_view = df[cols] if cols else df

# ---------- TABLA + DESCARGA ----------
st.dataframe(
    df_view,
    use_container_width=True,
    height=480,
    column_config={

        "v (m/s)": st.column_config.NumberColumn(
            "v (m/s)",
            help=(
                "Descripción: Velocidad del viento incidente sobre el rotor.\n"
                "Fórmula: — (dato de entrada / SCADA / Weibull).\n"
                "Parámetros: v = velocidad del viento [m/s]."
            )
        ),

        "rpm_rotor": st.column_config.NumberColumn(
            "rpm_rotor",
            help=(
                "Descripción: Velocidad de giro del rotor según la ley de control MPPT.\n"
                "Fórmulas:\n"
                "• Región 2 (MPPT): rpm_rotor = (30 / (π · R)) · λ_ctrl · v.\n"
                "• Región 3 (nominal): rpm_rotor = rpm_rotor_rated (constante).\n"
                "Parámetros:\n"
                "• λ_ctrl = TSR objetivo definido en el panel de control.\n"
                "• R = radio del rotor [m].\n"
                "• v = velocidad del viento [m/s].\n"
                "• rpm_rotor_rated = velocidad nominal fija del rotor."
            )
        ),

        "rpm_gen": st.column_config.NumberColumn(
            "rpm_gen",
            help=(
                "Descripción: Velocidad de giro del generador resultante del control MPPT.\n"
                "Fórmula: rpm_gen = rpm_rotor · G.\n"
                "Parámetros:\n"
                "• rpm_rotor = velocidad del rotor (MPPT en Región 2, fija en Región 3).\n"
                "• G = relación de transmisión rpm_gen/rpm_rotor."
            )
        ),

        "λ_efectiva": st.column_config.NumberColumn(
            "λ_efectiva",
            help=(
                "Descripción: TSR efectiva del rotor.\n"
                "Fórmula general: λ_efectiva = ω_rot · R / v.\n"
                "Notas:\n"
                "• En Región 2: λ_efectiva ≈ λ_ctrl (MPPT mantiene TSR constante).\n"
                "• En Región 3: λ_efectiva baja al mantenerse rpm_rotor constante.\n"
                "Parámetros:\n"
                "• ω_rot = 2π · rpm_rotor / 60.\n"
                "• R = radio del rotor [m].\n"
                "• v = velocidad del viento [m/s].\n"
                "• λ_ctrl = TSR objetivo del panel (control MPPT)."
            )
        ),

        "U_tip (m/s)": st.column_config.NumberColumn(
            "U_tip (m/s)",
            help=(
                "Descripción: Velocidad de punta de pala.\n"
                "Fórmula: U_tip = λ_efectiva · v.\n"
                "Parámetros: λ_efectiva = TSR efectiva, v = velocidad del viento [m/s]."
            )
        ),

        "Cp(λ_efectiva)": st.column_config.NumberColumn(
            "Cp(λ_efectiva)",
            help=(
                "Descripción: Coeficiente de potencia aerodinámico del rotor en λ_efectiva.\n"
                "Fórmula: Cp(λ) ≈ c_max · (λ/λ_opt) · exp(1 − λ/λ_opt) (modelo Cp(λ)).\n"
                "Parámetros: c_max = Cp máximo, λ_opt = TSR óptimo, λ = λ_efectiva."
            )
        ),

        "Cp_aero_equiv": st.column_config.NumberColumn(
            "Cp_aero_equiv",
            help=(
                "Descripción: Cp equivalente de la potencia aerodinámica.\n"
                "Fórmula: Cp_aero = P_aero / (0.5 · ρ · A · v³).\n"
                "Parámetros: P_aero = potencia aerodinámica [W], ρ = densidad del aire [kg/m³], "
                "A = área barrida D·H [m²], v = velocidad del viento [m/s]."
            )
        ),

        "Cp_shaft_equiv": st.column_config.NumberColumn(
            "Cp_shaft_equiv",
            help=(
                "Descripción: Cp equivalente en el eje del generador (tras pérdidas mecánicas).\n"
                "Fórmula: Cp_shaft = P_mec_gen / (0.5 · ρ · A · v³).\n"
                "Parámetros: P_mec_gen = potencia mecánica en eje del generador [W], ρ, A, v como antes."
            )
        ),

        "Cp_el_equiv": st.column_config.NumberColumn(
            "Cp_el_equiv",
            help=(
                "Descripción: Cp equivalente eléctrico tras todas las pérdidas hasta entrega AC (salida útil).\n"
                "Fórmula: Cp_el = P_out / (0.5 · ρ · A · v³).\n"
                "Parámetros: P_out = potencia eléctrica útil con clipping [W], ρ = densidad, A = D·H, v = viento."
            )
        ),

        "Re (mid-span)": st.column_config.NumberColumn(
            "Re (mid-span)",
            help=(
                "Descripción: Número de Reynolds en la sección media de la pala.\n"
                "Fórmula: Re = ρ · U_tip · c / μ.\n"
                "Parámetros: ρ = densidad del aire [kg/m³], U_tip = velocidad de punta [m/s], "
                "c = cuerda de la pala [m], μ = viscosidad dinámica [Pa·s]."
            )
        ),

        "P_aero (kW)": st.column_config.NumberColumn(
            "P_aero (kW)",
            help=(
                "Descripción: Potencia aerodinámica capturada por el rotor.\n"
                "Fórmula: P_aero = 0.5 · ρ · A · v³ · Cp(λ_efectiva).\n"
                "Parámetros: ρ, A, v, Cp(λ_efectiva) según modelo aerodinámico."
            )
        ),

        "P_mec_gen (kW)": st.column_config.NumberColumn(
            "P_mec_gen (kW)",
            help=(
                "Descripción: Potencia mecánica disponible en el eje del generador.\n"
                "Fórmula: P_mec_gen = P_aero · η_mec.\n"
                "Parámetros: P_aero = potencia aerodinámica [W], η_mec = η_rodamientos · η_caja."
            )
        ),

        "P_gen_curve (kW)": st.column_config.NumberColumn(
            "P_gen_curve (kW)",
            help=(
                "Descripción: Potencia nominal del generador según su curva P(rpm).\n"
                "Fórmula: P_gen_curve = interp_P(rpm_gen).\n"
                "Parámetros: rpm_gen = velocidad del generador [rpm], curva P_kW(rpm) de datasheet/CSV."
            )
        ),

        "η_gen (curve)": st.column_config.NumberColumn(
            "η_gen (curve)",
            help=(
                "Descripción: Eficiencia instantánea del generador.\n"
                "Fórmula: η_gen = P_el_gen / P_mec_gen.\n"
                "Parámetros: P_el_gen = potencia eléctrica en bornes del generador [W], "
                "P_mec_gen = potencia mecánica de entrada [W]."
            )
        ),

        "V_LL (V)": st.column_config.NumberColumn(
            "V_LL (V)",
            help=(
                "Descripción: Tensión línea-línea del generador según curva nominal.\n"
                "Fórmula: V_LL = interp_V(rpm_gen).\n"
                "Parámetros: rpm_gen = velocidad del generador [rpm], curva V_LL(rpm) de datasheet/CSV."
            )
        ),

        "V_LL (Ke) [V]": st.column_config.NumberColumn(
            "V_LL (Ke) [V]",
            help=(
                "Descripción: Tensión línea-línea estimada usando la constante eléctrica Ke.\n"
                "Fórmula: V_LL_Ke = Ke · ω_gen.\n"
                "Parámetros: Ke = constante [V·s/rad], ω_gen = velocidad angular del generador [rad/s]."
            )
        ),

        "f_e (Hz)": st.column_config.NumberColumn(
            "f_e (Hz)",
            help=(
                "Descripción: Frecuencia eléctrica trifásica del generador.\n"
                "Fórmula: f_e = (p/2) · (rpm_gen / 60).\n"
                "Parámetros: p = número total de polos, rpm_gen = velocidad del generador [rpm]."
            )
        ),

        "f_1P (Hz)": st.column_config.NumberColumn(
            "f_1P (Hz)",
            help=(
                "Descripción: Frecuencia de paso 1P del rotor (una vuelta completa).\n"
                "Fórmula: f_1P = rpm_rotor / 60.\n"
                "Parámetros: rpm_rotor = velocidad del rotor [rpm]."
            )
        ),

        "f_3P (Hz)": st.column_config.NumberColumn(
            "f_3P (Hz)",
            help=(
                "Descripción: Frecuencia de paso 3P (paso de palas en rotor de 3 palas).\n"
                "Fórmula: f_3P = 3 · f_1P.\n"
                "Parámetros: f_1P = frecuencia de paso fundamental [Hz], N_pal = 3."
            )
        ),

        "T_rotor (N·m)": st.column_config.NumberColumn(
            "T_rotor (N·m)",
            help=(
                "Descripción: Par aerodinámico en el eje del rotor.\n"
                "Fórmula: T_rotor = P_aero / ω_rot.\n"
                "Parámetros: P_aero = potencia aerodinámica [W], ω_rot = velocidad angular del rotor [rad/s]."
            )
        ),

        "T_gen (N·m)": st.column_config.NumberColumn(
            "T_gen (N·m)",
            help=(
                "Descripción: Par transmitido al eje del generador.\n"
                "Fórmula: T_gen = T_rotor / G.\n"
                "Parámetros: T_rotor = par en el rotor [N·m], G = relación de transmisión."
            )
        ),

        "P_el (kW)": st.column_config.NumberColumn(
            "P_el (kW)",
            help=(
                "Descripción: Potencia eléctrica AC antes del clipping (tras electrónica de potencia).\n"
                "Fórmula: P_el = P_el_gen · η_elec.\n"
                "Parámetros: P_el_gen = potencia eléctrica del generador [W], η_elec = eficiencia electrónica (rect+inv)."
            )
        ),

        "P_out (clip) kW": st.column_config.NumberColumn(
            "P_out (clip) kW",
            help=(
                "Descripción: Potencia eléctrica útil limitada por la potencia nominal (clipping).\n"
                "Fórmula: P_out = min(P_el, P_nom).\n"
                "Parámetros: P_el = potencia eléctrica antes de clipping [W], P_nom = potencia nominal del sistema [W]."
            )
        ),

        "I_est (A)": st.column_config.NumberColumn(
            "I_est (A)",
            help=(
                "Descripción: Corriente trifásica estimada en bornes del generador/inversor.\n"
                "Fórmula: I_est = P_out / (√3 · V_LL · PF).\n"
                "Parámetros: P_out = potencia de salida [W], V_LL = tensión línea-línea [V], PF = factor de potencia (≈0.95)."
            )
        ),

        "Lw (dB)": st.column_config.NumberColumn(
            "Lw (dB)",
            help=(
                "Descripción: Nivel de potencia sonora de la turbina.\n"
                "Fórmula: L_w = L_w_ref + 10 · n · log10(U_tip / U_tip_ref).\n"
                "Parámetros: L_w_ref = nivel de referencia [dB], n = exponente, U_tip = velocidad de punta, "
                "U_tip_ref = velocidad de referencia."
            )
        ),

        "Lp_obs (dB)": st.column_config.NumberColumn(
            "Lp_obs (dB)",
            help=(
                "Descripción: Nivel de presión sonora estimado en el punto del observador.\n"
                "Fórmula: L_p = L_w − 20 · log10(r_obs) − 11.\n"
                "Parámetros: L_w = nivel de potencia sonora [dB], r_obs = distancia al observador [m]."
            )
        ),

        "P_loss_mec (kW)": st.column_config.NumberColumn(
            "P_loss_mec (kW)",
            help=(
                "Descripción: Pérdidas mecánicas entre el rotor y el eje del generador.\n"
                "Fórmula: P_loss_mec = P_aero − P_mec_gen.\n"
                "Parámetros: P_aero = potencia aerodinámica [W], P_mec_gen = potencia mecánica en el eje [W]."
            )
        ),

        "P_loss_gen (kW)": st.column_config.NumberColumn(
            "P_loss_gen (kW)",
            help=(
                "Descripción: Pérdidas internas del generador eléctrico.\n"
                "Fórmula: P_loss_gen = P_mec_gen − P_el_gen.\n"
                "Parámetros: P_mec_gen = potencia mecánica [W], P_el_gen = potencia eléctrica generador [W]."
            )
        ),

        "P_loss_elec (kW)": st.column_config.NumberColumn(
            "P_loss_elec (kW)",
            help=(
                "Descripción: Pérdidas en electrónica de potencia (rectificador + inversor, etc.).\n"
                "Fórmula: P_loss_elec = P_el_gen − P_el.\n"
                "Parámetros: P_el_gen = potencia eléctrica del generador [W], P_el = potencia después de electrónica [W]."
            )
        ),

        "P_loss_clip (kW)": st.column_config.NumberColumn(
            "P_loss_clip (kW)",
            help=(
                "Descripción: Potencia recortada por clipping al alcanzar el límite nominal.\n"
                "Fórmula: P_loss_clip = P_el − P_out.\n"
                "Parámetros: P_el = potencia eléctrica antes de clipping [W], P_out = potencia útil tras clipping [W]."
            )
        ),
    },
)



# --- Botón para descargar CSV de la tabla ---


st.download_button(
    f"📥 Descargar CSV – vista: {mod_sel}",
    data=df_view.to_csv(index=False).encode("utf-8"),
    file_name=f"vawt_resultados_{mod_sel.replace(' ', '_')}.csv",
    mime="text/csv",
    key="csv_tabla_resultados"
)
# --- Ficha técnica de columnas principales ---
with st.expander("📘 Guía rápida – columnas clave de la tabla"):
    st.markdown(
        """
<span class="formula-bullet"><b>λ_efectiva</b><br>
<span class="formula-inline">
Descripción: TSR efectiva del rotor (relación entre velocidad de punta y viento).<br>
Fórmula: λ = ω<sub>rot</sub> · R / v<br>
Parámetros: ω<sub>rot</sub> = 2π·rpm_rotor/60 [rad/s], R = radio del rotor [m], v = velocidad del viento [m/s].
</span>
</span>

<br>

<span class="formula-bullet"><b>Cp_el_equiv</b><br>
<span class="formula-inline">
Descripción: Cp equivalente eléctrico tras todas las pérdidas hasta la entrega AC (potencia útil).<br>
Fórmula: Cp<sub>el</sub> = P_out / (0.5 · ρ · A · v³)<br>
Parámetros: P_out = potencia eléctrica útil con clipping [W], ρ = densidad del aire [kg/m³], A = D·H [m²], v = viento [m/s].
</span>
</span>

<br>

<span class="formula-bullet"><b>P_out (clip) kW</b><br>
<span class="formula-inline">
Descripción: Potencia eléctrica de salida limitada por la potencia nominal del sistema.<br>
Fórmula: P_out = min(P_el, P_nom)<br>
Parámetros: P_el = potencia eléctrica antes de clipping [W], P_nom = potencia nominal [W].
</span>
</span>

<br>

<span class="formula-bullet"><b>Re (mid-span)</b><br>
<span class="formula-inline">
Descripción: Número de Reynolds en la sección media de la pala, asociado al régimen aerodinámico del perfil.<br>
Fórmula: Re = ρ · U_tip · c / μ<br>
Parámetros: ρ = densidad del aire [kg/m³], U_tip = velocidad de punta [m/s], c = cuerda [m], μ = viscosidad dinámica [Pa·s].
</span>
</span>
        """,
        unsafe_allow_html=True,
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


# =========================================================
# Gráfico 1 – rpm rotor / rpm generador (ancho completo)
# =========================================================
st.subheader("⚙️ rpm rotor / rpm generador")

# Datos ordenados + región de operación
df_rpm_plot = df.sort_values("v (m/s)").copy()
v_vals = df_rpm_plot["v (m/s)"].values

region = np.where(
    v_vals < v_cut_in, "Parado",
    np.where(v_vals <= v_rated, "MPPT (λ≈const)",
             np.where(v_vals <= v_cut_out, "Potencia limitada", "Parado"))
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
    annotation_text="Región MPPT",
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
  Este gráfico muestra simultáneamente el comportamiento del rotor y el generador bajo la ley de control por regiones.
  Las líneas verticales indican los puntos de transición entre <em>cut-in</em>, operación MPPT y potencia nominal.
  Las bandas sombreadas distinguen la región de <strong>seguimiento de λ (MPPT)</strong> y la región de
  <strong>potencia limitada</strong>.
  Las líneas horizontales de <strong>rpm_rated</strong> permiten verificar que la relación de transmisión
  <strong>G</strong> lleva al generador a su régimen nominal sin sobrepasarlo.
  </p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# Gráfico – λ_efectiva, U_tip y Frecuencia eléctrica
# =========================================================
st.subheader("🚀 λ_efectiva, U_tip y Frecuencia eléctrica")

df_u = df.sort_values("v (m/s)").copy()

fig_u = px.line(
    df_u,
    x="v (m/s)",
    y=["λ_efectiva", "U_tip (m/s)", "f_e (Hz)"],
    markers=True,
)

fig_u.update_layout(
    xaxis_title="v (m/s)",
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

# Líneas verticales v_cut-in, v_rated, v_cut-out
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

# Región sombreada entre v_rated y v_cut-out (frecuencia / punta de pala limitadas)
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
  Aquí se observa cómo varía el TSR efectivo (<strong>λ_efectiva</strong>), la velocidad de punta de pala
  (<strong>U_tip</strong>) y la frecuencia eléctrica (<strong>f<sub>e</sub></strong>) con el viento.
  Entre <em>v_cut-in</em> y <em>v_rated</em> el control mantiene <strong>λ</strong> cercano a
  <strong>λ<sub>opt</sub></strong>, por lo que U_tip y f<sub>e</sub> crecen de forma controlada (región MPPT).
  En la zona sombreada (entre <em>v_rated</em> y <em>v_cut-out</em>) se aprecia la operación a potencia limitada,
  donde la velocidad del generador y la frecuencia tienden a estabilizarse, permitiendo verificar restricciones
  de ruido, fatiga y compatibilidad con la electrónica de potencia.
  </p>
</div>
""", unsafe_allow_html=True)


    # =====================================================================
# POTENCIAS VS VIENTO – DOS MODOS
# =====================================================================
st.subheader("Potencia vs Viento")

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

    y_cols_P = [
        "P_aero (kW)",
        "P_mec_gen (kW)",
        "P_out (clip) kW",
    ]

    dfP = df.sort_values("v (m/s)").copy()

    if pot_norm and P_nom_kW > 0:
        for col in y_cols_P:
            dfP[col] = dfP[col] / P_nom_kW
        y_label = "Potencia [p.u. de P_nom]"
        hline_y = 1.0
    else:
        y_label = "Potencia [kW]"
        hline_y = P_nom_kW

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

    st.plotly_chart(figP, use_container_width=True)

    # INTERPRETACIÓN TÉCNICA – MODO VIENTO
    st.markdown(
        """
<div class="comment-box">
  <div class="comment-title">🔍 Interpretación técnica</div>
  <p>
    El gráfico muestra la evolución de <strong>P_aero</strong>,
    <strong>P_mec_gen</strong> y <strong>P_out</strong> en función de la
    velocidad del viento. En la región MPPT (entre <em>v_cut-in</em> y
    <em>v_rated</em>) <strong>P_aero</strong> y <strong>P_mec_gen</strong>
    crecen aproximadamente con <em>v³</em>, lo que indica un seguimiento correcto
    del punto de máxima potencia y permite cuantificar las pérdidas mecánicas
    entre rotor y eje del generador.
  </p>
  <p>
    A partir de <em>v_rated</em>, <strong>P_out</strong> se recorta y se
    mantiene cercana a <em>P_nom</em> hasta <em>v_cut-out</em>, definiendo la
    región de potencia limitada. La separación entre
    <strong>P_aero</strong>, <strong>P_mec_gen</strong> y
    <strong>P_out</strong> refleja las pérdidas aerodinámicas, mecánicas y
    eléctricas del aerogenerador, y permite verificar que el control protege
    al generador respetando su potencia nominal.
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

# customdata para mostrar eficiencias en el hover
custom = np.stack([eta_mec_loc, eta_el_loc, eta_tot_loc], axis=-1)

fig_cp_eq = go.Figure()

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

# --- Línea horizontal: límite de Betz ---
CP_BETZ = 16.0 / 27.0
fig_cp_eq.add_hline(
    y=CP_BETZ,
    line_dash="dot",
    line_color="rgba(234,179,8,0.9)",
    annotation_text="Límite de Betz",
    annotation_position="top left",
)

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

# --- Estilo de ejes ---
fig_cp_eq.update_xaxes(
    title_text="v (m/s)",
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
    El gráfico muestra cómo evoluciona el <strong>Cp equivalente</strong> en cada etapa del sistema:
    <strong>rotor (Cp_aero)</strong>, <strong>eje del generador (Cp_shaft)</strong> y
    <strong>salida eléctrica (Cp_el)</strong>, en función de la velocidad del viento.
  </p>
  <p>
    Entre <em>v_cut-in</em> y <em>v_rated</em> las tres curvas se mantienen casi planas: el control MPPT
    mantiene la TSR cercana a <strong>λ_opt</strong>, por lo que el rotor opera cerca de su rendimiento máximo.
    La separación casi constante entre <strong>Cp_aero</strong> y <strong>Cp_shaft</strong> refleja las
    pérdidas mecánicas (rodamientos + caja), mientras que la diferencia entre <strong>Cp_shaft</strong> y
    <strong>Cp_el</strong> cuantifica las pérdidas del generador y de la electrónica de potencia.
  </p>
  <p>
    A partir de <em>v_rated</em>, en la región sombreada de <strong>potencia limitada</strong>, el
    <strong>Cp_el</strong> cae de forma marcada: la potencia eléctrica se mantiene prácticamente constante
    mientras la potencia disponible del viento sigue creciendo con <em>v³</em>, por lo que el rendimiento
    global baja aunque el tren mecánico y el generador sigan siendo eficientes. El hecho de que
    <strong>Cp_aero</strong> se mantenga bien por debajo del <strong>límite de Betz</strong> es coherente
    con un VAWT realista, donde valores entorno al 40–50&nbsp;% de dicho límite son típicos.
  </p>
  <p>
    En conjunto, este gráfico permite ver en qué rango de vientos el piloto convierte mejor la energía del
    viento y en qué etapas (mecánica, generador, electrónica o <em>clipping</em>) se concentran las
    pérdidas que alejan al sistema del máximo teórico.
  </p>
</div>
""", unsafe_allow_html=True)




# =========================================================
# PÉRDIDAS POR ETAPA (MECÁNICA, GENERADOR, ELECTRÓNICA, CLIPPING)
# =========================================================
st.subheader("🔍 Pérdidas por etapa (mecánica, generador, electrónica, clipping)")

dfL = df.sort_values("v (m/s)").copy()

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
        x="v (m/s)",
        y=loss_cols,
    )

    fig_loss.update_layout(
        xaxis_title="v (m/s)",
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

    st.plotly_chart(fig_loss, use_container_width=True)

    # ===========================
    # INTERPRETACIÓN TÉCNICA
    # ===========================
    st.markdown(
        """
<div class="comment-box">
  <div class="comment-title">🔍 Interpretación técnica</div>
  <p>
    El área apilada muestra cuánto se pierde en cada etapa del sistema
    (rodamientos+caja, generador, electrónica y <em>clipping</em> por nominal/par)
    en función del viento.
  </p>
  <p>
    A bajas velocidades las pérdidas totales son reducidas; a partir de
    <em>v_rated</em>, la región sombreada de potencia limitada evidencia cómo
    aumentan principalmente las pérdidas del generador y el <em>clipping</em> para
    mantener <em>P_nom</em>. Este gráfico permite priorizar dónde conviene actuar:
    mejorar el tren mecánico, optimizar el diseño del generador o ajustar la
    electrónica de potencia y la potencia nominal.
  </p>
</div>
""",
        unsafe_allow_html=True,
    )

# ==========================================================
# Torque (rotor y generador)
# ==========================================================
st.subheader("🧲 Torque (rotor y generador) ")

# Datos importantes del generador (ficha técnica)
T_gen_nom = GDG_RATED_T_Nm   # 3460 N·m
I_nom     = GDG_RATED_I
T_gen_safe = T_gen_nom * 1.10  # umbral “zona amarilla”

# Ordenar por viento
dfT = df.sort_values("v (m/s)").copy()

# Pasar a formato largo para usar px.line
dfT_long = dfT.melt(
    id_vars=["v (m/s)"],
    value_vars=["T_rotor (N·m)", "T_gen (N·m)"],
    var_name="Variable",
    value_name="T [N·m]",
)

# Mapa más legible de nombres
dfT_long["Variable"] = dfT_long["Variable"].map({
    "T_rotor (N·m)": "T_rotor (N·m)",
    "T_gen (N·m)":   "T_gen (N·m)",
})

# FIGURA BASE
figT = px.line(
    dfT_long,
    x="v (m/s)",
    y="T [N·m]",
    color="Variable",
    markers=True,
)

# Estilo general coherente con el resto
figT.update_layout(
    xaxis_title="v (m/s)",
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
figT.update_traces(
    hovertemplate=(
        "v = %{x:.1f} m/s<br>"
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

st.plotly_chart(figT, use_container_width=True)

st.markdown("""
<div class="comment-box">
  <div class="comment-title">🔍 Interpretación técnica (Par)</div>
  <p>
    La curva <strong>T_rotor</strong> muestra el par disponible en el eje de la turbina, mientras que
    <strong>T_gen</strong> representa el par efectivo en el eje del generador después de la caja multiplicadora.
    La franja verde indica la zona de operación segura del generador (0–T<sub>nom</sub>), mientras que la zona
    rojiza marca regímenes de <em>sobre-torque</em> que deberían ser transitorios o estar protegidos por el control.
  </p>
  <p>
    La línea discontinua de <strong>T_nom gen</strong> (3460 N·m) y el límite
    <strong>IEC T_rotor</strong> permiten verificar si, para el rango de vientos analizado, la estrategia de control
    y la elección de la relación de transmisión <strong>G</strong> mantienen a la máquina dentro de un esfuerzo
    mecánico admisible tanto en el rotor como en el generador, considerando además las velocidades
    <strong>v_rated</strong>, <strong>v_cut-out</strong> y <strong>v_shutdown IEC</strong>.
  </p>
</div>
""", unsafe_allow_html=True)

# =========================================================
# Módulo 3 – Alertas de diseño / operación (IEC-style)
# =========================================================
st.subheader("🚨 Alertas de diseño / operación")

flags = []

# Máximos de operación desde la simulación
max_T_gen   = float(df["T_gen (N·m)"].max())
max_T_rotor = float(df["T_rotor (N·m)"].max())
max_I_est   = float(df["I_est (A)"].max())
max_rpm_rot = float(df["rpm_rotor"].max())
max_P_out   = float(df["P_out (clip) kW"].max())

# 1) Torque generador vs nominal y vs T_gen_max de entrada
#    (T_gen_nom viene de la ficha GDG-1100 más arriba)
margen_Tgen_nom = 1.0
if T_gen_nom > 0:
    over_pct = (max_T_gen - T_gen_nom) / T_gen_nom * 100
    if over_pct > 5:
        flags.append(
            f"⚠️ El par máximo en el generador ({max_T_gen:,.0f} N·m) "
            f"supera el par nominal de ficha ({T_gen_nom:,.0f} N·m) "
            f"en un {over_pct:,.0f} %.Revisa G, TSR objetivo o estrategia de control."
        )


# Límite adicional definido por usuario (T_gen_max)
if T_gen_max > 0 and max_T_gen > 1.05 * T_gen_max:
    flags.append(
        f"⚠️ El par máximo en el generador ({max_T_gen:,.0f} N·m) excede el límite de diseño "
        f"configurado T_gen_max = {T_gen_max:,.0f} N·m (IEC / criterio estructural)."
    )

# 2) Torque rotor vs límite IEC (T_rotor_max_iec)
margen_Trot_iec = 1.0
try:
    if T_rotor_max_iec > 0:
        margen_Trot_iec = (T_rotor_max_iec - max_T_rotor) / T_rotor_max_iec
        if max_T_rotor > 1.02 * T_rotor_max_iec:
            flags.append(
                f"⚠️ El par máximo en el rotor ({max_T_rotor:,.0f} N·m) supera el límite IEC configurado "
                f"T_rotor_max_iec = {T_rotor_max_iec:,.0f} N·m. Requiere revisión estructural."
            )
except NameError:
    # Si por algún motivo no se definió en el sidebar
    pass

# 3) Corriente vs nominal del generador
margen_I = 1.0
if GDG_RATED_I > 0:
    margen_I = (GDG_RATED_I - max_I_est) / GDG_RATED_I
    if max_I_est > 1.05 * GDG_RATED_I:
        flags.append(
            f"⚠️ La corriente máxima estimada ({max_I_est:,.1f} A) supera en más de un 5% "
            f"la corriente nominal de la máquina ({GDG_RATED_I:.1f} A). "
            "Revisa el dimensionamiento de cables, protecciones y el setpoint de potencia."
        )

# 4) rpm rotor vs límite IEC
margen_rpm = 1.0
try:
    if rpm_rotor_max_iec > 0:
        margen_rpm = (rpm_rotor_max_iec - max_rpm_rot) / rpm_rotor_max_iec
        if max_rpm_rot > 1.02 * rpm_rotor_max_iec:
            flags.append(
                f"⚠️ La rpm máxima del rotor ({max_rpm_rot:.1f} rpm) excede el límite IEC configurado "
                f"rpm_rotor_max_iec = {rpm_rotor_max_iec:.1f} rpm. Ajusta el control de velocidad / shutdown."
            )
except NameError:
    pass

# 5) Potencia eléctrica vs nominal P_nom_kW
margen_P = 1.0
if P_nom_kW > 0:
    margen_P = (P_nom_kW - max_P_out) / P_nom_kW
    if max_P_out > 1.02 * P_nom_kW:
        flags.append(
            f"⚠️ La potencia máxima de salida ({max_P_out:.1f} kW) supera en más de un 2% "
            f"la potencia nominal del sistema ({P_nom_kW:.1f} kW). Revisa el clipping y los límites del inversor."
        )

# Panel de márgenes de seguridad
cA, cB, cC, cD = st.columns(4)

def fmt_pct(m):
    return f"{m*100:.1f} %" if np.isfinite(m) else "N/A"

with cA:
    st.metric(
        "Margen T_gen vs T_nom",
        fmt_pct(margen_Tgen_nom),
        help="(T_nom - T_max) / T_nom. Valores negativos indican sobre-carga."
    )
with cB:
    st.metric(
        "Margen T_rotor vs IEC",
        fmt_pct(margen_Trot_iec),
        help="(T_rotor_max_iec - T_rotor_max) / T_rotor_max_iec."
    )
with cC:
    st.metric(
        "Margen I_est vs I_nom",
        fmt_pct(margen_I),
        help="(I_nom - I_max_est) / I_nom."
    )
with cD:
    st.metric(
        "Margen P_out vs P_nom",
        fmt_pct(margen_P),
        help="(P_nom - P_max_out) / P_nom."
    )

# Listado de alertas
if flags:
    st.markdown("#### Estado de diseño / operación")
    for f in flags:
        st.markdown(f"- {f}")
else:
    st.success("✅ Dentro de los límites configurados: sin alertas críticas para el rango de viento analizado.")

st.markdown("""
<div class="comment-box">
  <div class="comment-title">🔍 Interpretación técnica (alertas)</div>
  <p>
  Este módulo resume si la configuración del piloto respeta los límites mecánicos, eléctricos y normativos que definiste:
  </p>
  <ul>
    <li><strong>Margen T_gen vs T_nom</strong>: cuánto espacio queda entre el par máximo simulado y el nominal del generador.</li>
    <li><strong>Margen T_rotor vs IEC</strong>: qué tan cerca estás del límite estructural del rotor definido por IEC 61400-2.</li>
    <li><strong>Margen I_est vs I_nom</strong>: cuánto margen hay antes de saturar térmicamente el generador y los cables.</li>
    <li><strong>Margen P_out vs P_nom</strong>: indica si la electrónica y el dimensionamiento de potencia están bien escalados.</li>
  </ul>
  <p>
  Si aparecen alertas, el siguiente paso es iterar G, TSR objetivo, v_rated o el dimensionamiento del generador antes de escalar
  la tecnología hacia la turbina de 80 kW.
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
    Este mapa muestra la curva de operación del generador en el plano <strong>T_gen–rpm_gen</strong> y la compara
    con una envolvente admisible simplificada:
  </p>
  <ul>
    <li>La zona <strong>verde</strong> corresponde a operación dentro de <em>T_nom</em> y hasta ~105&nbsp;% de rpm nominal.</li>
    <li>La zona <strong>amarilla</strong> indica regímenes donde se aproxima o supera ligeramente el par o la velocidad de diseño:
        se toleran de forma transitoria, pero no deberían ser el punto de operación habitual.</li>
    <li>La zona <strong>roja</strong> representa combinaciones de par y rpm que quedan fuera de la envolvente admisible y que,
        en un diseño real, deberían gatillar limitación de par o estrategias de protección (derating, frenado, shutdown).</li>
  </ul>
  <p>
    Comparar la curva simulada con el punto nominal permite verificar si la estrategia MPPT y la elección de <em>G</em>
    mantienen al generador dentro de un sobreesfuerzo razonable, especialmente al escalar el piloto hacia potencias mayores.
  </p>
</div>
""", unsafe_allow_html=True)


# ==========================================================
# Corriente estimada vs velocidad de viento (con IEC)
# ==========================================================
# ==========================================================
# Corriente estimada vs velocidad de viento (con hover x-unified)
# ==========================================================
st.subheader("🔌 Corriente estimada vs velocidad de viento")

# Ordenamos por viento para que la curva quede limpia
dfI = df.sort_values("v (m/s)").copy()

figI = px.line(
    dfI,
    x="v (m/s)",
    y="I_est (A)",
    markers=True,
)

# Estilo de traza + tooltip
figI.update_traces(
    line=dict(width=2.6),
    marker=dict(size=7),
    hovertemplate=(
        "v = %{x:.1f} m/s<br>"
        "I_est = %{y:.1f} A<extra></extra>"
    ),
    name="I_est (A)",
    showlegend=False,
)

# Layout general + hover unificado
figI.update_layout(
    xaxis_title="v (m/s)",
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
    Este gráfico muestra la corriente trifásica estimada en función de la velocidad del viento
    y el comportamiento real del generador:
  </p>
  <ul>
    <li>La línea punteada <strong>I_nom gen</strong> representa la corriente nominal de ficha del generador.</li>
    <li>La franja resaltada sobre <strong>I_nom</strong> indica la zona donde, según los criterios de diseño de
        turbinas de pequeña potencia (IEC 61400-2), debería actuarse con protección térmica o limitar el par.</li>
    <li>Las líneas verticales en <strong>v_rated</strong> y <strong>v_cut-out</strong> permiten ver en qué rango de viento
        se alcanzan las corrientes nominales y si la estrategia de control mantiene el generador dentro de un
        sobreesfuerzo razonable.</li>
    <li>Con esta vista puedes chequear compatibilidad con cables, protecciones y electrónica de potencia
        para el piloto (&lt; 200 kW).</li>
  </ul>
</div>
""", unsafe_allow_html=True)

# ==========================================================
# Eficiencias por etapa
# ==========================================================
st.subheader("📈 Eficiencias: mecánica, generador y global")

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

# Estilo de trazas + hover
figE.update_traces(
    line=dict(width=2.4),
    marker=dict(size=7),
    hovertemplate=(
        "v = %{x:.1f} m/s<br>"
        "%{y:.1f} %<extra>%{fullData.name}</extra>"
    ),
)

# Layout general + hover unificado
figE.update_layout(
    xaxis_title="v (m/s)",
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

st.plotly_chart(figE, use_container_width=True)

st.markdown("""
<div class="comment-box">
  <div class="comment-title">🔍 Interpretación técnica</div>
  <p>
  Aquí se visualizan las eficiencias mecánica, del generador y global en función del viento.
  Una <strong>η_mec</strong> alta indica un tren de potencia bien diseñado; una <strong>η_gen</strong> estable
  refleja un generador correctamente dimensionado; y <strong>η_total</strong> resume el rendimiento real de la turbina
  desde el viento hasta la energía eléctrica útil, integrando todas las pérdidas intermedias.
  </p>
  <p>
  La banda sombreada entre <strong>v_rated</strong> y <strong>v_cut-out</strong> corresponde a la región de
  <em>potencia limitada</em> típica de turbinas de pequeña potencia (&lt; 200 kW, IEC 61400-2):
  en esta zona la potencia eléctrica se mantiene prácticamente constante por límites nominales,
  por lo que <strong>η_total</strong> disminuye con la velocidad aun cuando <strong>η_mec</strong> y
  <strong>η_gen</strong> se mantengan elevadas. No es un fallo del tren de potencia, sino una consecuencia
  directa de limitar la potencia de salida.
  </p>
</div>
""", unsafe_allow_html=True)

st.caption(
    "η_total = P_out / P_aero. Si la curva de 'Pérdida por clipping' domina desde cierta v, "
    "estás en región de potencia constante; considera redimensionar G/TSR o estrategia de control."
)


# ==========================================================
# Frecuencias 1P / 3P del rotor
# ==========================================================
st.subheader("📡 Frecuencias 1P / 3P del rotor")

# Ordenamos por viento y preparamos info extra para el hover
df_freq = df.sort_values("v (m/s)").copy()
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
            x=df_freq["v (m/s)"],
            y=df_freq[col],
            mode="lines+markers",
            name=name,
            customdata=custom,
            line=dict(width=2.4),
            marker=dict(size=7),
            hovertemplate=(
                "v = %{x:.1f} m/s<br>"
                "f = %{y:.3f} Hz<br>"
                "rpm_rotor = %{customdata[0]:.1f} rpm<br>"
                "λ_efectiva = %{customdata[1]:.2f}"
                "<extra></extra>"
            ),
        )
    )

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
    xaxis_title="v (m/s)",
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
  Las curvas muestran las <strong>frecuencias 1P</strong> (una vez por vuelta) y 
  <strong>3P</strong> (tres veces por vuelta para rotor de 3 palas), que concentran las principales
  cargas periódicas que excitan torre, cimentación y tren de potencia.
  </p>
  <p>
  La banda sombreada ilustra una <em>banda típica</em> de frecuencias propias de torre/fundación
  para turbinas de pequeña potencia; en tu diseño real debes reemplazarla por los modos calculados.
  El objetivo es que 1P y 3P no coincidan con esos modos: así evitas trabajar en
  <strong>resonancia</strong> o en zonas de amplificación dinámica.
  </p>
</div>
""", unsafe_allow_html=True)

# ==========================================================
# Curva Cp(λ)
# ==========================================================

st.subheader("🧩 Cp(λ) – Promedio, upwind y downwind")

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
fig_cp.add_vline(
    x=float(tsr),
    line_dash="dot",
    line_color="rgba(249,115,22,0.9)",  # naranja
    annotation_text="TSR objetivo",
    annotation_position="top left",
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
    annotation_position="top left",
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
  La curva <strong>Cp(λ)</strong> resume el rendimiento aerodinámico teórico del rotor, separando la contribución
  <em>upwind</em> y <em>downwind</em>. La comparación entre <strong>λ_opt</strong> y el <strong>TSR objetivo</strong>
  ayuda a ajustar el control y la geometría (solidez, helicoidal, perfil) para operar lo más cerca posible del máximo Cp
  en el rango de vientos de interés del proyecto.
  </p>
  <p>
  La banda sombreada alrededor de <strong>λ_opt</strong> representa la zona de operación recomendada para el control
  MPPT en turbinas de pequeña potencia (IEC 61400-2): mientras la turbina se mantenga dentro de esta banda, trabaja
  cerca del máximo rendimiento aerodinámico. La línea punteada del <strong>límite de Betz</strong> sirve como
  referencia del máximo teórico absoluto de cualquier rotor eólico.
  </p>
</div>
""", unsafe_allow_html=True)

# ==========================================================
# Ruido (si aplica)
# ==========================================================

if use_noise:
    st.subheader("🔈 Ruido estimado vs velocidad de viento")

    # --- Curva principal ---
    figNoise = px.line(
        df,
        x="v (m/s)",
        y=["Lw (dB)", "Lp_obs (dB)"],
        markers=True,
    )

    # --- Hover unificado y estilo principal ---
    figNoise.update_layout(
        xaxis_title="v (m/s)",
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
      El modelo de ruido usa como referencia un nivel <strong>Lw_ref = {Lw_ref_dB:.0f} dB</strong> a 
      <em>v_rated</em> y escala el nivel con una ley de potencia de la velocidad de punta
      (<code>U_tip^n</code>, con n={n_noise:.1f}). A partir de Lw se estima el nivel de presión
      sonora <strong>Lp_obs</strong> percibido a una distancia de <strong>{r_obs:.0f} m</strong>,
      asumiendo propagación en campo libre.
      </p>
      <p>
      La línea verde marca un <strong>nivel objetivo</strong> en el receptor (por ejemplo, 45 dB para
      entornos residenciales o sensibles) y la franja sombreada indica el rango de vientos en el que
      el piloto podría superar ese valor. Esto permite anticipar si será necesario:
      ajustar <em>TSR</em>, limitar rpm, rediseñar palas o considerar medidas de mitigación acústica
      en el proyecto &lt; 200 kW.
      </p>
    </div>
    """, unsafe_allow_html=True)



# =========================================================
# WEIBULL – SIEMPRE ACTIVO
# =========================================================

# Título ANTES de mostrar AEP y CF
st.subheader("🌬️ Distribución de viento vs curva de potencia")

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
  La curva <strong>Weibull f(v)</strong> muestra cómo se distribuyen las horas de viento a lo largo del año.
  La curva <strong>P_out(kW)</strong> es la potencia esperada del piloto para cada velocidad.
  La trazada punteada <strong>P_out·f(v)</strong> muestra directamente cómo contribuye cada velocidad al AEP.
  </p>
</div>
""", unsafe_allow_html=True)

# =========================================================
# NUEVO: Calibración modelo vs datos piloto (SCADA)
# =========================================================
st.subheader("🧪 Calibración modelo vs datos piloto (SCADA)")

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
    st.markdown("#### Potencia eléctrica: modelo vs piloto")

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
        st.markdown("#### rpm rotor: modelo vs piloto")

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
        st.markdown("#### Corriente: modelo vs piloto")

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
      La comparación modelo vs mediciones permite ajustar el diseño del piloto:
      <ul>
        <li><strong>Bias</strong> positivo indica que el modelo está siendo optimista en potencia.</li>
        <li><strong>RMSE</strong> cuantifica el error típico por bin de viento.</li>
        <li><strong>R²</strong> muestra qué tan bien el modelo reproduce la variabilidad real del piloto.</li>
      </ul>
      Si se observan desvíos sistemáticos en cierto rango de vientos, conviene revisar:
      Cp(λ), pérdidas mecánicas, curva interna del generador o configuración de control (TSR objetivo y G).
      </p>
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
# Resumen IEC 61400-2 – tabla operativa
# =========================================================
st.subheader("📋 Resumen IEC 61400-2 – operación por bin de viento")

df_iec = df[[
    "v (m/s)",
    "rpm_rotor",
    "rpm_gen",
    "λ_efectiva",
    "P_aero (kW)",
    "P_mec_gen (kW)",
    "P_out (clip) kW",
    "T_rotor (N·m)",
    "T_gen (N·m)",
    "Cp_aero_equiv",
    "Cp_el_equiv",
    "I_est (A)",
]]

st.dataframe(df_iec, use_container_width=True)

st.download_button(
    "📥 Descargar tabla IEC 61400-2 (CSV)",
    data=df_iec.to_csv(index=False).encode("utf-8"),
    file_name="IEC61400_2_resumen_operativo.csv",
    mime="text/csv"
)
st.markdown("""
---

### 📄 Nota técnica (IEC 61400-2)

Esta es la **tabla de operación del prototipo conforme a IEC 61400-2**:  
para cada *bin* de viento se documentan:

- **rpm del rotor y del generador**,  
- **TSR (λ)**,  
- **Torque** (rotor y eje lento/rápido),  
- **Potencia aerodinámica, mecánica y eléctrica**,  
- **Cp equivalente** según región de control (cut-in / rated / cut-out),  
- **Corriente trifásica estimada** en el generador al punto operativo.

Este registro es requerido para **validación estructural, evaluación energética (AEP), chequeo de límites de diseño** y para la preparación de documentación técnica del piloto en conformidad con IEC 61400-2 e IEC 61400-12-1.
""")

# =========================================================
# Escenarios de diseño y comparador
# =========================================================
st.subheader("🧬 Escenarios de diseño y comparación")

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
                "TSR objetivo": tsr,
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

            # KPIs energéticos
            "AEP_kWh": float(AEP_kWh),
            "CF": float(CF),
            "P_nom_kW": float(P_nom_kW),

            # KPIs de esfuerzo y márgenes (desde módulo de alertas)
            "max_T_gen": float(max_T_gen),
            "max_T_rotor": float(max_T_rotor),
            "max_I_est": float(max_I_est),
            "margen_Tgen_nom": float(margen_Tgen_nom),
            "margen_Trot_iec": float(margen_Trot_iec),
            "margen_I": float(margen_I),
            "margen_P": float(margen_P),
        }

        st.session_state["escenarios"].append(escenario)
        st.success(f"Escenario '{nombre_esc}' guardado en memoria de la sesión.")

# Mostrar listado resumen de escenarios guardados
if st.session_state["escenarios"]:
    st.markdown("#### Escenarios guardados en sesión")
    for i, esc in enumerate(st.session_state["escenarios"], start=1):
        st.markdown(
            f"- **{i}. {esc['nombre']}** "
            f"({esc['gen_label']}, G={esc['inputs']['G']:.2f}) – "
            f"P_nom = {esc['P_nom_kW']:.1f} kW, "
            f"AEP = {esc['AEP_kWh']:,.0f} kWh/año, "
            f"CF = {esc['CF']*100:.1f} %, "
            f"margen T_gen = {esc['margen_Tgen_nom']*100:.1f} %"
        )

# =========================================================
# Comparador A vs B
# =========================================================
if len(st.session_state["escenarios"]) < 2:
    st.info("Guarda al menos **dos escenarios** para habilitar el comparador A vs B.")
else:
    st.markdown("### ⚖️ Comparar dos escenarios")

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
        st.markdown("#### Márgenes de diseño (par, corriente, potencia)")

        colM1, colM2, colM3 = st.columns(3)
        colM1.metric(
            f"Margen T_gen {escA_name}",
            f"{escA['margen_Tgen_nom']*100:.1f} %",
            help="(T_nom - T_max)/T_nom – A"
        )
        colM2.metric(
            f"Margen T_gen {escB_name}",
            f"{escB['margen_Tgen_nom']*100:.1f} %",
            help="(T_nom - T_max)/T_nom – B"
        )
        colM3.metric(
            "Δ margen T_gen (B - A)",
            f"{(escB['margen_Tgen_nom']-escA['margen_Tgen_nom'])*100:.1f} pts",
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
        st.markdown("#### Curva de potencia eléctrica P_out(kW) vs viento")

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
        st.markdown("#### Cp_el_equiv (eficiencia global viento → eléctrica)")

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
        st.markdown("#### Torque en rotor (N·m) – impacto estructural")

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
        st.markdown("#### Torque en generador (N·m) – esfuerzo en el eje rápido")

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
        st.markdown("#### Corriente estimada en generador (A)")

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
          El comparador permite evaluar compromisos entre escenarios:
          </p>
          <ul>
            <li>Si <strong>{escB_name}</strong> entrega mayor AEP y CF, pero también incrementa el 
            <em>torque máximo</em> del rotor o del generador, puede requerir una estructura 
            y un tren de potencia más robustos.</li>
            <li>Las diferencias en <strong>Cp_el_equiv</strong> muestran si la mejora viene de la aerodinámica 
            y del tren de potencia, o solo de subir P_nominal.</li>
            <li>Las curvas de <strong>I_est(A)</strong> permiten ver en qué rango de vientos se 
            tensionan más las corrientes y si alguno de los escenarios se acerca demasiado a I_nom.</li>
            <li>Comparar <strong>P_out(kW)</strong> vs viento permite ver en qué rango de velocidades
            realmente se gana energía entre configuraciones (helicoidal vs no helicoidal, G distinta, 
            generador distinto, etc.).</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)



# =========================================================
# Descargar reporte técnico (PDF)
# =========================================================
st.subheader("📄 Descargar reporte técnico (PDF)")

kpi_summary = (
    f"Geometría evaluada: D = {D:.1f} m, H = {H:.1f} m, N = {N} palas. "
    f"TSR objetivo λ = {tsr:.2f}, solidez σ_int = {sig_int:.2f} (σ_conv ≈ {sig_conv:.2f}). "
    f"Potencia nominal configurada: {P_nom_kW:.1f} kW; "
    f"relación de transmisión G = {G:.2f}; "
    f"η_mec ≈ {eta_mec:.3f}, η_elec ≈ {eta_elec:.3f}."
)

# --- Elegir qué figura de potencia mandar al PDF (según modo seleccionado) ---
if dominio_pot == "Potencias vs viento (recomendada)":
    fig_pot = figP
else:
    fig_pot = figG

# --- Diccionario de figuras para el reporte ---
figs_report = {
    "rpm rotor / generador vs velocidad de viento": fig_r,
    "Curva de potencia (según vista seleccionada)": fig_pot,
    "Cp equivalente por etapa": fig_cp_eq,
    "Pérdidas por etapa": fig_loss,
    "Par en rotor / generador": figT,
    "Corriente estimada vs velocidad de viento": figI,
    "Frecuencias 1P / 3P del rotor": figF,
    "Curva Cp(λ) – promedio y componentes": fig_cp,
}


# -------------------------------------------------------
# Construcción diccionario de figuras
# -------------------------------------------------------
if use_noise:
    figs_report["Ruido estimado vs velocidad de viento"] = figNoise

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

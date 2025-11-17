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


def kpi_card(title: str, value: str, sub: str = ""):
    html = f"""
    <div class="kpi-card kpi-container">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

st.markdown("""
<style>
h2 {
    margin-top: -25px !important;
}
</style>
""", unsafe_allow_html=True)

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


# =========================================================
# Modelo Cp(λ) con efectos de perfil de pala
# =========================================================
def build_cp_params(
    lam_opt_base=2.6,
    cmax_base=0.33,
    shape=1.0,
    sigma=0.24,
    helical=True,
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
    - Helicoidal, end-plates, trips, struts perfilados
    - Perfil de pala: espesor relativo, simetría, ángulo de calaje
    - Efectos upwind / downwind (dynamic stall lumped)
    """
    lam_opt = lam_opt_base
    cmax    = cmax_base

    # 1) Solidez: más σ → Cp↑ pero λ_opt↓
    lam_opt -= 0.30 * (sigma - 0.20)
    cmax    += 0.05 * (sigma - 0.20)

    # 2) Configuración global del rotor
    if helical:
        cmax    += 0.03
        lam_opt += 0.10
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
        f_up   *= 1.03
        f_down *= 1.05

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
# Curvas del generador GDG-1100 (por defecto)
# =========================================================
GDG_POWER_TABLE = pd.DataFrame({
    "rpm":      [  0,  24,  48,  72,  96, 120, 144, 168, 192, 216, 240, 264],
    "P_kW":     [  0,   2,   3,   7,  12,  19,  28,  38,  50,  64,  80,  97],
})
GDG_VOLT_TABLE = pd.DataFrame({
    "rpm":      [  0,  24,  48,  72,  96, 120, 144, 168, 192, 216, 240, 264],
    "V_LL":     [  0,  40,  80, 120, 160, 200, 240, 280, 320, 360, 400, 440],
})

GDG_RATED_RPM   = 240.0
GDG_RATED_PkW   = 80.0
GDG_RATED_VLL   = 400.0
GDG_RATED_I     = 115.0
GDG_RATED_T_Nm  = 3460.0
GDG_POLES       = 48
GDG_OMEGA_RATED = 2 * pi * GDG_RATED_RPM / 60.0
GDG_KE_DEFAULT  = GDG_RATED_VLL / GDG_OMEGA_RATED
GDG_KT_DEFAULT  = GDG_RATED_T_Nm / GDG_RATED_I


def interp_curve(x, x_tab, y_tab):
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

    # Operación / Control
    with st.expander("Operación / Control", expanded=False):
        tsr = st.number_input("TSR objetivo (λ)", min_value=1.6, value=2.6, step=0.1)
        rho = st.number_input("Densidad aire ρ [kg/m³]", min_value=1.0, value=1.225, step=0.025)
        mu  = st.number_input(
            "Viscosidad dinámica μ [Pa·s]",
            min_value=1.0e-5, max_value=3.0e-5,
            value=1.8e-5, step=0.1e-5, format="%.6f"
        )
        v_cut_in  = st.number_input("v_cut-in [m/s]",  min_value=0.5, value=3.0, step=0.5)
        v_rated   = st.number_input("v_rated [m/s]",   min_value=v_cut_in + 0.5, value=10.0, step=0.5)
        v_cut_out = st.number_input("v_cut-out [m/s]", min_value=v_rated + 0.5, value=25.0, step=0.5)

    # Tweaks aerodinámicos
    with st.expander("Tweaks aerodinámicos", expanded=False):
        helical     = st.checkbox("Helicoidal 60–90°", True)
        endplates   = st.checkbox("End-plates / winglets", True)
        trips       = st.checkbox("Trips / micro-tabs", True)
        struts_perf = st.checkbox("Struts perfilados (0012)", True)

    # Perfil de pala / masa
    with st.expander("Perfil de pala / masa", expanded=False):
        airfoil_name = st.text_input("Perfil (ej: NACA 0018)", "NACA 0018")
        tipo_perfil  = st.selectbox("Tipo de perfil", ["Simétrico", "Asimétrico"])
        is_symmetric = (tipo_perfil == "Simétrico")
        t_rel = st.number_input("Espesor relativo e/c [%]", min_value=8.0, max_value=40.0, value=18.0, step=1.0)
        pitch_deg = st.number_input("Ángulo de calaje (pitch) [°]", min_value=-10.0, max_value=10.0, value=0.0, step=0.5)
        m_blade = st.number_input("Masa por pala [kg]", min_value=10.0, value=350.0, step=10.0)
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
        v_max  = st.number_input("v máx [m/s]", min_value=v_min+0.5, value=15.0, step=0.5)

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

    # Tren de potencia / Generador
    with st.expander("Tren de potencia / Generador", expanded=False):
        auto_G = st.checkbox("Calcular G con rpm rated", True)
        rpm_rotor_rated = st.number_input("rpm rotor rated", min_value=10.0, value=35.0, step=1.0)
        rpm_gen_rated   = st.number_input("rpm gen rated",   min_value=100.0, value=240.0, step=10.0)
        if auto_G:
            G = rpm_gen_rated / rpm_rotor_rated
            st.write(f"**G (calc)** = {G:.2f}")
        else:
            G = st.number_input("Relación G = rpm_gen/rpm_rotor", min_value=1.0, value=6.85, step=0.05)

        eta_bear = st.number_input("η rodamientos", min_value=0.90, value=0.98, step=0.005)
        eta_gear = st.number_input("η caja",       min_value=0.85, value=0.96, step=0.005)

        poles_total   = st.number_input("N° de polos (total)", min_value=4, value=GDG_POLES, step=2)
        eta_gen_max   = st.number_input("η_gen máx (tope)", min_value=0.80, value=0.93, step=0.005)
        Ke_vsr_default= st.number_input("Ke [V·s/rad] (≈V_LL/ω)", min_value=1.0, value=float(GDG_KE_DEFAULT), step=0.1)
        Kt_nm_per_A   = st.number_input("Kt [N·m/A]", min_value=1.0, value=float(GDG_KT_DEFAULT), step=0.1)

        st.caption("Puedes subir una curva alternativa del generador (cols: rpm, P_kW, V_LL).")
        gen_csv = st.file_uploader("CSV rendimiento generador", type=["csv"])
        eta_elec = st.number_input("η electrónica (rect+inv)", min_value=0.90, value=0.975, step=0.005)

        P_nom_kW  = st.number_input("P_nom [kW]", min_value=5.0, value=80.0, step=5.0)
        T_gen_max = st.number_input("T_gen máx [N·m] (opcional)", min_value=0.0, value=float(GDG_RATED_T_Nm), step=100.0)

    # Weibull (opcional)
    with st.expander("Weibull (opcional)", expanded=False):
        use_weibull = st.checkbox("Calcular AEP/FP con Weibull", False)
        k_w = st.number_input("k (forma)",  min_value=1.0, value=2.0, step=0.1)
        c_w = st.number_input("c (escala) [m/s]", min_value=2.0, value=8.0, step=0.5)

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
    endplates=endplates,
    trips=trips,
    struts_perf=struts_perf,
    airfoil_thickness=t_rel,
    symmetric=is_symmetric,
    pitch_deg=pitch_deg,
)

# Grid de vientos
v_grid = np.arange(v_min, v_max + 1e-9, 0.5 if v_max - v_min > 1 else 0.1)
# advertencia si v_max < v_rated
if v_max < v_rated:
    st.warning("⚠️ v_max es menor que v_rated; la región nominal no se ve completa en los gráficos.")

# Ley de operación por regiones (TSR constante solo en región 2)
rpm_tsr = rpm_from_tsr(v_grid, D, tsr)
rpm_rotor = np.zeros_like(v_grid)

mask_reg2 = (v_grid >= v_cut_in) & (v_grid <= v_rated)
rpm_rotor[mask_reg2] = rpm_tsr[mask_reg2]

rpm_rated_val = rpm_from_tsr(v_rated, D, tsr)
mask_reg3 = (v_grid > v_rated) & (v_grid <= v_cut_out)
rpm_rotor[mask_reg3] = rpm_rated_val

# v < cut-in o v > cut-out → rpm_rotor = 0

# Chequeo de consistencia entre rpm_rated (control) y rpm por TSR
if abs(rpm_rotor_rated - rpm_rated_val) > 5:
    st.warning(
        f"⚠️ rpm_rotor_rated ({rpm_rotor_rated:.1f} rpm) difiere de la rpm por TSR @ v_rated ({rpm_rated_val:.1f} rpm). "
        "Revisa consistencia entre el diseño aerodinámico y el control/MPPT."
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

# Curvas reales del GDG-1100 (o CSV alternativo)
if gen_csv is not None:
    df_gen = pd.read_csv(gen_csv)
    if not {"rpm","P_kW","V_LL"}.issubset(df_gen.columns):
        st.error("El CSV debe tener columnas: rpm, P_kW, V_LL")
        st.stop()
    tab_power = df_gen[["rpm","P_kW"]].sort_values("rpm").reset_index(drop=True)
    tab_volt  = df_gen[["rpm","V_LL"]].sort_values("rpm").reset_index(drop=True)
else:
    tab_power = GDG_POWER_TABLE.copy()
    tab_volt  = GDG_VOLT_TABLE.copy()

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

tab_rotor, tab_tren, tab_pala = st.tabs(
    ["Rotor & aerodinámica", "Tren de potencia", "Pala & cargas inerciales"]
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
        kpi_card("F centrífuga/pala @ rpm_rated", f"{F_centripetal_per_blade/1000:.1f} kN", "Esfuerzo en uniones y raíces de pala")

    p7, p8 = st.columns(2)
    with p7:
        kpi_card("Re @ v = 8 m/s", f"{Re_8:,.0f}", "Régimen aerodinámico de diseño (mid-span)")
    with p8:
        kpi_card("Re @ v_max", f"{Re_max:,.0f}", "Re límite operativo de la pala")

    st.caption(
        "Las propiedades de la pala permiten evaluar esfuerzos en uniones, ejes y rodamientos, "
        "además de la respuesta dinámica del rotor. Re indica el régimen aerodinámico del perfil."
    )

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
    box-shadow: 0 3px 10px rgba(15,23,42,0.45);
}

/* Centro el bloque horizontal del radio */
div[data-testid="stHorizontalBlock"] {
    justify-content: center !important;
}

/* ===== ESTILO TABLA (st.dataframe) ===== */
[data-testid="stDataFrame"] {
    border-radius: 14px;
    border: 1px solid rgba(148,163,184,0.6);
    box-shadow: 0 14px 30px rgba(15,23,42,0.55);
    overflow: hidden;
    background: #020617;
}

/* Cabecera */
[data-testid="stDataFrame"] thead tr th {
    background: #020617;
    color: #e5e7eb;
    font-weight: 600;
    font-size: 0.85rem;
    border-bottom: 1px solid #1f2937;
}

/* Filas */
[data-testid="stDataFrame"] tbody tr td {
    font-size: 0.85rem;
    color: #e5e7eb;
    border-bottom: 1px solid rgba(15,23,42,0.85);
}

[data-testid="stDataFrame"] tbody tr:nth-child(even) td {
    background: #02091b;
}

[data-testid="stDataFrame"] tbody tr:nth-child(odd) td {
    background: #020617;
}

/* Hover fila */
[data-testid="stDataFrame"] tbody tr:hover td {
    background: #020f2e;
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
st.dataframe(df_view, use_container_width=True)

st.download_button(
    f"📥 Descargar CSV – vista: {mod_sel}",
    data=df_view.to_csv(index=False).encode("utf-8"),
    file_name=f"vawt_resultados_{mod_sel.replace(' ', '_')}.csv",
    mime="text/csv"
)


# =========================================================
# Gráficos + comentarios técnicos
# =========================================================
c1, c2 = st.columns(2)
with c1:
    st.subheader("⚙️ rpm rotor / rpm generador vs v")
    fig_r = px.line(df, x="v (m/s)", y=["rpm_rotor","rpm_gen"], markers=True)
    fig_r.update_layout(xaxis_title="v (m/s)", yaxis_title="rpm")
    st.plotly_chart(fig_r, use_container_width=True)
    st.markdown("""
    <div class="comment-box">
      <div class="comment-title">🔍 Interpretación técnica</div>
      <p>
      Este gráfico muestra cómo crecen las rpm del rotor y del generador con el viento según la ley de control por regiones.
      Permite verificar que el rotor opera a TSR casi constante entre <em>v_cut-in</em> y <em>v_rated</em>, y que la relación de transmisión <strong>G</strong>
      lleva al generador a su zona de rpm nominal sin sobrepasarla, evitando sobrevelocidades mecánicas y eléctricas.
      </p>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.subheader("🚀 λ_efectiva, U_tip y Frecuencia eléctrica")
    fig_u = px.line(df, x="v (m/s)", y=["λ_efectiva","U_tip (m/s)","f_e (Hz)"], markers=True)
    fig_u.update_layout(xaxis_title="v (m/s)", yaxis_title="λ / U_tip [m/s] / f_e [Hz]")
    st.plotly_chart(fig_u, use_container_width=True)
    st.markdown("""
    <div class="comment-box">
      <div class="comment-title">🔍 Interpretación técnica</div>
      <p>
      Aquí se observa cómo varía el TSR efectivo, la velocidad de punta de pala y la frecuencia eléctrica con el viento.
      Este gráfico permite comprobar si el control mantiene <strong>λ</strong> cercano a <strong>λ<sub>opt</sub></strong> en región 2 y si
      <strong>U_tip</strong> y <strong>f<sub>e</sub></strong> se mantienen dentro de rangos aceptables de ruido, fatiga y compatibilidad
      con la electrónica de potencia.
      </p>
    </div>
    """, unsafe_allow_html=True)

st.subheader("🔋 Potencias: aero, mecánica, curva generador y salida (con límites)")
figP = px.line(
    df,
    x="v (m/s)",
    y=["P_aero (kW)", "P_mec_gen (kW)", "P_gen_curve (kW)", "P_out (clip) kW"],
    markers=True
)
figP.update_layout(xaxis_title="v (m/s)", yaxis_title="Potencia [kW]")
st.plotly_chart(figP, use_container_width=True)
st.markdown("""
<div class="comment-box">
  <div class="comment-title">🔍 Interpretación técnica</div>
  <p>
  La curva compara la potencia disponible en el viento (aero), la que llega al eje del generador (mecánica),
  la definida por la curva nominal del generador y la potencia eléctrica final con <em>clipping</em>.
  Permite identificar en qué rango de vientos domina la aerodinámica, las pérdidas mecánicas, las limitaciones del generador
  o la potencia nominal de la máquina, orientando decisiones de redimensionamiento y control.
  </p>
</div>
""", unsafe_allow_html=True)

st.subheader("📉 Cp equivalente por etapa")
fig_cp_eq = px.line(
    df,
    x="v (m/s)",
    y=["Cp_aero_equiv", "Cp_shaft_equiv", "Cp_el_equiv"],
    markers=True
)
fig_cp_eq.update_layout(xaxis_title="v (m/s)", yaxis_title="Cp equivalente")
st.plotly_chart(fig_cp_eq, use_container_width=True)
st.markdown("""
<div class="comment-box">
  <div class="comment-title">🔍 Interpretación técnica</div>
  <p>
  Este gráfico traduce las potencias de cada etapa a un <strong>Cp equivalente</strong> (aerodinámico, en eje y eléctrico).
  Es útil para visualizar cuánta eficiencia se pierde entre rotor → tren mecánico → generador → electrónica,
  y determinar si el diseño está limitado principalmente por la aerodinámica o por la integración electro-mecánica.
  </p>
</div>
""", unsafe_allow_html=True)

# Pérdidas por etapa
st.subheader("🔎 Pérdidas por etapa (mecánica, generador, electrónica, clipping)")

v_axis   = df["v (m/s)"].values
P_aero   = df["P_aero (kW)"].values
P_mec    = df["P_mec_gen (kW)"].values
P_el_ac_kW  = df["P_el (kW)"].values
P_out    = df["P_out (clip) kW"].values

P_el_before = np.divide(
    P_el_ac_kW,
    max(eta_elec, 1e-9),
    out=np.zeros_like(P_el_ac_kW),
    where=(eta_elec > 0)
)

loss_mech = np.maximum(P_aero - P_mec, 0.0)
loss_gen  = np.maximum(P_mec  - P_el_before, 0.0)
loss_elec = np.maximum(P_el_before - P_el_ac_kW, 0.0)
loss_clip = np.maximum(P_el_ac_kW - P_out, 0.0)

losses_df = pd.DataFrame({
    "v (m/s)": v_axis,
    "Pérdida mecánica (rodamientos+caja)": np.round(loss_mech, 2),
    "Pérdida generador (cobre/hierro)":     np.round(loss_gen, 2),
    "Pérdida electrónica (rect+inv)":       np.round(loss_elec, 2),
    "Pérdida por clipping (nom/par)":       np.round(loss_clip, 2),
})

loss_cols = [
    "Pérdida mecánica (rodamientos+caja)",
    "Pérdida generador (cobre/hierro)",
    "Pérdida electrónica (rect+inv)",
    "Pérdida por clipping (nom/par)",
]
figL = px.area(
    losses_df,
    x="v (m/s)",
    y=loss_cols,
    labels={"value": "Pérdidas [kW]", "variable": "Etapa"},
)
figL.update_layout(
    yaxis_title="Pérdidas [kW]",
    legend_title="Etapa"
)
st.plotly_chart(figL, use_container_width=True)
st.markdown("""
<div class="comment-box">
  <div class="comment-title">🔍 Interpretación técnica</div>
  <p>
  El área apilada muestra cuánto se pierde en cada etapa del sistema (rodamientos+caja, generador, electrónica
  y <em>clipping</em> por nominal/par) en función del viento.
  Este gráfico sirve para priorizar dónde conviene actuar: mejorar el tren mecánico, optimizar el diseño del generador,
  revisar la electrónica de potencia o ajustar la potencia nominal y la estrategia de control.
  </p>
</div>
""", unsafe_allow_html=True)

st.subheader("🧲 Par (rotor y generador) y Corriente estimada")
figT = px.line(df, x="v (m/s)", y=["T_rotor (N·m)", "T_gen (N·m)"], markers=True)
figT.update_layout(xaxis_title="v (m/s)", yaxis_title="Par [N·m]")
st.plotly_chart(figT, use_container_width=True)
st.markdown("""
<div class="comment-box">
  <div class="comment-title">🔍 Interpretación técnica (Par)</div>
  <p>
  Este gráfico muestra el par que ve el rotor y el generador según el viento.
  Es clave para revisar el dimensionamiento de ejes, rodamientos, caja multiplicadora y el límite <strong>T_gen_max</strong>,
  además de comprobar que la estrategia de control no lleve al generador a zonas de sobrepar crítico.
  </p>
</div>
""", unsafe_allow_html=True)

figI = px.line(df, x="v (m/s)", y="I_est (A)", markers=True)
figI.update_layout(xaxis_title="v (m/s)", yaxis_title="Corriente estimada [A]")
st.plotly_chart(figI, use_container_width=True)
st.markdown("""
<div class="comment-box">
  <div class="comment-title">🔍 Interpretación técnica (Corriente)</div>
  <p>
  La curva de corriente estimada en función del viento permite dimensionar cables, protecciones e inversores,
  y verificar que, bajo la ley de control definida, la máquina no supera las corrientes nominales de su equipamiento eléctrico
  en el rango de operación esperado.
  </p>
</div>
""", unsafe_allow_html=True)

st.subheader("🎯 Frecuencias 1P / 3P del rotor")
figF = px.line(df, x="v (m/s)", y=["f_1P (Hz)", "f_3P (Hz)"], markers=True)
figF.update_layout(xaxis_title="v (m/s)", yaxis_title="Frecuencia [Hz]")
st.plotly_chart(figF, use_container_width=True)
st.markdown("""
<div class="comment-box">
  <div class="comment-title">🔍 Interpretación técnica</div>
  <p>
  Las frecuencias 1P y 3P están asociadas al paso de palas y a las cargas periódicas principales del rotor.
  Este gráfico se utiliza para comparar estas frecuencias con los modos propios de torre, cimentación y estructura
  y así evitar configuraciones de resonancia en el diseño mecánico y estructural del sistema.
  </p>
</div>
""", unsafe_allow_html=True)

# Eficiencias por etapa
st.subheader("📈 Eficiencias: mecánica, generador y global")

eta_mec_pct   = 100 * np.divide(P_mec,      P_aero, out=np.zeros_like(P_aero), where=(P_aero>0))
eta_gen_pct   = 100 * np.divide(P_el_before,P_mec,  out=np.zeros_like(P_mec),  where=(P_mec>0))
eta_tot_pct   = 100 * np.divide(P_out,      P_aero, out=np.zeros_like(P_aero), where=(P_aero>0))

eff_df = pd.DataFrame({
    "v (m/s)": v_axis,
    "η_mec [%]": np.round(eta_mec_pct, 1),
    "η_gen [%]": np.round(eta_gen_pct, 1),
    "η_total [%]": np.round(eta_tot_pct, 1),
})
figE = px.line(
    eff_df, x="v (m/s)", y=["η_mec [%]", "η_gen [%]", "η_total [%]"], markers=True
)
figE.update_layout(yaxis_title="Eficiencia [%]", legend_title="Etapa")
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
</div>
""", unsafe_allow_html=True)

st.caption(
    "η_total = P_out / P_aero. Si la curva de 'Pérdida por clipping' domina desde cierta v, "
    "estás en región de potencia constante; considera redimensionar G/TSR o estrategia de control."
)

# Curva Cp(λ)
st.subheader("🧩 Cp(λ) – Promedio, upwind y downwind")
df_cp = cp_curve_for_plot(cp_params)
fig_cp = px.line(df_cp, x="λ", y=["Cp_prom", "Cp_upwind", "Cp_downwind"], markers=False)
fig_cp.add_vline(x=tsr, line_dash="dot", line_color="orange", annotation_text="TSR objetivo")
fig_cp.add_vline(x=cp_params["lam_opt"], line_dash="dash", line_color="green", annotation_text="λ_opt")
fig_cp.update_layout(xaxis_title="λ", yaxis_title="Cp", legend_title="Componente")
st.plotly_chart(fig_cp, use_container_width=True)
st.markdown("""
<div class="comment-box">
  <div class="comment-title">🔍 Interpretación técnica</div>
  <p>
  La curva <strong>Cp(λ)</strong> resume el rendimiento aerodinámico teórico del rotor, separando la contribución
  <em>upwind</em> y <em>downwind</em>. La comparación entre <strong>λ_opt</strong> y el TSR objetivo
  ayuda a ajustar el control y la geometría (solidez, helicoidal, perfil) para operar lo más cerca posible del máximo Cp
  en el rango de vientos de interés del proyecto.
  </p>
</div>
""", unsafe_allow_html=True)

# Ruido (si aplica)
if use_noise:
    st.subheader("🔈 Ruido estimado vs velocidad de viento")
    figNoise = px.line(
        df,
        x="v (m/s)",
        y=["Lw (dB)", "Lp_obs (dB)"],
        markers=True
    )
    figNoise.update_layout(
        xaxis_title="v (m/s)",
        yaxis_title="Nivel sonoro [dB]",
        legend_title="Magnitud"
    )
    st.plotly_chart(figNoise, use_container_width=True)

    st.markdown(f"""
    <div class="comment-box">
      <div class="comment-title">🔍 Interpretación técnica (ruido)</div>
      <p>
      El modelo de ruido usa como referencia un nivel <strong>Lw_ref = {Lw_ref_dB:.0f} dB</strong> a 
      <em>v_rated</em> y escala el nivel con una ley de potencia de la velocidad de punta
      (<code>U_tip^n</code>, con n={n_noise:.1f}). A partir de Lw se estima el nivel de presión
      sonora <strong>Lp</strong> percibido a una distancia de <strong>{r_obs:.0f} m</strong>,
      asumiendo propagación en campo libre.
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
        name="Weibull f(v)"
    ),
    secondary_y=False,
)

# Curva de potencia
figW.add_trace(
    go.Scatter(
        x=df_weib["v (m/s)"],
        y=df_weib["P_out (kW)"],
        mode="lines",
        name="P_out (kW)"
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
        line=dict(dash="dot")
    ),
    secondary_y=True,
)

figW.update_xaxes(title_text="Velocidad de viento v [m/s]")
figW.update_yaxes(
    title_text="f_W(v) [1/(m/s)]",
    secondary_y=False
)
figW.update_yaxes(
    title_text="Potencia / Contribución [kW]",
    secondary_y=True
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
    f"TSR objetivo λ = {tsr:.2f}, solidez σ_int = {sig_int:.2f} (σ_conv ≈ {sig_conv:.2f}). "
    f"Potencia nominal configurada: {P_nom_kW:.1f} kW; "
    f"relación de transmisión G = {G:.2f}; "
    f"η_mec ≈ {eta_mec:.3f}, η_elec ≈ {eta_elec:.3f}."
)

figs_report = {
    "rpm rotor / generador vs velocidad de viento": fig_r,
    "Potencias: aero, mecánica, generador y salida": figP,
    "Cp equivalente por etapa": fig_cp_eq,
    "Pérdidas por etapa": figL,
    "Par en rotor / generador": figT,
    "Corriente estimada vs velocidad de viento": figI,
    "Frecuencias 1P / 3P del rotor": figF,
    "Curva Cp(λ) – promedio y componentes": fig_cp,
}
if use_noise:
    figs_report["Ruido estimado vs velocidad de viento"] = figNoise

pdf_bytes = build_pdf_report(df_view, figs_report, kpi_summary)
st.download_button(
    "📥 Descargar reporte técnico (PDF)",
    data=pdf_bytes,
    file_name="reporte_tecnico_VAWT.pdf",
    mime="application/pdf"
)

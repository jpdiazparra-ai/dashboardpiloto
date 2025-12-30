# capex_piloto_80kw.py
# Dashboard CAPEX Piloto Eólico 80 kW (versión mejorada)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO
import plotly.io as pio
import re
import unicodedata
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib import colors
import math
import plotly.graph_objects as go


# =========================
# CONFIGURACIÓN GLOBAL
# =========================
st.set_page_config(
    page_title="CAPEX Piloto Eólico 80 kW",
    layout="wide"
)

CAT_COLOR_MAP = {
    "Desarrollo Tecnológico": "#4C956C",            # verde musgo
    "Componentes Mecánicos": "#5C6B73",             # acero opaco
    "Sistema Eléctrico y Control": "#C58940",       # ámbar oscuro
    "Obras Civiles": "#C75C5C",                     # ladrillo mate
    "Montaje y Logística": "#7F6A93",               # violeta apagado
    "Ensayos y Certificación": "#E3B23C",           # dorado mate
    "Contingencias y Administración": "#7A7E8C",    # gris grafito
}

PX_COLORS = [
    "#4C956C",
    "#5C6B73",
    "#C58940",
    "#C75C5C",
    "#7F6A93",
    "#E3B23C",
]
px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = PX_COLORS

CAPEX_CLP_DEFAULT = 480_000_000
CAPEX_CSV_URL_DEFAULT = (
    "https://docs.google.com/spreadsheets/d/e/"
    "2PACX-1vSlNd3zXc1zV6TUQHnhXlfZtv7QVOv0mBfR_HH69Ht-0qi2aDtCfw5ouLDGIoPH_knhSAtyT2DYE-Qo/"
    "pub?gid=467592026&single=true&output=csv"
)

# =========================
# FUNCIONES
# =========================
@st.cache_data(show_spinner=True)
def load_capex_data(url: str) -> pd.DataFrame:
    """
    Carga el CSV de CAPEX y lo normaliza.
    Soporta dos formatos:
    - Versión CON encabezados: 'ITEM', 'Categoría', 'Participación (%)', 'Monto USD', 'Bullet técnico',
      y opcionalmente 'Mes_inicio', 'Mes_termino', 'Dependencia'.
    - Versión SIN encabezados (formato antiguo): toma las primeras 5 columnas como Item/Categoria/Participacion/Monto/Bullet.
    """
    df_raw = pd.read_csv(url, dtype=str)
    # Normalizamos nombres de columnas para detección por nombre
    df_raw.columns = [str(c).strip() for c in df_raw.columns]

    has_named_header = set(["ITEM", "Categoría", "Participación (%)", "Monto USD"]).issubset(
        set(df_raw.columns)
    )

    if has_named_header:
        # --- Formato nuevo, con encabezados ---
        df = pd.DataFrame()
        df["Item"] = df_raw["ITEM"].astype(str).str.strip()
        df["Categoria"] = df_raw["Categoría"].astype(str).str.strip()
        df["Participacion_raw"] = df_raw["Participación (%)"]
        df["Monto_USD_raw"] = df_raw["Monto USD"]
        if "Bullet técnico" in df_raw.columns:
            df["Bullet"] = df_raw["Bullet técnico"].astype(str).str.strip()
        else:
            df["Bullet"] = ""

        # Columnas de calendario para la línea de tiempo (opcionales)
        if "Mes_inicio" in df_raw.columns:
            df["Mes_inicio"] = pd.to_numeric(df_raw["Mes_inicio"], errors="coerce")
        if "Mes_termino" in df_raw.columns:
            df["Mes_termino"] = pd.to_numeric(df_raw["Mes_termino"], errors="coerce")
        if "Dependencia" in df_raw.columns:
            df["Dependencia"] = df_raw["Dependencia"].astype(str).str.strip()

    else:
        # --- Formato antiguo (sin encabezados) ---
        df = df_raw.iloc[:, :5].copy()
        df.columns = ["Item", "Categoria", "Participacion_raw", "Monto_USD_raw", "Bullet"]

    # Limpieza de texto base
    for col in ["Item", "Categoria", "Bullet"]:
        df[col] = df[col].astype(str).str.strip()

    # Parseo de porcentaje
    def parse_pct(x: str) -> float:
        if pd.isna(x):
            return 0.0
        raw = str(x).strip()
        if not raw:
            return 0.0
        has_pct = "%" in raw
        normalized = raw.replace("%", "").replace(",", ".").replace(" ", "")
        try:
            val = float(normalized)
        except ValueError:
            return 0.0
        if has_pct or val >= 1.0:
            val /= 100.0
        return min(max(val, 0.0), 1.0)

    df["Participacion_pct"] = df["Participacion_raw"].apply(parse_pct)

    # Parseo de dinero en USD
    def parse_money(x: str) -> float:
        if pd.isna(x):
            return 0.0
        x = str(x).strip()
        x = x.replace(".", "").replace(" ", "")
        x = x.replace(",", ".")
        try:
            return float(x)
        except ValueError:
            return 0.0

    df["Monto_USD"] = df["Monto_USD_raw"].apply(parse_money)

    return df


def compute_capex_clp(df: pd.DataFrame, capex_total_clp: float):
    df = df.copy()
    total_usd = df["Monto_USD"].sum()
    tipo_cambio = capex_total_clp / total_usd if total_usd > 0 else np.nan
    df["Monto_CLP"] = df["Monto_USD"] * tipo_cambio
    return df, tipo_cambio, total_usd


def format_clp(x: float) -> str:
    return f"${x:,.0f}".replace(",", ".")


def format_usd(x: float) -> str:
    return f"US${x:,.0f}".replace(",", ".")


def parse_money_usd_robusto(x: str) -> float:
    """
    Convierte strings tipo:
      'US$9.090,91' -> 9090.91
      '9,090.91'    -> 9090.91
      '9090,91'     -> 9090.91
      'US$ 6.200'   -> 6200
    Regla: el separador decimal es el ÚLTIMO (',' o '.') que aparece.
    Todo lo demás se interpreta como separador de miles y se elimina.
    """
    if pd.isna(x):
        return 0.0
    s = str(x).strip()
    if not s:
        return 0.0

    s = s.replace("US$", "").replace("$", "").replace(" ", "")
    s = re.sub(r"[^0-9\-,\.]", "", s)

    if s in ("", "-", ".", ","):
        return 0.0

    last_comma = s.rfind(",")
    last_dot = s.rfind(".")

    if last_comma == -1 and last_dot == -1:
        try:
            return float(s)
        except ValueError:
            return 0.0

    if "," in s and "." not in s:
        parts = s.split(",")
        if len(parts) == 2 and len(parts[1]) == 3:
            s = s.replace(",", "")
        else:
            s = s.replace(",", ".")
    elif "." in s and "," not in s:
        parts = s.split(".")
        if len(parts) == 2 and len(parts[1]) == 3:
            s = s.replace(".", "")
    else:
        dec_sep = "," if last_comma > last_dot else "."
        if dec_sep == ",":
            s = s.replace(".", "")
            s = s.replace(",", ".")
        else:
            s = s.replace(",", "")

    try:
        return float(s)
    except ValueError:
        return 0.0


def build_item_color_map(item_to_category: dict) -> dict:
    """Asigna a cada ítem el color de su categoría usando el orden corporativo."""
    ordered_categories = [
        "Desarrollo Tecnológico",
        "Componentes Mecánicos",
        "Sistema Eléctrico y Control",
        "Obras Civiles",
        "Montaje y Logística",
        "Ensayos y Certificación",
        "Contingencias y Administración",
    ]
    mapping = {}
    palette_cycle = list(ordered_categories)
    for item, category in item_to_category.items():
        cat_norm = str(category).strip()
        if cat_norm in CAT_COLOR_MAP:
            color = CAT_COLOR_MAP[cat_norm]
        else:
            # color no definido: usar la siguiente categoría de referencia
            idx = len(mapping) % len(palette_cycle)
            ref_cat = palette_cycle[idx]
            color = CAT_COLOR_MAP[ref_cat]
        mapping[item] = color
    return mapping


def render_category_palette():
    """Muestra una tira de categorías con sus colores corporativos."""
    palette_css = """
    <style>
    .cat-chip-container {
        display: flex;
        flex-wrap: wrap;
        gap: 0.35rem;
        margin-bottom: 0.5rem;
    }
    .cat-chip {
        border-radius: 999px;
        padding: 0.2rem 0.75rem;
        font-size: 0.78rem;
        font-weight: 600;
        color: #fff;
        white-space: nowrap;
    }
    </style>
    """
    chips = "".join(
        f'<span class="cat-chip" style="background:{color};">{cat}</span>'
        for cat, color in CAT_COLOR_MAP.items()
    )
    st.markdown(palette_css + f'<div class="cat-chip-container">{chips}</div>', unsafe_allow_html=True)


def render_pagos_hitos(capex_url: str, fx_used: float, pagos_scale: float):
    st.markdown("---")
    st.subheader("Pagos por hitos (Anticipo / FAT / SAT)")

    try:
        df_raw_pagos = pd.read_csv(capex_url, dtype=str)
        df_raw_pagos.columns = [str(c).strip() for c in df_raw_pagos.columns]
        col_map = {}
        if "ITEM" in df_raw_pagos.columns and "Item" not in df_raw_pagos.columns:
            col_map["ITEM"] = "Item"
        if "Categoría" in df_raw_pagos.columns and "Categoria" not in df_raw_pagos.columns:
            col_map["Categoría"] = "Categoria"
        if col_map:
            df_raw_pagos = df_raw_pagos.rename(columns=col_map)

        required_cols = [
            "Mes_Anticipo",
            "Pago_USD_Anticipo",
            "Mes_Entrega_FAT",
            "Pago_USD_Entrega",
            "Mes_SAT",
            "Pago_USD_SAT",
        ]
        missing_cols = [c for c in required_cols if c not in df_raw_pagos.columns]
        if missing_cols:
            st.error(f"Faltan columnas en la hoja de pagos: {missing_cols}")
            return

        def _sum_by_month(mes_col: str, pago_col: str, out_col: str) -> pd.DataFrame:
            df_tmp = df_raw_pagos[[mes_col, pago_col]].copy()
            df_tmp["Mes_i"] = pd.to_numeric(df_tmp[mes_col], errors="coerce")
            df_tmp["Pago_f"] = df_tmp[pago_col].apply(parse_money_usd_robusto) * pagos_scale
            out = (
                df_tmp.dropna(subset=["Mes_i"])
                .groupby("Mes_i", as_index=False)
                .agg(**{out_col: ("Pago_f", "sum")})
                .rename(columns={"Mes_i": "Mes"})
            )
            return out

        df_meses = pd.DataFrame({"Mes": list(range(1, 16))})

        df_anticipo = _sum_by_month(
            "Mes_Anticipo",
            "Pago_USD_Anticipo",
            "Pago_USD_Anticipo",
        )
        df_entrega = _sum_by_month(
            "Mes_Entrega_FAT",
            "Pago_USD_Entrega",
            "Pago_USD_Entrega",
        )
        df_sat = _sum_by_month("Mes_SAT", "Pago_USD_SAT", "Pago_USD_SAT")

        df_consolidado = (
            df_meses.merge(df_anticipo, on="Mes", how="left")
            .merge(df_entrega, on="Mes", how="left")
            .merge(df_sat, on="Mes", how="left")
            .fillna(0.0)
        )
        df_consolidado["Total_USD"] = (
            df_consolidado["Pago_USD_Anticipo"]
            + df_consolidado["Pago_USD_Entrega"]
            + df_consolidado["Pago_USD_SAT"]
        )

        item_rows = []
        for _, row in df_raw_pagos.iterrows():
            item = str(row.get("Item", "")).strip() or "Sin ítem"
            categoria = str(row.get("Categoria", "")).strip() or "Sin categoría"
            mes_ant = pd.to_numeric(row.get("Mes_Anticipo"), errors="coerce")
            pago_ant = parse_money_usd_robusto(row.get("Pago_USD_Anticipo")) * pagos_scale
            mes_fat = pd.to_numeric(row.get("Mes_Entrega_FAT"), errors="coerce")
            pago_fat = parse_money_usd_robusto(row.get("Pago_USD_Entrega")) * pagos_scale
            mes_sat = pd.to_numeric(row.get("Mes_SAT"), errors="coerce")
            pago_sat = parse_money_usd_robusto(row.get("Pago_USD_SAT")) * pagos_scale

            if pd.notna(mes_ant):
                item_rows.append(
                    {
                        "Mes": int(mes_ant),
                        "Item": item,
                        "Categoria": categoria,
                        "Pago_USD": pago_ant,
                    }
                )
            if pd.notna(mes_fat):
                item_rows.append(
                    {
                        "Mes": int(mes_fat),
                        "Item": item,
                        "Categoria": categoria,
                        "Pago_USD": pago_fat,
                    }
                )
            if pd.notna(mes_sat):
                item_rows.append(
                    {
                        "Mes": int(mes_sat),
                        "Item": item,
                        "Categoria": categoria,
                        "Pago_USD": pago_sat,
                    }
                )

        df_item_periodo = pd.DataFrame(item_rows)
        if not df_item_periodo.empty:
            df_item_periodo = (
                df_item_periodo.groupby(["Mes", "Item", "Categoria"], as_index=False)
                .agg(Pago_USD=("Pago_USD", "sum"))
                .sort_values(["Mes", "Item"])
            )

        st.subheader("Hitos de pagos")
        unit_sel = st.selectbox(
            "Moneda/escala",
            ["USD (miles)", "CLP (millones)"],
            index=0,
            key="pay_currency_selector",
        )
        if unit_sel.startswith("USD"):
            scale_factor = 1.0 / 1_000.0
            axis_unit = "miles USD"
            line_label = "Acumulado (miles USD)"
            def fmt_bar_value(val: float) -> str:
                return f"{val:,.0f}k".replace(",", ".")
        else:
            scale_factor = fx_used / 1_000_000.0
            axis_unit = "MM CLP"
            line_label = "Acumulado (MM CLP)"
            def fmt_bar_value(val: float) -> str:
                return f"{val:.1f} MM"

        def scale_usd(series: pd.Series) -> pd.Series:
            return series * scale_factor

        view_sel = st.selectbox(
            "Selecciona vista",
            [
                "1) Inyección por hito (Anticipo/FAT/SAT)",
                "2) Inyección por ítem",
                "3) Total por período + categoría",
            ],
            index=0,
            key="pay_view_selector",
        )

        if view_sel.startswith("1"):
            df_flujo_plot = df_consolidado.copy()
            df_flujo_plot["Total_plot"] = scale_usd(df_flujo_plot["Total_USD"])
            df_flujo_plot["Acum_plot"] = df_flujo_plot["Total_plot"].cumsum()
            df_flujo_long = df_flujo_plot.melt(
                id_vars=["Mes"],
                value_vars=[
                    "Pago_USD_Anticipo",
                    "Pago_USD_Entrega",
                    "Pago_USD_SAT",
                ],
                var_name="Tipo",
                value_name="Pago_USD",
            )
            df_flujo_long["Pago_plot"] = scale_usd(df_flujo_long["Pago_USD"])
            df_flujo_long["Tipo"] = df_flujo_long["Tipo"].map(
                {
                    "Pago_USD_Anticipo": "Anticipo",
                    "Pago_USD_Entrega": "Entrega FAT",
                    "Pago_USD_SAT": "SAT",
                }
            )

            fig_iny = px.bar(
                df_flujo_long,
                x="Mes",
                y="Pago_plot",
                color="Tipo",
                labels={
                    "Mes": "Mes del proyecto",
                    "Pago_plot": f"Pago mensual ({axis_unit})",
                    "Tipo": "Hito de pago",
                },
                title="Inyección por hito (Anticipo/FAT/SAT)",
            )
            fig_iny.add_scatter(
                x=df_flujo_plot["Mes"],
                y=df_flujo_plot["Acum_plot"],
                mode="lines+markers",
                name=line_label,
                yaxis="y2",
                line=dict(color="#0f172a", width=2),
            )
            fig_iny.update_layout(
                barmode="stack",
                height=420,
                margin=dict(l=10, r=10, t=50, b=30),
                yaxis=dict(title=f"Pago mensual ({axis_unit})"),
                yaxis2=dict(
                    title=line_label,
                    overlaying="y",
                    side="right",
                    showgrid=False,
                ),
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.2,
                    xanchor="center",
                    x=0.5,
                ),
            )
            fig_iny.update_xaxes(dtick=1)
            for mes, total in zip(df_flujo_plot["Mes"], df_flujo_plot["Total_plot"]):
                fig_iny.add_annotation(
                    x=mes,
                    y=total * 1.05,
                    text=fmt_bar_value(total),
                    showarrow=False,
                    yanchor="bottom",
                    font=dict(color="#111827", size=11),
                )
            st.plotly_chart(fig_iny, use_container_width=True)

        elif view_sel.startswith("2"):
            if df_item_periodo.empty:
                st.info("No hay pagos disponibles para construir la inyección por ítem.")
            else:
                df_item_total = (
                    df_item_periodo.groupby("Mes", as_index=False)
                    .agg(Total_USD=("Pago_USD", "sum"))
                    .sort_values("Mes")
                )
                df_item_total["Total_plot"] = scale_usd(df_item_total["Total_USD"])
                df_item_total["Acum_plot"] = df_item_total["Total_plot"].cumsum()
                df_item_periodo_plot = df_item_periodo.copy()
                df_item_periodo_plot["Pago_plot"] = scale_usd(df_item_periodo_plot["Pago_USD"])
                fig_item_iny = px.bar(
                    df_item_periodo_plot,
                    x="Mes",
                    y="Pago_plot",
                    color="Item",
                    color_discrete_map=item_color_map,
                    labels={
                        "Mes": "Mes del proyecto",
                        "Pago_plot": f"Pago mensual ({axis_unit})",
                        "Item": "Ítem",
                    },
                    title="Inyección por ítem (Anticipo/FAT/SAT)",
                )
                fig_item_iny.add_scatter(
                    x=df_item_total["Mes"],
                    y=df_item_total["Acum_plot"],
                    mode="lines+markers",
                    name=line_label,
                    yaxis="y2",
                    line=dict(color="#0f172a", width=2),
                )
                fig_item_iny.update_layout(
                    barmode="stack",
                    height=420,
                    margin=dict(l=10, r=10, t=50, b=30),
                    yaxis=dict(title=f"Pago mensual ({axis_unit})"),
                    yaxis2=dict(
                        title=line_label,
                        overlaying="y",
                        side="right",
                        showgrid=False,
                    ),
                    legend=dict(
                        orientation="h",
                        yanchor="top",
                        y=-0.2,
                        xanchor="center",
                        x=0.5,
                    ),
                )
                fig_item_iny.update_xaxes(dtick=1)
                for mes, total in zip(df_item_total["Mes"], df_item_total["Total_plot"]):
                    fig_item_iny.add_annotation(
                        x=mes,
                        y=total * 1.05,
                        text=fmt_bar_value(total),
                        showarrow=False,
                        yanchor="bottom",
                        font=dict(color="#111827", size=11),
                    )
                st.plotly_chart(fig_item_iny, use_container_width=True)

        elif view_sel.startswith("3"):
            if df_item_periodo.empty:
                st.info("No hay pagos disponibles para construir el total por período.")
            else:
                df_cat_periodo = (
                    df_item_periodo.groupby(["Mes", "Categoria"], as_index=False)
                    .agg(Pago_USD=("Pago_USD", "sum"))
                )
                df_cat_periodo["Pago_plot"] = scale_usd(df_cat_periodo["Pago_USD"])
                df_total = (
                    df_cat_periodo.groupby("Mes", as_index=False)
                    .agg(Total_USD=("Pago_USD", "sum"))
                    .sort_values("Mes")
                )
                df_total["Total_plot"] = scale_usd(df_total["Total_USD"])
                fig_cat_total = px.bar(
                    df_cat_periodo,
                    x="Mes",
                    y="Pago_plot",
                    color="Categoria",
                    color_discrete_map=CAT_COLOR_MAP,
                    labels={
                        "Mes": "Mes del proyecto",
                        "Pago_plot": f"Pago mensual ({axis_unit})",
                        "Categoria": "Categoría",
                    },
                    title="Total por período (Anticipo/FAT/SAT) por categoría",
                )
                fig_cat_total.update_layout(
                    barmode="stack",
                    height=420,
                    margin=dict(l=10, r=10, t=50, b=30),
                    yaxis=dict(title=f"Pago mensual ({axis_unit})"),
                    legend=dict(
                        orientation="h",
                        yanchor="top",
                        y=-0.2,
                        xanchor="center",
                        x=0.5,
                    ),
                )
                fig_cat_total.update_xaxes(dtick=1)
                for mes, total in zip(df_total["Mes"], df_total["Total_plot"]):
                    fig_cat_total.add_annotation(
                        x=mes,
                        y=total * 1.05,
                        text=fmt_bar_value(total),
                        showarrow=False,
                        yanchor="bottom",
                        font=dict(color="#111827", size=11),
                    )
                st.plotly_chart(fig_cat_total, use_container_width=True)

    except Exception as e:
        st.error(f"No se pudo construir el análisis de pagos: {e}")

# =========================
# SIDEBAR
# =========================
st.sidebar.title("⚙️ Parámetros")

capex_url = st.sidebar.text_input(
    "URL CSV CAPEX",
    value=CAPEX_CSV_URL_DEFAULT,
    help="URL pública del CSV con la tabla de CAPEX."
)

capex_total_clp_input = st.sidebar.number_input(
    "CAPEX total (CLP)",
    min_value=1_000_000,
    value=CAPEX_CLP_DEFAULT,
    step=1_000_000,
    help="Inversión total del piloto en pesos chilenos."
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "El dashboard se alimenta directamente de Google Sheets y recalcula "
    "montos, tipo de cambio y gráficos en tiempo real."
)

# =========================
# DATOS
# =========================
df_capex_base = load_capex_data(capex_url)
df_capex, tipo_cambio_implicito, capex_total_usd_base = compute_capex_clp(
    df_capex_base,
    capex_total_clp_input,
)
fx_base = (
    CAPEX_CLP_DEFAULT / capex_total_usd_base
    if np.isfinite(capex_total_usd_base) and capex_total_usd_base > 0
    else tipo_cambio_implicito
)

manual_fx_enabled = st.sidebar.checkbox(
    "Usar tipo de cambio manual",
    value=False,
)
manual_fx_value = st.sidebar.number_input(
    "Tipo de cambio (CLP/US$)",
    min_value=100.0,
    max_value=5000.0,
    value=float(round(tipo_cambio_implicito)) if np.isfinite(tipo_cambio_implicito) else 900.0,
    step=10.0,
    help="Si está activo, ajusta los montos en USD y conversiones.",
    disabled=not manual_fx_enabled,
)

fx_used = (
    manual_fx_value
    if manual_fx_enabled and np.isfinite(manual_fx_value) and manual_fx_value > 0
    else fx_base
)

if np.isfinite(fx_used) and fx_used > 0:
    df_capex["Monto_USD"] = df_capex["Monto_CLP"] / fx_used
    capex_total_usd = df_capex["Monto_USD"].sum()
else:
    capex_total_usd = capex_total_usd_base

capex_total_clp = capex_total_clp_input
tipo_cambio_implicito = fx_used
pagos_scale = (
    capex_total_usd / capex_total_usd_base
    if np.isfinite(capex_total_usd_base) and capex_total_usd_base > 0
    else 1.0
)

if not np.isfinite(tipo_cambio_implicito) or tipo_cambio_implicito <= 0:
    st.error(
        "No se pudo calcular un tipo de cambio implícito confiable. "
        "Revisa que la hoja tenga montos en USD válidos."
    )
    st.stop()

if not 500 <= tipo_cambio_implicito <= 1200:
    st.warning(
        f"El tipo de cambio implícito ({tipo_cambio_implicito:,.0f} CLP/US$) "
        "está fuera del rango esperado para un proyecto piloto. "
        "Verifica los datos cargados."
    )

df_cat = (
    df_capex
    .groupby("Categoria", as_index=False)
    .agg(
        Monto_USD=("Monto_USD", "sum"),
        Monto_CLP=("Monto_CLP", "sum"),
        Participacion_sum=("Participacion_pct", "sum"),
        Items=("Item", "count"),
    )
    .sort_values("Monto_CLP", ascending=False)
    .reset_index(drop=True)
)

item_category_lookup = (
    df_capex[["Item", "Categoria"]]
    .drop_duplicates(subset=["Item"])
    .set_index("Item")["Categoria"]
    .to_dict()
)
item_color_map = build_item_color_map(item_category_lookup)

# columnas auxiliares para gráficos en MM CLP
df_cat["Monto_CLP_MM"] = df_cat["Monto_CLP"] / 1e6
df_capex["Monto_CLP_MM"] = df_capex["Monto_CLP"] / 1e6

# =========================
# HEADER
# =========================
st.title("📊 CAPEX Piloto Eólico 80 kW")
st.caption(
    "Panel interactivo para analizar la estructura de inversión del piloto de turbina eólica vertical híbrida. "
    "Diseñado para uso en directorio, comité técnico y seguimiento de proyecto."
)

total_items = len(df_capex)
total_categorias = df_cat["Categoria"].nunique()
cat_top = df_cat.iloc[0]["Categoria"] if total_categorias > 0 else "-"
cat_top_pct = df_cat.iloc[0]["Participacion_sum"] * 100 if total_categorias > 0 else 0

# =========================
# ESTILO KPI CARDS  (SIEMPRE ANTES DE USAR kpi_card)
# =========================
st.markdown(
    """
    <style>
    .kpi-row {
        margin-top: 0.5rem;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background: #F9FAFB;
        border-radius: 0.9rem;
        padding: 1.1rem 1.4rem 1.0rem 1.4rem;
        border: 1px solid #E5E7EB;
        box-shadow: 0 6px 14px rgba(15, 23, 42, 0.06);
    }
    .kpi-label {
        font-size: 0.80rem;
        font-weight: 600;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: .08em;
        margin-bottom: 0.15rem;
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #111827;
        line-height: 1.1;
        margin-bottom: 0.15rem;
        word-break: break-word;
    }
    .kpi-sub {
        font-size: 0.78rem;
        color: #9CA3AF;
        margin-top: 0.15rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def kpi_card(title: str, value: str, subtitle: str = ""):
    """Renderiza una tarjeta KPI con título, valor y subtítulo."""
    html = f"""
    <div class="kpi-card">
        <div class="kpi-label">{title}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{subtitle}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

# =========================
# KPI CARDS – DISEÑO PRO
# =========================
st.markdown('<div class="kpi-row"></div>', unsafe_allow_html=True)
k1, k2, k3, k4 = st.columns(4)

with k1:
    kpi_card(
        "CAPEX total (CLP)",
        format_clp(capex_total_clp),
        "Inversión piloto 80 kW, incluye I+D, componentes, montaje y contingencias."
    )

with k2:
    kpi_card(
        "CAPEX total (USD)",
        format_usd(capex_total_usd),
        f"Calculado con tipo de cambio implícito ≈ {tipo_cambio_implicito:,.0f} CLP/US$."
    )

with k3:
    kpi_card(
        "Tipo de cambio implícito",
        f"{fx_used:,.1f} CLP/US$",
        "Derivado de CAPEX CLP / suma de costos en USD del desglose."
    )

with k4:
    kpi_card(
        "Categoría de mayor peso",
        cat_top,
        f"Representa aproximadamente {cat_top_pct:.1f}% del CAPEX total."
    )


# =========================
# PREPARACIÓN PARA GRÁFICOS
# =========================
df_capex["Monto_CLP_MM"] = df_capex["Monto_CLP"] / 1e6
df_cat["Monto_CLP_MM"] = df_cat["Monto_CLP"] / 1e6

# =========================
# TABS PRINCIPALES
# =========================
tab_resumen, tab_categoria, tab_items, tab_explorador = st.tabs(
    ["📌 Resumen ejecutivo", "📂 Por categoría", "📄 Detalle de ítems", "🔍 Explorador interactivo"]
)

# -------------------------
# TAB RESUMEN
# -------------------------
with tab_resumen:
    st.subheader("Vista general del CAPEX")

    # ========================
    # 1) CAPEX por Ítem — barra única + monto total + %
    # ========================

    df_items_tot = (
        df_capex
        .groupby("Item", as_index=False)
        .agg(Total_CLP=("Monto_CLP", "sum"))
    )

    df_items_tot = df_items_tot.merge(
        pd.DataFrame(
            list(item_category_lookup.items()),
            columns=["Item", "Categoria"]
        ),
        on="Item",
        how="left",
    )
    # Total proyecto (por seguridad se calcula desde la tabla)
    capex_total_clp_calc = df_items_tot["Total_CLP"].sum()

    # Porcentaje de cada ítem sobre el total
    df_items_tot["Pct_total"] = df_items_tot["Total_CLP"] / capex_total_clp_calc

    # Monto en millones de CLP
    df_items_tot["Total_MM"] = df_items_tot["Total_CLP"] / 1e6

    # Texto dentro de la barra: "XX.X MM / YY.Y%"
    df_items_tot["Texto"] = df_items_tot.apply(
        lambda r: f"{r['Total_MM']:.1f} MM / {r['Pct_total']*100:.1f}%",
        axis=1
    )

    # Orden descendente por monto
    df_items_tot = df_items_tot.sort_values("Total_CLP", ascending=False)

    fig_item_total = px.bar(
        df_items_tot,
        x="Total_MM",
        y="Item",
        orientation="h",
        text="Texto",
        color="Item",
        color_discrete_map=item_color_map,
        labels={
            "Total_MM": "Monto (millones de CLP)",
            "Item": "Ítem",
            "Item": "Ítem",
        },
        title="CAPEX por ítem (monto total y % del CAPEX)",
    )

    fig_item_total.update_traces(
        textposition="inside",
        insidetextanchor="middle",
        textfont_size=11,
    )

    fig_item_total.update_layout(
        xaxis_title="Monto total (millones de CLP)",
        yaxis_title="",
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False,
        height=420,
        bargap=0.25,
    )
    st.plotly_chart(fig_item_total, use_container_width=True)

    st.session_state["fig_item_total"] = fig_item_total

 # ========================
with tab_resumen:
    render_pagos_hitos(capex_url, fx_used, pagos_scale)
    
    # 1.bis) LÍNEA DE TIEMPO TÉCNICA POR CATEGORÍA
    # ========================
    st.markdown("### Línea de tiempo del proyecto por categoría")
    
    if "Mes_inicio" in df_capex.columns and "Mes_termino" in df_capex.columns:
    
        # Filtrar filas válidas
        df_timeline = df_capex.dropna(subset=["Mes_inicio", "Mes_termino"]).copy()
    
        if not df_timeline.empty:
            # Convertir a numérico
            df_timeline["Mes_inicio"] = pd.to_numeric(df_timeline["Mes_inicio"], errors="coerce")
            df_timeline["Mes_termino"] = pd.to_numeric(df_timeline["Mes_termino"], errors="coerce")
            df_timeline = df_timeline.dropna(subset=["Mes_inicio", "Mes_termino"])
    
            if not df_timeline.empty:
                # Agregar a nivel técnico por categoría + ítem
                df_timeline_cat = (
                    df_timeline
                    .groupby(["Categoria", "Item"], as_index=False)
                    .agg(
                        Mes_inicio=("Mes_inicio", "min"),
                        Mes_termino=("Mes_termino", "max"),
                        Monto_CLP=("Monto_CLP", "sum"),
                        Monto_USD=("Monto_USD", "sum"),
                    )
                )
    
                # =============================================
                # Filtro estilo "píldoras" con emojis (radio button)
                # =============================================
                items_todos = sorted(df_timeline_cat["Item"].unique().tolist())
                opciones = ["Todas"] + items_todos
    
                def _fmt_item(opt: str) -> str:
                    if opt == "Todas":
                        return "🔴 Todas"
                    mapa = {
                        "Desarrollo Tecnológico": "🧪 Desarrollo Tecnológico",
                        "Componentes Mecánicos": "⚙️ Componentes Mecánicos",
                        "Sistema Eléctrico y Control": "🔌 Sistema Eléctrico y Control",
                        "Obras Civiles": "🏗️ Obras Civiles",
                        "Montaje y Logística": "📦 Montaje y Logística",
                        "Ensayos y Certificación": "📏 Ensayos y Certificación",
                        "Contingencias y Administración": "🧾 Contingencias y Administración",
                    }
                    return mapa.get(opt, opt)
    
                item_sel = st.radio(
                    "Filtrar por ítem:",
                    options=opciones,
                    index=0,
                    horizontal=True,
                    key="timeline_radio_item_cat",
                    format_func=_fmt_item,
                )
    
                # Filtrar ítems
                df_tl_plot = (
                    df_timeline_cat.copy()
                    if item_sel == "Todas"
                    else df_timeline_cat[df_timeline_cat["Item"] == item_sel].copy()
                )
    
                if not df_tl_plot.empty:
                    # =============================================
                    # Mapear Mes 1–15 a fechas reales ficticias
                    # =============================================
                    base_date = pd.to_datetime("2025-01-01")
    
                    df_tl_plot["Fecha_inicio"] = base_date + pd.to_timedelta(
                        (df_tl_plot["Mes_inicio"] - 1) * 30, unit="D"
                    )
                    df_tl_plot["Fecha_termino"] = base_date + pd.to_timedelta(
                        (df_tl_plot["Mes_termino"] - 1) * 30, unit="D"
                    )
    
                    # =============================================
                    # ORDEN TÉCNICO: tareas de la más lejana a la más cercana
                    # =============================================
                    df_tl_plot = df_tl_plot.sort_values(
                        by=["Fecha_inicio", "Fecha_termino"],
                        ascending=[False, False],  # descendente
                    )
    
                    # Mantener este orden en el eje Y
                    df_tl_plot["Categoria"] = pd.Categorical(
                        df_tl_plot["Categoria"],
                        categories=df_tl_plot["Categoria"].tolist(),
                        ordered=True,
                    )
    
                    # =============================================
                    # Construcción de la Gantt técnica
                    # =============================================
                    fig_timeline_cat = px.timeline(
                        df_tl_plot,
                        x_start="Fecha_inicio",
                        x_end="Fecha_termino",
                        y="Categoria",
                       color="Item",
                       color_discrete_map=item_color_map,   # <--- CLAVE
                       hover_data={
                           "Categoria": True,
                           "Item": True,
                           "Mes_inicio": True,
                           "Mes_termino": True,
                           "Monto_CLP": ":,.0f",
                           "Monto_USD": ":,.0f",
                     },
        
                    )
    
    
                    fig_timeline_cat.update_yaxes(
                        categoryorder="array",
                        title="Categoría / Tarea",
                    )
    
                    # ==========================
                    # Eje X formateado como meses 1–15
                    # ==========================
                    fig_timeline_cat.update_xaxes(
                        title="Mes del proyecto",
                        tickmode="array",
                        tickvals=df_tl_plot["Fecha_inicio"].sort_values().unique(),
                        ticktext=df_tl_plot["Mes_inicio"].sort_values().unique(),
                        showgrid=True,
                    )
    
                    fig_timeline_cat.update_layout(
                        margin=dict(l=10, r=10, t=60, b=10),
                        height=520,
                        legend_title_text="Ítem",
                    )
    
                    st.plotly_chart(fig_timeline_cat, use_container_width=True)
    
                    st.markdown("---")
                    st.subheader("Línea de tiempo de hitos por profesional")
    
                    try:
                        hitos_url = (
                            "https://docs.google.com/spreadsheets/d/e/"
                            "2PACX-1vSlNd3zXc1zV6TUQHnhXlfZtv7QVOv0mBfR_HH69Ht-0qi2aDtCfw5ouLDGIoPH_knhSAtyT2DYE-Qo/"
                            "pub?gid=1007478838&single=true&output=csv"
                        )
                        df_hitos = pd.read_csv(hitos_url, dtype=str)
                        df_hitos.columns = [str(c).strip() for c in df_hitos.columns]
    
                        required_cols = [
                            "Hito_ID",
                            "Hito",
                            "Descripción",
                            "Entregables",
                            "Criterio de salida",
                            "Owner",
                            "Mes objetivo",
                            "Depende de",
                        ]
                        missing_cols = [c for c in required_cols if c not in df_hitos.columns]
                        if missing_cols:
                            st.error(f"Faltan columnas en la hoja de hitos: {missing_cols}")
                        else:
                            df_hitos["Mes_objetivo_i"] = pd.to_numeric(
                                df_hitos["Mes objetivo"], errors="coerce"
                            )
                            df_hitos = df_hitos.dropna(subset=["Mes_objetivo_i"]).copy()
    
                            if df_hitos.empty:
                                st.info("No hay hitos con Mes objetivo válido.")
                            else:
                                df_hitos["Hito_label"] = (
                                    df_hitos["Hito_ID"].astype(str).str.strip()
                                    + " — "
                                    + df_hitos["Hito"].astype(str).str.strip()
                                )
                                df_hitos = df_hitos.sort_values("Mes_objetivo_i")
                                y_order = df_hitos["Hito_label"].tolist()
    
                                fig_hitos = px.scatter(
                                    df_hitos,
                                    x="Mes_objetivo_i",
                                    y="Hito_label",
                                    color="Owner",
                                    labels={
                                        "Mes_objetivo_i": "Mes objetivo",
                                        "Hito_label": "Hito",
                                        "Owner": "Owner",
                                    },
                                    hover_data={
                                        "Hito_ID": True,
                                        "Hito": True,
                                        "Descripción": True,
                                        "Entregables": True,
                                        "Criterio de salida": True,
                                        "Owner": True,
                                        "Mes objetivo": True,
                                        "Depende de": True,
                                    },
                                    title="Ruta crítica de hitos del proyecto",
                                )
    
                                fig_hitos.update_traces(
                                    marker=dict(size=14, line=dict(width=1, color="#111827"))
                                )
    
                                hito_pos = {
                                    row["Hito_ID"]: (row["Mes_objetivo_i"], row["Hito_label"])
                                    for _, row in df_hitos.iterrows()
                                }
                                for _, row in df_hitos.iterrows():
                                    dep = str(row.get("Depende de", "")).strip()
                                    if not dep or dep == "-":
                                        continue
                                    if dep in hito_pos:
                                        x0, y0 = hito_pos[dep]
                                        x1, y1 = row["Mes_objetivo_i"], row["Hito_label"]
                                        fig_hitos.add_trace(
                                            go.Scatter(
                                                x=[x0, x1],
                                                y=[y0, y1],
                                                mode="lines",
                                                line=dict(color="#9CA3AF", width=2, dash="dot"),
                                                hoverinfo="skip",
                                                showlegend=False,
                                            )
                                        )
    
                                fig_hitos.update_layout(
                                    height=520,
                                    margin=dict(l=10, r=10, t=60, b=20),
                                    yaxis=dict(categoryorder="array", categoryarray=y_order),
                                    xaxis=dict(
                                        dtick=1,
                                        showgrid=True,
                                        title="Mes objetivo",
                                    ),
                                    legend=dict(
                                        orientation="h",
                                        yanchor="top",
                                        y=-0.2,
                                        xanchor="center",
                                        x=0.5,
                                    ),
                                )
                                st.plotly_chart(fig_hitos, use_container_width=True)
    
                    except Exception as e:
                        st.error(f"No se pudo construir la línea de tiempo de hitos: {e}")
    
                    st.markdown("---")
                    st.subheader("Mapa de zonas críticas de riesgos")
    
                    try:
                        riesgos_url = (
                            "https://docs.google.com/spreadsheets/d/e/"
                            "2PACX-1vSlNd3zXc1zV6TUQHnhXlfZtv7QVOv0mBfR_HH69Ht-0qi2aDtCfw5ouLDGIoPH_knhSAtyT2DYE-Qo/"
                            "pub?gid=1912427793&single=true&output=csv"
                        )
                        df_riesgos = pd.read_csv(riesgos_url, dtype=str)
                        df_riesgos.columns = [str(c).strip() for c in df_riesgos.columns]
    
                        def _norm_col(s: str) -> str:
                            s = str(s)
                            s = unicodedata.normalize("NFKD", s)
                            s = "".join(c for c in s if not unicodedata.combining(c))
                            return re.sub(r"[^a-z0-9]", "", s.lower())
    
                        col_lookup = {_norm_col(c): c for c in df_riesgos.columns}
                        required_norm = {
                            "riesgoid": "Riesgo_ID",
                            "categoria": "Categoria",
                            "riesgo": "Riesgo",
                            "probabilidad15": "Probabilidad (1-5)",
                            "impacto15": "Impacto (1-5)",
                            "severidadpxi": "Severidad (PxI)",
                            "owner": "Owner",
                            "relacionadohito": "Relacionado Hito",
                            "estado": "Estado",
                        }
                        missing_cols = [
                            display
                            for key, display in required_norm.items()
                            if key not in col_lookup
                        ]
                        if missing_cols:
                            st.error(f"Faltan columnas en la hoja de riesgos: {missing_cols}")
                        else:
                            rename_map = {
                                col_lookup[key]: display
                                for key, display in required_norm.items()
                            }
                            df_riesgos = df_riesgos.rename(columns=rename_map)
    
                            df_riesgos["Severidad_i"] = pd.to_numeric(
                                df_riesgos["Severidad (PxI)"], errors="coerce"
                            )
                            df_riesgos["Probabilidad_i"] = pd.to_numeric(
                                df_riesgos["Probabilidad (1-5)"], errors="coerce"
                            )
                            df_riesgos["Impacto_i"] = pd.to_numeric(
                                df_riesgos["Impacto (1-5)"], errors="coerce"
                            )
                            df_riesgos = df_riesgos.dropna(
                                subset=["Probabilidad_i", "Impacto_i", "Severidad_i"]
                            ).copy()
    
                            if df_riesgos.empty:
                                st.info("No hay riesgos con Probabilidad/Impacto válidos.")
                            else:
                                categorias = sorted(df_riesgos["Categoria"].astype(str).unique().tolist())
                                offset_map = {}
                                n_cats = len(categorias)
                                for idx, cat in enumerate(categorias):
                                    offset = (idx - (n_cats - 1) / 2) * 0.08
                                    offset_map[cat] = offset
    
                                df_riesgos["Prob_plot"] = df_riesgos["Probabilidad_i"] + df_riesgos[
                                    "Categoria"
                                ].map(offset_map).fillna(0.0)
                                df_riesgos["Imp_plot"] = df_riesgos["Impacto_i"] - df_riesgos[
                                    "Categoria"
                                ].map(offset_map).fillna(0.0)
    
                                df_riesgos["Prob_plot"] = df_riesgos["Prob_plot"].clip(0.6, 5.4)
                                df_riesgos["Imp_plot"] = df_riesgos["Imp_plot"].clip(0.6, 5.4)
    
                                fig_riesgos = px.scatter(
                                    df_riesgos,
                                    x="Prob_plot",
                                    y="Imp_plot",
                                    color="Categoria",
                                    symbol="Categoria",
                                    size="Severidad_i",
                                    size_max=28,
                                    labels={
                                        "Prob_plot": "Probabilidad (1–5)",
                                        "Imp_plot": "Impacto (1–5)",
                                        "Categoria": "Categoría",
                                    },
                                    hover_data={
                                        "Riesgo_ID": True,
                                        "Riesgo": True,
                                        "Probabilidad (1-5)": True,
                                        "Impacto (1-5)": True,
                                        "Severidad (PxI)": True,
                                        "Owner": True,
                                        "Estado": True,
                                    },
                                    title="Mapa de riesgos (Probabilidad × Impacto)",
                                )
                                fig_riesgos.update_traces(
                                    marker=dict(line=dict(width=1, color="#111827"))
                                )
                                shapes = []
                                for p in range(1, 6):
                                    for i in range(1, 6):
                                        sev = p * i
                                        if sev <= 6:
                                            fill = "#D1FAE5"
                                        elif sev <= 12:
                                            fill = "#FEF3C7"
                                        else:
                                            fill = "#FEE2E2"
                                        shapes.append(
                                            dict(
                                                type="rect",
                                                xref="x",
                                                yref="y",
                                                x0=p - 0.5,
                                                y0=i - 0.5,
                                                x1=p + 0.5,
                                                y1=i + 0.5,
                                                fillcolor=fill,
                                                opacity=0.35,
                                                line=dict(width=0),
                                                layer="below",
                                            )
                                        )
    
                                fig_riesgos.update_layout(
                                    height=520,
                                    margin=dict(l=10, r=10, t=60, b=20),
                                    xaxis=dict(dtick=1, range=[0.5, 5.5], showgrid=True),
                                    yaxis=dict(dtick=1, range=[0.5, 5.5], showgrid=True),
                                    legend=dict(
                                        orientation="h",
                                        yanchor="top",
                                        y=-0.2,
                                        xanchor="center",
                                        x=0.5,
                                    ),
                                    shapes=shapes,
                                )
                                st.plotly_chart(fig_riesgos, use_container_width=True)
    
                    except Exception as e:
                        st.error(f"No se pudo construir la línea de tiempo de riesgos: {e}")
    
                else:
                    st.info("No hay categorías para el ítem seleccionado en la línea de tiempo.")
            else:
                st.info("No hay datos válidos en 'Mes_inicio' y 'Mes_termino'.")
        else:
            st.info("La tabla de CAPEX no contiene datos suficientes para construir la línea de tiempo.")
    else:
        st.info("Agrega columnas 'Mes_inicio' y 'Mes_termino' en Google Sheets para habilitar esta sección.")
    
    
        # ========================
        # 1) Gráfico PRO: ítems segmentados por categoría
        # ========================
    
        # Totales por categoría
        df_cat_tot = (
            df_capex
            .groupby("Categoria", as_index=False)
            .agg(Total_CLP=("Monto_CLP", "sum"))
        )
    
        df_capex_merged = df_capex.merge(df_cat_tot, on="Categoria", how="left")
    
        # % del ítem dentro de su categoría y del proyecto
        total_clp = df_capex_merged["Monto_CLP"].sum()
        df_capex_merged["Pct_en_categoria"] = df_capex_merged["Monto_CLP"] / df_capex_merged["Total_CLP"]
        df_capex_merged["Pct_total"] = df_capex_merged["Monto_CLP"] / total_clp
    
        # Texto dentro del segmento: "x.x MM / yy% cat"
        df_capex_merged["Texto_seg"] = df_capex_merged.apply(
            lambda r: f"{r['Monto_CLP_MM']:.1f} MM\n{r['Pct_en_categoria']*100:.0f}% cat",
            axis=1
        )
    
        # Ordenar categorías por total CLP (de menor a mayor para que la mayor quede abajo)
        cat_order = (
            df_capex_merged
            .groupby("Categoria")["Monto_CLP"]
            .sum()
            .sort_values(ascending=True)
            .index
            .tolist()
        )
    
        df_capex_plot = df_capex_merged.copy()
        df_capex_plot["Categoria"] = pd.Categorical(
            df_capex_plot["Categoria"],
            categories=cat_order,
            ordered=True,
        )
    
        fig_stack = px.bar(
            df_capex_plot,
            x="Monto_CLP_MM",
            y="Categoria",
            color="Item",
            orientation="h",
            text="Texto_seg",
            labels={
                "Monto_CLP_MM": "Monto (millones de CLP)",
                "Categoria": "",
                "Item": "Ítem",
            },
            title="CAPEX por ítem segmentado por categoría (millones de CLP)",
        )
    
        fig_stack.update_traces(
            textposition="inside",
            insidetextanchor="middle",
            textfont_size=10,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Categoría: %{y}<br>"
                "Monto: %{x:.2f} MM CLP<br>"
                "Monto CLP: %{customdata[1]:,.0f}<br>"
                "Monto USD: %{customdata[2]:,.0f}<br>"
                "% en categoría: %{customdata[3]:.1%}<br>"
                "% del proyecto: %{customdata[4]:.1%}<br>"
                "<extra></extra>"
            ),
            customdata=np.stack([
                df_capex_plot["Item"],
                df_capex_plot["Monto_CLP"],
                df_capex_plot["Monto_USD"],
                df_capex_plot["Pct_en_categoria"],
                df_capex_plot["Pct_total"],
            ], axis=-1),
        )
    
        fig_stack.update_layout(
            xaxis_title="Monto (millones de CLP)",
            yaxis_title="",
            margin=dict(l=10, r=10, t=60, b=10),
            legend_title_text="Ítem",
            height=650,
        )
    
        # Totales por categoría al final de cada barra
        for _, row in df_cat_tot.iterrows():
            fig_stack.add_annotation(
                x=row["Total_CLP"] / 1e6,
                y=row["Categoria"],
                text=f"{row['Total_CLP']/1e6:.1f} MM",
                xanchor="left",
                yanchor="middle",
                showarrow=False,
                font=dict(size=11, color="black"),
                align="left",
                xshift=12,
            )
    
        st.plotly_chart(fig_stack, use_container_width=True)
    
        # ========================
        # 2) Gráfico por CATEGORÍA (abajo)
        # ========================
    
        st.markdown("### Participación relativa por categoría")
    
        fig_cat_pie = px.pie(
            df_cat,
            values="Monto_CLP",
            names="Categoria",
            title="Participación relativa por categoría",
            hole=0.45,
        )
    
        fig_cat_pie.update_traces(
            textposition="inside",
            textinfo="percent+label",
            hovertemplate="<b>%{label}</b><br>"
                          "Participación: %{percent:.1%}<br>"
                          "Monto CLP: %{value:,.0f}<br>"
                          "<extra></extra>",
        )
        fig_cat_pie.update_layout(
            margin=dict(l=10, r=10, t=60, b=10),
            height=600,
        )
    
        st.plotly_chart(fig_cat_pie, use_container_width=True)
        st.session_state["fig_cat_pie"] = fig_cat_pie
    
        # ========================
        # 3) Comentario automático
        # ========================
    
        st.markdown("### Lectura rápida")
        st.write(
            f"- La categoría **{cat_top}** concentra aproximadamente **{cat_top_pct:.1f}%** del CAPEX total.\n"
            f"- Se consideran **{total_items} ítems** distribuidos en **{total_categorias} categorías**.\n"
            f"- El tipo de cambio implícito de la tabla es de **{tipo_cambio_implicito:,.0f} CLP/US$**, "
            f"coherente con un CAPEX de {format_clp(capex_total_clp)}."
        )
    
# -------------------------
# TAB CATEGORÍA (VERSIÓN PRO)
# -------------------------
with tab_categoria:
    st.subheader("Análisis técnico por categoría")

    df_cat_filtrado = df_cat.copy()

    # =======================
    # 2.bis) Dónuts por ÍTEM (distribución por categoría) + porcentaje al centro
    # =======================
    st.markdown("### Gráfico por ítem - distribución del capex por categoría")

    df_capex_filtrado = df_capex.copy()
    items_unicos = df_capex_filtrado["Item"].unique().tolist()
    num_items = len(items_unicos)

    if num_items > 0:
        n_cols = 3
        n_rows = math.ceil(num_items / n_cols)

        for row_idx in range(n_rows):
            cols = st.columns(n_cols)

            for col_idx in range(n_cols):
                idx = row_idx * n_cols + col_idx
                if idx >= num_items:
                    break

                item_name = items_unicos[idx]

                with cols[col_idx]:
                    st.markdown(f"**{item_name}**")

                    df_item_cat = (
                        df_capex_filtrado[df_capex_filtrado["Item"] == item_name]
                        .groupby("Categoria", as_index=False)
                        .agg(Monto_CLP=("Monto_CLP", "sum"))
                    )

                    if df_item_cat.empty:
                        st.caption("Sin distribución disponible para este ítem.")
                        continue

                    # CALCULAR PORCENTAJE TOTAL DEL ÍTEM RESPECTO AL CAPEX SELECCIONADO
                    total_item = df_item_cat["Monto_CLP"].sum()
                    total_capex_visible = df_capex_filtrado["Monto_CLP"].sum()
                    pct_item_total = (total_item / total_capex_visible) if total_capex_visible > 0 else 0

                    fig_donut_item = px.pie(
                        df_item_cat,
                        values="Monto_CLP",
                        names="Categoria",
                        hole=0.70,
                    )

                    fig_donut_item.update_traces(
                        textinfo="percent",
                        textposition="inside",
                        hovertemplate="<b>%{label}</b><br>"
                                      "Participación dentro del ítem: %{percent:.1%}<br>"
                                      "Monto CLP: %{value:,.0f}<br>"
                                      "<extra></extra>",
                        insidetextorientation="horizontal"
                    )

                    # TEXTO CENTRAL
                    fig_donut_item.add_annotation(
                        x=0.5,
                        y=0.5,
                        text=f"{pct_item_total*100:.1f}%",
                        showarrow=False,
                        font=dict(size=22, color="black"),
                        xanchor="center",
                        yanchor="middle"
                    )

                    fig_donut_item.update_layout(
                        showlegend=True,
                        legend=dict(
                            orientation="v",
                            x=1.25,
                            y=0.5,
                            xanchor="left",
                            font=dict(size=11),
                        ),
                        margin=dict(l=0, r=120, t=10, b=10),
                        height=280,
                    )

                    st.plotly_chart(fig_donut_item, use_container_width=True)

    else:
        st.info("No hay ítems para mostrar en los dónuts según las categorías seleccionadas.")

    # =======================
    # 2) Gráfico vertical profesional
    # =======================
    st.markdown("### Participación porcentual por categoría")

    total_clp_cat = df_cat_filtrado["Monto_CLP"].sum()
    df_cat_plot = df_cat_filtrado.copy()
    df_cat_plot = df_cat_plot.sort_values("Monto_CLP", ascending=False)

    if total_clp_cat > 0:
        df_cat_plot["Pct_cat"] = df_cat_plot["Monto_CLP"] / total_clp_cat
    else:
        df_cat_plot["Pct_cat"] = 0.0

    df_cat_item = (
        df_capex.groupby(["Categoria", "Item"], as_index=False)
        .agg(Monto_CLP=("Monto_CLP", "sum"))
        .sort_values(["Categoria", "Monto_CLP"], ascending=[True, False])
    )
    top_item_by_cat = df_cat_item.drop_duplicates(subset=["Categoria"], keep="first")
    cat_item_color_map = {
        row["Categoria"]: item_color_map.get(
            row["Item"],
            CAT_COLOR_MAP.get(row["Categoria"], "#2563EB"),
        )
        for _, row in top_item_by_cat.iterrows()
    }

    fig_cat = px.bar(
        df_cat_plot,
        x="Categoria",
        y="Pct_cat",
        color="Categoria",
        color_discrete_map=cat_item_color_map,
        text="Pct_cat",
        labels={
            "Categoria": "Categoría",
            "Pct_cat": "Participación",
        },
        title="Distribución porcentual del CAPEX por categoría",
    )

    fig_cat.update_traces(
        texttemplate="%{text:.1%}",
        textposition="outside",
    )

    max_part = float(df_cat_plot["Pct_cat"].max() or 0)
    fig_cat.update_yaxes(
        tickformat=".0%",
        range=[0, max_part * 1.15 if max_part > 0 else 1]
    )

    fig_cat.update_layout(
        xaxis_title="",
        yaxis_title="Participación (%)",
        margin=dict(l=10, r=10, t=80, b=120),
        height=460,
        bargap=0.25,
        showlegend=False,
    )

    st.plotly_chart(fig_cat, use_container_width=True)
    legend_css = """
    <style>
    .item-legend {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem 0.75rem;
        margin-top: 0.4rem;
        margin-bottom: 0.8rem;
    }
    .item-legend-title {
        font-size: 0.82rem;
        font-weight: 700;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: .08em;
        margin-top: 0.4rem;
        margin-bottom: 0.25rem;
    }
    .item-legend-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        font-size: 0.78rem;
        font-weight: 600;
        color: #111827;
    }
    .item-legend-swatch {
        width: 12px;
        height: 12px;
        border-radius: 2px;
        border: 1px solid rgba(17, 24, 39, 0.2);
    }
    </style>
    """
    legend_items = []
    legend_order = [
        "Desarrollo Tecnológico",
        "Componentes Mecánicos",
        "Sistema Eléctrico y Control",
        "Obras Civiles",
        "Montaje y Logística",
        "Ensayos y Certificación",
        "Contingencias y Administración",
    ]
    for item in legend_order:
        color = CAT_COLOR_MAP.get(item, "#2563EB")
        legend_items.append(
            f'<span class="item-legend-chip"><span class="item-legend-swatch" '
            f'style="background:{color}"></span>{item}</span>'
        )
    st.markdown(
        legend_css
        + '<div class="item-legend-title">Ítem</div>'
        + f'<div class="item-legend">{"".join(legend_items)}</div>',
        unsafe_allow_html=True,
    )
    st.session_state["fig_cat_categoria"] = fig_cat

    # =======================
    # 3) Tabla ejecutiva (detalle técnico)
    # =======================
    st.markdown("### Tabla distribución por categoría")

    df_bullet_cat = (
        df_capex.groupby("Categoria", as_index=False)
        .agg(Bullet_cat=("Bullet", lambda x: x.dropna().iloc[0] if len(x.dropna()) > 0 else ""))
    )

    df_show = df_cat_plot.merge(df_bullet_cat, on="Categoria", how="left")

    df_show["Participación (%)"] = (df_show["Pct_cat"] * 100).map(lambda v: f"{v:.1f}%")
    df_show["Monto_CLP_fmt"] = df_show["Monto_CLP"].apply(format_clp)
    df_show["Monto_USD_fmt"] = df_show["Monto_USD"].apply(format_usd)

    st.dataframe(
        df_show[[
            "Categoria",
            "Participación (%)",
            "Monto_CLP_fmt",
            "Monto_USD_fmt",
            "Bullet_cat",
        ]],
        hide_index=True,
        use_container_width=True,
    )

# -------------------------
# TAB ÍTEMS
# -------------------------
with tab_items:
    st.subheader("Top ítems por monto")

    render_category_palette()

    top_n = st.slider("Número de ítems a mostrar (Top N):", 5, 30, 15, step=1)

    df_top = (
        df_capex
        .sort_values("Monto_CLP", ascending=False)
        .head(top_n)
        .copy()
    )
    df_top["Monto_CLP_fmt"] = df_top["Monto_CLP"].apply(format_clp)
    df_top["Monto_USD_fmt"] = df_top["Monto_USD"].apply(format_usd)
    df_top["Participación (%)"] = df_top["Participacion_pct"] * 100
    df_top["Monto_CLP_MM"] = df_top["Monto_CLP"] / 1e6

    fig_top = px.bar(
        df_top,
        x="Monto_CLP_MM",
        y="Item",
        color="Categoria",
        color_discrete_map=CAT_COLOR_MAP,
        orientation="h",
        hover_data={
            "Monto_CLP_MM": False,
            "Monto_CLP": ":,.0f",
            "Monto_USD": ":,.0f",
            "Participacion_pct": ":.2%",
            "Categoria": True,
        },
        labels={
            "Monto_CLP_MM": "Monto (MM CLP)",
            "Item": "Ítem",
            "Categoria": "Categoría",
        },
        title=f"Top {top_n} ítems por monto (millones de CLP)",
    )
    fig_top.update_traces(
        text=df_top["Monto_CLP_MM"].apply(lambda v: f"{v:.1f} MM"),
        textposition="outside",
    )
    fig_top.update_layout(
        xaxis_title="Monto (millones de CLP)",
        yaxis_title="",
        margin=dict(l=10, r=10, t=60, b=10),
    )
    st.plotly_chart(fig_top, use_container_width=True)
    st.session_state["fig_top_items"] = fig_top

    st.markdown("#### Tabla detallada")
    st.dataframe(
        df_top[[
            "Item",
            "Categoria",
            "Participación (%)",
            "Monto_CLP_fmt",
            "Monto_USD_fmt",
            "Bullet",
        ]],
        hide_index=True,
        use_container_width=True,
    )

# -------------------------
# TAB EXPLORADOR
# -------------------------
with tab_explorador:
    st.subheader("Explorador interactivo de la tabla CAPEX")

    col_f1, col_f2, col_f3, col_f4 = st.columns(4)

    with col_f1:
        categoria_filter = st.selectbox(
            "Filtrar por categoría:",
            options=["(Todas)"] + sorted(df_capex["Categoria"].unique().tolist()),
            index=0,
        )

    with col_f2:
        item_filter = st.selectbox(
            "Filtrar por ítem:",
            options=["(Todos)"] + sorted(df_capex["Item"].unique().tolist()),
            index=0,
        )

    with col_f3:
        min_pct = st.slider(
            "Participación mínima del ítem (%)",
            min_value=0.0,
            max_value=5.0,
            value=0.0,
            step=0.1,
        )

    with col_f4:
        ordenar_por = st.selectbox(
            "Ordenar por:",
            options=["Monto_CLP", "Monto_USD", "Participacion_pct"],
            index=0,
        )

    df_exp = df_capex.copy()
    if categoria_filter != "(Todas)":
        df_exp = df_exp[df_exp["Categoria"] == categoria_filter]
    if item_filter != "(Todos)":
        df_exp = df_exp[df_exp["Item"] == item_filter]
    df_exp = df_exp[df_exp["Participacion_pct"] * 100 >= min_pct]
    df_exp = df_exp.sort_values(ordenar_por, ascending=False)

    df_exp["Participación (%)"] = df_exp["Participacion_pct"] * 100
    df_exp["Monto_CLP_fmt"] = df_exp["Monto_CLP"].apply(format_clp)
    df_exp["Monto_USD_fmt"] = df_exp["Monto_USD"].apply(format_usd)

    st.markdown("#### Tabla filtrada")
    st.dataframe(
        df_exp[[
            "Item",
            "Categoria",
            "Participación (%)",
            "Monto_CLP_fmt",
            "Monto_USD_fmt",
            "Bullet",
        ]],
        hide_index=True,
        use_container_width=True,
    )

    st.markdown("#### Vista gráfica de los ítems filtrados")
    if not df_exp.empty:
        df_exp["Monto_CLP_MM"] = df_exp["Monto_CLP"] / 1e6
        df_plot = df_exp.sort_values("Monto_CLP", ascending=False).head(20)
        fig_exp = px.bar(
            df_plot,
            x="Monto_CLP_MM",
            y="Categoria",
            color="Item",
            color_discrete_map=item_color_map,
            orientation="h",
            labels={
                "Monto_CLP_MM": "Monto (MM CLP)",
                "Categoria": "Categoría (sub-ítem)",
                "Item": "Ítem",
            },
            title="Ítems filtrados (hasta 20 primeros)",
        )
        fig_exp.update_traces(
            text=df_plot["Monto_CLP_MM"].apply(lambda v: f"{v:.1f} MM"),
            textposition="outside",
        )
        fig_exp.update_layout(
            xaxis_title="Monto (millones de CLP)",
            yaxis_title="",
            margin=dict(l=10, r=10, t=60, b=10),
            legend_title_text="Ítem",
        )
        st.plotly_chart(fig_exp, use_container_width=True)
    else:
        st.info("No hay ítems que cumplan con los filtros seleccionados.")

    csv_bytes = df_exp.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Descargar CSV filtrado",
        data=csv_bytes,
        file_name="capex_filtrado.csv",
        mime="text/csv",
    )

# =========================
# REPORTING PDF TÉCNICO
# =========================

def build_pdf_report() -> bytes:
    """Genera un informe técnico en PDF para directivos con KPIs y gráficos principales."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    h1 = styles["Heading1"]
    h2 = styles["Heading2"]
    body = styles["BodyText"]

    elementos = []

    # --- Portada ---
    elementos.append(Paragraph("Reporte CAPEX – Piloto Eólico 80 kW", h1))
    elementos.append(Spacer(1, 0.4 * cm))
    elementos.append(Paragraph("Informe técnico ejecutivo para directorio y comité de inversiones.", body))
    elementos.append(Spacer(1, 1.0 * cm))

    # Tabla de KPIs
    kpi_data = [
        ["Indicador", "Valor", "Comentario"],
        ["CAPEX total (CLP)", format_clp(capex_total_clp), "Inversión total piloto 80 kW"],
        ["CAPEX total (USD)", format_usd(capex_total_usd), f"Tipo de cambio implícito ≈ {tipo_cambio_implicito:,.0f} CLP/US$"],
        ["Tipo de cambio implícito", f"{tipo_cambio_implicito:,.1f} CLP/US$", "CAPEX CLP / suma de costos en USD"],
        ["Categoría de mayor peso", cat_top, f"≈ {cat_top_pct:.1f}% del CAPEX total"],
    ]
    kpi_table = Table(kpi_data, colWidths=[5 * cm, 4 * cm, 7 * cm])
    kpi_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("ALIGN", (1, 1), (-1, -1), "LEFT"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ]
        )
    )
    elementos.append(kpi_table)
    elementos.append(PageBreak())

    # --- Gráfico 1: CAPEX por ítem ---
    if "fig_item_total" in st.session_state:
        elementos.append(Paragraph("1. Distribución de CAPEX por ítem", h2))
        img_bytes = pio.to_image(st.session_state["fig_item_total"], format="png", scale=2)
        elementos.append(Image(BytesIO(img_bytes), width=17 * cm, height=9 * cm))
        elementos.append(Spacer(1, 0.6 * cm))

    # --- Gráfico 2: Pie por categoría (resumen ejecutivo) ---
    if "fig_cat_pie" in st.session_state:
        elementos.append(Paragraph("2. Participación relativa por categoría", h2))
        img_bytes = pio.to_image(st.session_state["fig_cat_pie"], format="png", scale=2)
        elementos.append(Image(BytesIO(img_bytes), width=14 * cm, height=8 * cm))
        elementos.append(PageBreak())

    # --- Gráfico 3: Barra por categoría (TAB 'Por categoría') ---
    if "fig_cat_categoria" in st.session_state:
        elementos.append(Paragraph("3. Análisis de CAPEX por categoría (vista ingeniería)", h2))
        img_bytes = pio.to_image(st.session_state["fig_cat_categoria"], format="png", scale=2)
        elementos.append(Image(BytesIO(img_bytes), width=17 * cm, height=9 * cm))
        elementos.append(Spacer(1, 0.6 * cm))

    # --- Gráfico 4: Top ítems (TAB 'Detalle de ítems') ---
    if "fig_top_items" in st.session_state:
        elementos.append(Paragraph("4. Top ítems de inversión", h2))
        img_bytes = pio.to_image(st.session_state["fig_top_items"], format="png", scale=2)
        elementos.append(Image(BytesIO(img_bytes), width=17 * cm, height=9 * cm))
        elementos.append(PageBreak())

    # --- Tablas resumen clave ---
    elementos.append(Paragraph("5. Resumen tabular de categorías", h2))
    df_tab_cat = df_cat.sort_values("Monto_CLP", ascending=False).head(10).copy()
    df_tab_cat["Participación (%)"] = df_tab_cat["Participacion_sum"] * 100

    table_data = [["Categoría", "Participación (%)", "Monto CLP", "Monto USD"]]
    for _, row in df_tab_cat.iterrows():
        table_data.append(
            [
                row["Categoria"],
                f"{row['Participación (%)']:.1f}%",
                format_clp(row["Monto_CLP"]),
                format_usd(row["Monto_USD"]),
            ]
        )

    cat_table = Table(table_data, colWidths=[7 * cm, 3 * cm, 3.5 * cm, 3.5 * cm])
    cat_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ]
        )
    )
    elementos.append(cat_table)

    doc.build(elementos)
    pdf_value = buffer.getvalue()
    buffer.close()
    return pdf_value


with tab_resumen:
    st.markdown("---")
    st.subheader("📄 Exportar informe técnico")
    
    pdf_bytes = build_pdf_report()
    st.download_button(
        label="📥 Descargar reporte PDF técnico (CAPEX Piloto 80 kW)",
        data=pdf_bytes,
        file_name="Reporte_CAPEX_Piloto_80kW.pdf",
        mime="application/pdf",
    )

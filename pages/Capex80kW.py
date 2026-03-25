# capex_piloto_80kw.py
# Dashboard CAPEX Piloto Eólico 80 kW (versión mejorada)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO
import base64
import plotly.io as pio
import re
import unicodedata
import math
import plotly.graph_objects as go
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from pathlib import Path

try:
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
    REPORTLAB_AVAILABLE = True
except ModuleNotFoundError:
    REPORTLAB_AVAILABLE = False


# =========================
# CONFIGURACIÓN GLOBAL
# =========================
st.set_page_config(
    page_title="CAPEX Piloto Eólico 80 kW",
    layout="wide",
    initial_sidebar_state="collapsed",
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
VALORIZACION_CSV_URL_DEFAULT = (
    "https://docs.google.com/spreadsheets/d/e/"
    "2PACX-1vQfQcSn40boiOyRvYeX1j5SO2O9w3WoA6DkOEMxxf85v-WiWXuMC-uyBWb3-ff82pUfk1cSaBnmrcqU/"
    "pub?gid=1756066076&single=true&output=csv"
)
EERRV2_CSV_URL_DEFAULT = (
    "https://docs.google.com/spreadsheets/d/e/"
    "2PACX-1vQfQcSn40boiOyRvYeX1j5SO2O9w3WoA6DkOEMxxf85v-WiWXuMC-uyBWb3-ff82pUfk1cSaBnmrcqU/"
    "pub?gid=372370214&single=true&output=csv"
)

_HERO_CANDIDATES = [
    (Path(__file__).parent / "assets" / "hero_vawt.jpg").resolve(),
    (Path(__file__).parent / "hero_vawt.jpg").resolve(),
]


def _hero_path():
    for candidate in _HERO_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def render_inputs_main_hero() -> None:
    path = _hero_path()
    hero_bg_css = ""
    if path:
        suffix = path.suffix.lower()
        mime = "image/jpeg"
        if suffix == ".png":
            mime = "image/png"
        elif suffix == ".webp":
            mime = "image/webp"
        encoded = base64.b64encode(path.read_bytes()).decode()
        hero_bg_css = f"background-image:url('data:{mime};base64,{encoded}');"

    st.markdown(
        f"""
        <style>
        .inputs-main-hero-shell{{
            margin:4px 0 24px 0;
            min-height:210px;
            border-radius:30px;
            position:relative;
            overflow:hidden;
            padding:24px 30px 22px 30px;
            border:1px solid rgba(191,219,254,.42);
            box-shadow:0 22px 42px rgba(15,23,42,.12);
            background:#eff6ff;
        }}
        .inputs-main-hero-shell::before{{
            content:"";
            position:absolute;
            inset:0;
            background-size:cover;
            background-position:center 30%;
            opacity:.96;
            z-index:0;
            {hero_bg_css}
        }}
        .inputs-main-hero-shell::after{{
            content:"";
            position:absolute;
            inset:0;
            z-index:1;
            background:
                linear-gradient(180deg, rgba(255,255,255,.84) 0%, rgba(255,255,255,.62) 28%, rgba(15,23,42,.20) 100%),
                linear-gradient(90deg, rgba(255,255,255,.22) 0%, rgba(255,255,255,.02) 34%, rgba(15,23,42,.14) 100%);
        }}
        .inputs-main-hero-content{{
            position:relative;
            z-index:2;
            max-width:1200px;
        }}
        .inputs-main-hero-top{{
            display:flex;
            align-items:flex-start;
            gap:16px;
        }}
        .inputs-main-hero-ico{{
            width:56px;
            height:56px;
            border-radius:18px;
            display:flex;
            align-items:center;
            justify-content:center;
            font-size:34px;
            background:linear-gradient(180deg,#eff6ff 0%,#dbeafe 100%);
            border:1px solid rgba(96,165,250,.26);
            box-shadow:0 10px 24px rgba(59,130,246,.12);
            flex:0 0 auto;
            margin-top:4px;
        }}
        .inputs-main-hero-title{{
            font-size: clamp(34px, 4.6vw, 58px);
            line-height:1.02;
            font-weight:900;
            letter-spacing:-0.03em;
            color:#111827;
            margin:0;
            text-shadow:0 1px 0 rgba(255,255,255,.30);
        }}
        .inputs-main-hero-sub{{
            font-size:15px;
            line-height:1.65;
            color:#4b5563;
            max-width:1080px;
            margin:14px 0 0 0;
        }}
        @media (max-width: 900px){{
            .inputs-main-hero-shell{{
                min-height:160px;
                border-radius:24px;
                padding:18px 18px 16px 18px;
            }}
            .inputs-main-hero-top{{
                gap:12px;
            }}
            .inputs-main-hero-ico{{
                width:46px;
                height:46px;
                font-size:28px;
                border-radius:14px;
            }}
        }}
        </style>
        <div class="inputs-main-hero-shell">
          <div class="inputs-main-hero-content">
            <div class="inputs-main-hero-top">
              <div class="inputs-main-hero-ico">📊</div>
              <div>
                <h1 class="inputs-main-hero-title">Arquitectura de Inversión y Creación de Valor</h1>
                <p class="inputs-main-hero-sub">
                  Panel interactivo para analizar la estructura de inversión del piloto de turbina eólica vertical híbrida.
                  Diseñado para uso en directorio, comité técnico y seguimiento de proyecto.
                </p>
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
DASHBOARD_FINANCIERO_CSV_URL_DEFAULT = (
    "https://docs.google.com/spreadsheets/d/e/"
    "2PACX-1vQmVzOg9X7VfxAmOImXHuMvyH4dQmxbFL3DIBqOubi32jKLncgqBEBwnl6j0dXWsm5FkRAcrY4y8BD2/"
    "pub?gid=1596174884&single=true&output=csv"
)
RESTANTE_PILOTO_10KW_CSV_URL_DEFAULT = (
    "https://docs.google.com/spreadsheets/d/e/"
    "2PACX-1vQmVzOg9X7VfxAmOImXHuMvyH4dQmxbFL3DIBqOubi32jKLncgqBEBwnl6j0dXWsm5FkRAcrY4y8BD2/"
    "pub?gid=621130518&single=true&output=csv"
)
BULLET_CONTEXTO_10KW_CSV_URL_DEFAULT = (
    "https://docs.google.com/spreadsheets/d/e/"
    "2PACX-1vQfQcSn40boiOyRvYeX1j5SO2O9w3WoA6DkOEMxxf85v-WiWXuMC-uyBWb3-ff82pUfk1cSaBnmrcqU/"
    "pub?gid=632353264&single=true&output=csv"
)
GANTT_PROJECT_CSV_URL_DEFAULT = (
    "https://docs.google.com/spreadsheets/d/e/"
    "2PACX-1vQOu_diukhhZWDV7kIcU9Ewto4lo_xQdSEZ0FMi2oto-Jb4r2e7aRNCBKF3qoVVk_4XsimMFx7eASkt/"
    "pub?gid=0&single=true&output=csv"
)
FIN_PALETTE_SM = {
    "Suministro": "#0EA5A4",
    "I+D": "#6366F1",
    "Montaje": "#F59E0B",
}
FIN_GRID = "rgba(148,163,184,.25)"
GANTT_DATE_COL_START = "Inicio (AAAA-MM-DD)"
GANTT_DATE_COL_END_PLAN = "Fin plan (AAAA-MM-DD)"
GANTT_DATE_COL_END_REAL = "Fin real"

# =========================
# FUNCIONES
# =========================
@st.cache_data(show_spinner=True, ttl=120)
def load_capex_data(url: str, refresh_nonce: int = 0) -> pd.DataFrame:
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


@st.cache_data(show_spinner=True, ttl=120)
def load_capex_total_real_clp(url: str, refresh_nonce: int = 0) -> float | None:
    """
    Intenta obtener un total real en CLP directamente desde la hoja publicada de CAPEX.
    Se usa para KPIs que deben reflejar cambios del archivo fuente sin depender del
    total fijo del sidebar.
    """
    try:
        df_raw = pd.read_csv(url, dtype=str)
    except Exception:
        return None

    if df_raw.empty:
        return None

    df_raw.columns = [str(c).strip() for c in df_raw.columns]

    item_col = find_best_column(df_raw, ["item", "concepto", "descripcion"])
    if item_col:
        df_raw = df_raw[df_raw[item_col].astype(str).str.strip().ne("")].copy()

    candidate_names = [
        "montoclp",
        "montoenclp",
        "montototalclp",
        "costoclp",
        "costototalclp",
        "inversionclp",
        "presupuestoclp",
        "valorclp",
        "montochile",
        "pesoschilenos",
    ]

    selected_col = find_best_column(df_raw, candidate_names)
    if not selected_col:
        for col in df_raw.columns:
            norm = normalize_key(col)
            if "clp" not in norm:
                continue
            if any(token in norm for token in ["usd", "anticipo", "entrega", "sat", "mes", "porcentaje", "pct"]):
                continue
            selected_col = col
            break

    if not selected_col:
        return None

    total_clp = df_raw[selected_col].apply(parse_money_clp_robusto).sum()
    return float(total_clp) if np.isfinite(total_clp) and total_clp > 0 else None


def parse_money_clp_robusto(x: str) -> float:
    """Convierte montos CLP escritos con separadores latinos o mixtos a float."""
    if pd.isna(x):
        return 0.0
    s = str(x).strip()
    if not s:
        return 0.0

    neg = s.startswith("(") and s.endswith(")")
    if neg:
        s = s[1:-1]

    s = (s.replace("$", "")
           .replace("CLP", "")
           .replace(" ", "")
           .replace("\u00a0", ""))
    s = re.sub(r"[^0-9,.\-]", "", s)

    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        s = s.replace(".", "").replace(",", ".")
    else:
        if s.count(".") > 1:
            s = s.replace(".", "")
        elif "." in s:
            left, right = s.split(".")
            if len(right) == 3 and left.isdigit():
                s = left + right

    try:
        val = float(s)
        return -val if neg else val
    except ValueError:
        return 0.0


def build_google_sheet_xlsx_candidates(url: str) -> list[str]:
    """Genera candidatos de URL XLSX a partir de una URL publicada de Google Sheets."""
    parsed = urlparse(url)
    query_pairs = dict(parse_qsl(parsed.query, keep_blank_values=True))
    candidates = []

    if "docs.google.com" not in parsed.netloc:
        return [url]

    query_xlsx = dict(query_pairs)
    query_xlsx["output"] = "xlsx"
    candidates.append(urlunparse(parsed._replace(query=urlencode(query_xlsx))))

    query_pub = {"output": "xlsx"}
    candidates.append(urlunparse(parsed._replace(query=urlencode(query_pub))))

    if "/pub" in parsed.path:
        candidates.append(urlunparse(parsed._replace(path=parsed.path.replace("/pub", "/pub", 1), query="output=xlsx")))

    deduped = []
    for cand in candidates:
        if cand not in deduped:
            deduped.append(cand)
    return deduped


@st.cache_data(show_spinner=True, ttl=120)
def load_director_general_data(sheet_source_url: str, sheet_name: str = "Director General Técnico", refresh_nonce: int = 0) -> pd.DataFrame:
    """Carga la hoja de Dirección Técnica desde el mismo Google Sheet publicado."""
    last_error = None
    for candidate_url in build_google_sheet_xlsx_candidates(sheet_source_url):
        try:
            df_raw = pd.read_excel(candidate_url, sheet_name=sheet_name, dtype=str)
            df_raw.columns = [str(c).strip() for c in df_raw.columns]

            cols_map = {str(c).strip().lower(): c for c in df_raw.columns}
            cargo_col = cols_map.get("cargo")
            meses_col = cols_map.get("meses")
            costo_col = cols_map.get("costo empresa mensual")
            total_col = cols_map.get("total")
            inicio_col = (
                cols_map.get("mes_inicio")
                or cols_map.get("mes inicio")
                or cols_map.get("inicio")
                or cols_map.get("mes de inicio")
                or cols_map.get("inicio mes")
            )

            if not all([cargo_col, meses_col, costo_col, total_col]):
                continue

            df = pd.DataFrame({
                "Cargo": df_raw[cargo_col].astype(str).str.strip(),
                "Meses": pd.to_numeric(df_raw[meses_col], errors="coerce"),
                "Costo empresa mensual": df_raw[costo_col].apply(parse_money_clp_robusto),
                "Total": df_raw[total_col].apply(parse_money_clp_robusto),
                "Mes_inicio": pd.to_numeric(df_raw[inicio_col], errors="coerce") if inicio_col else 1,
            })
            df = df[df["Cargo"].notna() & (df["Cargo"] != "") & (df["Cargo"].str.lower() != "nan")].copy()
            df = df[(df["Meses"].notna()) | (df["Total"] > 0)].copy()
            df["Mes_inicio"] = df["Mes_inicio"].fillna(1).clip(lower=1)
            return df.reset_index(drop=True)
        except Exception as exc:
            last_error = exc

    raise ValueError(
        f"No se pudo leer la hoja '{sheet_name}' desde la publicación de Google Sheets. Último error: {last_error}"
    )


@st.cache_data(show_spinner=True, ttl=120)
def load_valorizacion_data(url: str, refresh_nonce: int = 0) -> pd.DataFrame:
    df = pd.read_csv(url, dtype=str)
    df.columns = [str(c).strip() for c in df.columns]
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].where(df[col].notna(), np.nan)
            df[col] = df[col].apply(lambda x: str(x).strip() if pd.notna(x) else np.nan)
            df[col] = df[col].replace({"": np.nan, "nan": np.nan, "None": np.nan})
    return df


@st.cache_data(show_spinner=True, ttl=120)
def load_valorizacion_raw_data(url: str, refresh_nonce: int = 0) -> pd.DataFrame:
    return pd.read_csv(url, dtype=str, header=None)


@st.cache_data(show_spinner=True, ttl=120)
def load_eerrv2_data(url: str, refresh_nonce: int = 0) -> pd.DataFrame:
    return pd.read_csv(url, dtype=str, header=None)


@st.cache_data(show_spinner=True, ttl=120)
def load_restante_piloto_10kw_raw_data(url: str, refresh_nonce: int = 0) -> pd.DataFrame:
    return pd.read_csv(url, dtype=str, header=None)


@st.cache_data(show_spinner=True, ttl=120)
def load_bullet_contexto_10kw_raw_data(url: str, refresh_nonce: int = 0) -> pd.DataFrame:
    return pd.read_csv(url, dtype=str, header=None)


def build_restante_piloto_10kw_view(url: str, refresh_nonce: int = 0) -> pd.DataFrame:
    df_raw = load_restante_piloto_10kw_raw_data(url, refresh_nonce=refresh_nonce)
    if df_raw.empty or df_raw.shape[1] < 3:
        return pd.DataFrame(columns=["Columna A", "Columna B", "Columna C", "Valor B", "Valor C"])

    df_view = df_raw.iloc[:, :3].copy()
    df_view.columns = ["Columna A", "Columna B", "Columna C"]
    for col in df_view.columns:
        df_view[col] = df_view[col].apply(clean_sheet_cell)

    df_view = df_view[
        df_view[["Columna A", "Columna B", "Columna C"]]
        .apply(lambda row: any(str(v).strip() for v in row), axis=1)
    ].reset_index(drop=True)

    df_view["Valor B"] = df_view["Columna B"].apply(parse_model_number)
    df_view["Valor C"] = df_view["Columna C"].apply(parse_money_clp_robusto)

    if not df_view.empty:
        first_row = df_view.iloc[0]
        has_numeric_below = (df_view["Valor B"].iloc[1:] > 0).any() or (df_view["Valor C"].iloc[1:] > 0).any()
        if has_numeric_below and first_row["Valor B"] == 0 and first_row["Valor C"] == 0:
            df_view = df_view.iloc[1:].reset_index(drop=True)

    return df_view


def build_bullet_contexto_10kw_sections(url: str, refresh_nonce: int = 0) -> tuple[str, list[dict]]:
    df_raw = load_bullet_contexto_10kw_raw_data(url, refresh_nonce=refresh_nonce)
    if df_raw.empty:
        return "", []

    title = ""
    sections: list[dict] = []
    current_section: dict | None = None

    for _, row in df_raw.fillna("").iterrows():
        a_val = clean_sheet_cell(row.iloc[0]) if len(row) > 0 else ""
        b_val = clean_sheet_cell(row.iloc[1]) if len(row) > 1 else ""

        if not title and a_val:
            title = a_val

        if not b_val:
            continue

        if re.match(r"^\d+\.\s+", b_val):
            current_section = {"title": b_val, "bullets": []}
            sections.append(current_section)
        elif current_section is not None:
            current_section["bullets"].append(b_val)

    return title, sections


def normalize_key(text: str) -> str:
    s = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def find_best_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    normalized_map = {normalize_key(col): col for col in df.columns}
    for candidate in candidates:
        if candidate in normalized_map:
            return normalized_map[candidate]
    return None


def build_valorizacion_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        serie = df[col]
        non_empty = serie.replace({"": np.nan, "nan": np.nan}).dropna()
        example = str(non_empty.iloc[0]) if not non_empty.empty else "-"
        rows.append(
            {
                "Campo": col,
                "Tipo visible": "Texto" if serie.dtype == object else str(serie.dtype),
                "Registros válidos": int(non_empty.shape[0]),
                "Vacíos": int(len(serie) - non_empty.shape[0]),
                "Ejemplo": example[:80],
            }
        )
    return pd.DataFrame(rows)


def clean_sheet_cell(value) -> str:
    if pd.isna(value):
        return ""
    txt = str(value).strip()
    txt = re.sub(r"\s*_\)$", "", txt)
    txt = txt.replace("_)","").strip()
    return txt


def format_compact_usd(value: float) -> str:
    value = float(value or 0.0)
    abs_value = abs(value)
    if abs_value >= 1_000_000:
        return f"US${value / 1_000_000:.2f}M"
    if abs_value >= 1_000:
        return f"US${value / 1_000:.1f}k"
    return format_usd(value)


def style_engineering_table(df: pd.DataFrame, header_color: str = "#2C5783", row_color: str = "#EAF6FF"):
    return (
        df.style
        .set_properties(**{
            "text-align": "center",
            "border": "1px solid rgba(203,213,225,.65)",
        })
        .set_properties(subset=[df.columns[0]], **{"text-align": "left"})
        .set_table_styles([
            {
                "selector": "th",
                "props": [
                    ("background-color", header_color),
                    ("color", "white"),
                    ("font-weight", "700"),
                    ("border", "1px solid rgba(203,213,225,.75)"),
                ],
            },
            {
                "selector": "td",
                "props": [
                    ("padding", "8px 10px"),
                    ("font-size", "14px"),
                ],
            },
        ])
        .apply(
            lambda row: [
                f"background-color: {row_color if row.name % 2 == 0 else '#FFFFFF'}"
                for _ in row
            ],
            axis=1,
        )
    )


def parse_model_number(value) -> float:
    if pd.isna(value):
        return 0.0
    s = str(value).strip()
    if not s:
        return 0.0
    s = s.replace("US$", "").replace("USD", "").replace("USS", "").replace("$", "").replace("x", "").replace("%", "")
    s = s.replace(" ", "").replace("\u00a0", "")
    s = re.sub(r"[^0-9,.\-]", "", s)
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        if s.count(",") == 1 and len(s.split(",")[-1]) <= 2:
            s = s.replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "." in s:
        if s.count(".") > 1:
            s = s.replace(".", "")
        else:
            left, right = s.split(".")
            if len(right) == 3 and left.replace("-", "").isdigit():
                s = left + right
    try:
        return float(s)
    except ValueError:
        return 0.0


def parse_model_percent(value) -> float:
    if pd.isna(value):
        return 0.0
    s = str(value).strip()
    if not s:
        return 0.0
    if "%" not in s and parse_model_number(s) <= 1:
        return parse_model_number(s)
    return parse_model_number(s) / 100.0


def get_valorizacion_model_map(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    label_col = df.columns[0]
    value_col = find_best_column(df, ["unnamed6", "valor", "value", "monto", "total"]) or (df.columns[6] if len(df.columns) > 6 else df.columns[-1])
    comment_col = find_best_column(df, ["unnamed7", "comentario", "comment"]) or (df.columns[7] if len(df.columns) > 7 else None)

    model_df = pd.DataFrame(
        {
            "Label": df[label_col],
            "Value": df[value_col] if value_col in df.columns else np.nan,
            "Comment": df[comment_col] if comment_col and comment_col in df.columns else np.nan,
        }
    ).copy()
    model_df["Label"] = model_df["Label"].where(model_df["Label"].notna(), np.nan)
    model_df["Label"] = model_df["Label"].apply(lambda x: str(x).strip() if pd.notna(x) else np.nan)
    model_df["Label"] = model_df["Label"].replace({"": np.nan, "nan": np.nan})
    model_df = model_df.dropna(subset=["Label"]).reset_index(drop=True)
    model_map = {normalize_key(row["Label"]): row["Value"] for _, row in model_df.iterrows()}
    return model_df, model_map


def get_first_model_value(model_map: dict, candidates: list[str], default=0.0) -> float:
    for candidate in candidates:
        key = normalize_key(candidate)
        if key in model_map:
            return parse_model_number(model_map.get(key))
    return float(default)


def build_direccion_mensual(df_dir: pd.DataFrame, horizonte_meses: int = 15) -> pd.DataFrame:
    """Expande la hoja de dirección a una serie mensual respetando mes de inicio y duración."""
    if df_dir is None or df_dir.empty:
        return pd.DataFrame(columns=["Mes", "Cargo", "Pago_CLP"])

    rows = []
    for _, row in df_dir.iterrows():
        cargo = str(row.get("Cargo", "")).strip() or "Sin cargo"
        meses = int(pd.to_numeric(row.get("Meses"), errors="coerce") or 0)
        mes_inicio = int(pd.to_numeric(row.get("Mes_inicio"), errors="coerce") or 1)
        costo_mensual = float(pd.to_numeric(row.get("Costo empresa mensual"), errors="coerce") or 0.0)

        if meses <= 0 or costo_mensual <= 0:
            continue

        mes_fin = min(horizonte_meses, mes_inicio + meses - 1)
        for mes in range(mes_inicio, mes_fin + 1):
            rows.append({
                "Mes": mes,
                "Cargo": cargo,
                "Pago_CLP": costo_mensual,
            })

    return pd.DataFrame(rows)


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


def parse_money_mixed_robusto(x) -> float:
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null", "-", "s/n"}:
        return np.nan

    neg = s.startswith("(") and s.endswith(")")
    if neg:
        s = s[1:-1]

    s = (
        s.replace("$", "")
        .replace("CLP", "")
        .replace("USD", "")
        .replace("US$", "")
        .replace(" ", "")
        .replace("\u00a0", "")
    )

    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        s = s.replace(".", "").replace(",", ".")
    else:
        if s.count(".") > 1:
            s = s.replace(".", "")
        elif "." in s:
            left, right = s.split(".")
            if len(right) == 3 and left.replace("-", "").isdigit():
                s = left + right

    s = re.sub(r"[^0-9.\-]", "", s)
    try:
        num = float(s)
        return -num if neg else num
    except ValueError:
        return np.nan


@st.cache_data(show_spinner=True, ttl=120)
def load_dashboard_financiero_data(url: str, refresh_nonce: int = 0) -> pd.DataFrame:
    df = pd.read_csv(url, dtype=str)
    df.columns = [str(c).strip() for c in df.columns]

    rename_map = {
        "Proveedor": "Provedor",
        "Descripcion": "Descripciónn",
        "Descripción": "Descripciónn",
        "Suministro/montaje": "Suministro / montaje",
        "Boleta/fac": "Boleta / fac",
        "Num OC": "N° OC",
        "Dif_T": "diF-T",
        "Porc DI-T": "% DI-T",
        "Dias de Proyecto": "Dias de proyecto",
    }
    for src, dst in rename_map.items():
        if src in df.columns and dst not in df.columns:
            df.rename(columns={src: dst}, inplace=True)

    expected_text = [
        "Etapa", "Estado de pago", "Provedor", "item", "Sub-item", "Descripciónn",
        "Suministro / montaje", "Material", "Uni", "Centro de costo", "Observación",
        "Factor de costo", "Estado de costo", "Justificación % e E.E",
        "Tributa la HIBRIDA", "Tributa la DARRIEUS", "N° OC", "Boleta / fac",
        "Situación factura", "Forma de pago",
    ]
    expected_nums = [
        "Monto", "Dif-1", "Dif-2", "diF-T", "% DI-T", "Dias de proyecto",
        "Descuento ec escala", "Precio final ec esc", "ID-elemento",
    ]

    for col in expected_text + expected_nums:
        if col not in df.columns:
            df[col] = np.nan

    for col in expected_nums:
        df[col] = df[col].apply(parse_money_mixed_robusto)

    return df


def render_inputs_financial_main_kpis(df_in: pd.DataFrame):
    import html

    monto_series = df_in.get("Monto", pd.Series(dtype=float))
    monto_total = float(monto_series.sum(skipna=True) or 0.0)
    monto_prom = float(monto_series.mean(skipna=True) or 0.0)
    n_items = int(monto_series.notna().sum())
    prov_col = "Provedor" if "Provedor" in df_in.columns else ("Proveedor" if "Proveedor" in df_in.columns else None)
    n_prov = int(df_in[prov_col].dropna().astype(str).str.strip().replace({"": np.nan, "nan": np.nan}).nunique()) if prov_col else 0
    capacidades_externo = 0.0
    know_how_fw = 0.0
    try:
        df_val_raw = load_valorizacion_raw_data(VALORIZACION_CSV_URL_DEFAULT, refresh_nonce=data_refresh_nonce)
        if df_val_raw.shape[0] > 6 and df_val_raw.shape[1] > 6:
            capacidades_externo = float(parse_money_clp_robusto(clean_sheet_cell(df_val_raw.iloc[5, 6])) or 0.0)
            know_how_fw = float(parse_money_clp_robusto(clean_sheet_cell(df_val_raw.iloc[6, 6])) or 0.0)
    except Exception:
        capacidades_externo = 0.0
        know_how_fw = 0.0

    fin_nav_key = "inputs_financiero_asset_sel"
    if fin_nav_key not in st.session_state:
        st.session_state[fin_nav_key] = None

    st.markdown(
        """
        <style>
        .inputs-fin-summary{display:grid;grid-template-columns:1.45fr .85fr .85fr;gap:16px;margin:10px 0 18px}
        @media (max-width:1400px){.inputs-fin-summary{grid-template-columns:1fr;}}
        .inputs-fin-hero,
        .inputs-fin-side{
            border-radius:20px;
            padding:18px 18px 16px 18px;
            background:linear-gradient(180deg,#f8fafc 0%,#ffffff 68%);
            border:1px solid rgba(148,163,184,.30);
            box-shadow:0 8px 18px rgba(15,23,42,.06);
        }
        .inputs-fin-hero{
            background:linear-gradient(90deg,#EFF8FF 0%,#DFF4FF 42%,#C6ECFF 100%);
        }
        .inputs-fin-row{display:flex;align-items:center;gap:10px;margin-bottom:10px}
        .inputs-fin-ico{
            width:36px;height:36px;border-radius:999px;display:inline-flex;align-items:center;justify-content:center;
            background:#ecfeff;border:1px solid rgba(14,165,164,.25);font-size:22px
        }
        .inputs-fin-h{font-size:13px;font-weight:800;color:#0f172a;letter-spacing:.02em}
        .inputs-fin-v{font-size:28px;font-weight:900;color:#0f172a;line-height:1.05;margin-bottom:8px}
        .inputs-fin-hero .inputs-fin-v{font-size:36px}
        .inputs-fin-sub{display:flex;gap:8px;flex-wrap:wrap}
        .inputs-fin-chip{
            display:inline-block;font-size:12px;padding:5px 10px;border-radius:999px;
            border:1px solid rgba(165,180,252,.45);background:#eef2ff;color:#3730a3
        }
        .inputs-fin-note{font-size:13px;line-height:1.5;color:#475569;margin-top:6px}
        .inputs-fin-selector-row{
            display:grid;
            grid-template-columns:1.45fr .85fr .85fr;
            gap:16px;
            margin:0 0 18px 0;
        }
        @media (max-width:1400px){.inputs-fin-selector-row{grid-template-columns:1fr;}}
        </style>
        """,
        unsafe_allow_html=True,
    )

    cards = f"""
    <div class="inputs-fin-summary">
      <div class="inputs-fin-hero">
        <div class="inputs-fin-row"><div class="inputs-fin-ico">💰</div><div class="inputs-fin-h">Costo Ejecutado</div></div>
        <div class="inputs-fin-v">{html.escape(format_clp(monto_total))}</div>
        <div class="inputs-fin-sub">
          <span class="inputs-fin-chip">Base: {n_items:,} ítems</span>
          <span class="inputs-fin-chip">Proveedores: {n_prov:,}</span>
        </div>
        <div class="inputs-fin-note">Inversión efectivamente ejecutada para construir y poner en forma operativa el activo tecnológico.</div>
      </div>
      <div class="inputs-fin-side">
        <div class="inputs-fin-row"><div class="inputs-fin-ico">🧠</div><div class="inputs-fin-h">Capacidades externo</div></div>
        <div class="inputs-fin-v">{html.escape(format_clp(capacidades_externo))}</div>
        <div class="inputs-fin-sub"><span class="inputs-fin-chip">Valorización FW · G6</span></div>
        <div class="inputs-fin-note">Capacidades complementarias valorizadas fuera del gasto ejecutado directo.</div>
      </div>
      <div class="inputs-fin-side">
        <div class="inputs-fin-row"><div class="inputs-fin-ico">⚙️</div><div class="inputs-fin-h">Know-how FW</div></div>
        <div class="inputs-fin-v">{html.escape(format_clp(know_how_fw))}</div>
        <div class="inputs-fin-sub"><span class="inputs-fin-chip">Valorización FW · G7</span></div>
        <div class="inputs-fin-note">Valor del conocimiento técnico incorporado en la arquitectura y desarrollo del activo.</div>
      </div>
    </div>
    """
    st.markdown(cards, unsafe_allow_html=True)

    st.markdown('<div class="inputs-fin-selector-row">', unsafe_allow_html=True)
    selector_cards = [
        ("costo_ejecutado", "Costo Ejecutado"),
        ("capacidades_externas", "Capacidades externo"),
        ("know_how_fw", "Know-how FW"),
    ]
    selector_cols = st.columns([1.45, 0.85, 0.85])
    for idx, (value, _label) in enumerate(selector_cards):
        is_active = st.session_state.get(fin_nav_key) == value
        with selector_cols[idx]:
            st.button(
                "Seleccionado" if is_active else "Abrir bloque",
                key=f"inputs_fin_asset_selector_{idx}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
                on_click=lambda selected=value: st.session_state.__setitem__(fin_nav_key, selected),
            )
    st.markdown("</div>", unsafe_allow_html=True)
    return {
        "monto_total": monto_total,
        "capacidades_externo": capacidades_externo,
        "know_how_fw": know_how_fw,
        "selected": st.session_state.get(fin_nav_key),
    }


def make_inputs_suministro_chart(df_in: pd.DataFrame):
    df = df_in.copy()
    if "Suministro / montaje" not in df.columns:
        return None, None
    base = df[
        df["Suministro / montaje"].notna()
        & (df["Suministro / montaje"].astype(str).str.strip() != "")
        & df["Monto"].notna()
    ].copy()
    if base.empty:
        return None, None

    base["Categoria"] = base["Suministro / montaje"].astype(str).str.strip()
    agg = (
        base.groupby("Categoria", as_index=False)
        .agg(Monto=("Monto", "sum"), Items=("Monto", "count"))
    )
    if agg.empty:
        return None, None

    total = float(agg["Monto"].sum() or 0.0)
    agg["% del total"] = np.where(total > 0, agg["Monto"] / total * 100.0, 0.0)
    agg["Monto_MM"] = agg["Monto"] / 1_000_000.0
    orden = ["Suministro", "I+D", "Montaje"]
    agg["__ord"] = agg["Categoria"].apply(lambda x: orden.index(x) if x in orden else 999)
    agg = agg.sort_values(["__ord", "Monto"], ascending=[True, False]).drop(columns="__ord")
    agg["label_pct"] = agg.apply(lambda r: f"{r['% del total']:.2f}% · {r['Monto_MM']:.1f} MM", axis=1)

    fig = px.bar(
        agg,
        x="Monto_MM",
        y="Categoria",
        orientation="h",
        color="Categoria",
        color_discrete_map=FIN_PALETTE_SM,
        text="label_pct",
        title="Distribución técnica por frente de ejecución",
    )
    fig.update_traces(
        textposition="inside",
        insidetextanchor="middle",
        marker=dict(line=dict(color="rgba(255,255,255,0.85)", width=1.2)),
        hovertemplate=(
            "<b>%{y}</b><br>Monto: $%{customdata[2]:,.0f}<br>% del total: %{customdata[0]:.2f}%"
            "<br>N° ítems: %{customdata[1]}<extra></extra>"
        ),
        customdata=np.stack([agg["% del total"], agg["Items"], agg["Monto"]], axis=-1),
    )
    fig.update_layout(
        xaxis_title="Monto ejecutado (MM CLP)",
        yaxis_title="Frente técnico",
        margin=dict(l=12, r=12, t=64, b=18),
        legend_title="S/M",
        height=360,
        bargap=0.28,
        plot_bgcolor="white",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#334155", size=13),
        title=dict(font=dict(size=22, color="#0f172a"), x=0.02),
    )
    fig.update_xaxes(
        ticksuffix=" MM",
        showgrid=True,
        gridcolor="rgba(148,163,184,0.22)",
        zeroline=False,
    )
    fig.update_yaxes(showgrid=False)
    return fig, agg


def render_inputs_sm_kpi_cards(tabla_sm: pd.DataFrame):
    if tabla_sm is None or tabla_sm.empty:
        return
    st.markdown(
        """
        <style>
        .inputs-smkpi-grid{display:grid;grid-template-columns:repeat(3,minmax(220px,1fr));gap:12px;margin:0 0 12px}
        @media (max-width:1000px){.inputs-smkpi-grid{grid-template-columns:1fr;}}
        .inputs-smkpi-card{
            border-radius:16px;padding:12px 14px;background:linear-gradient(180deg,#f8fafc 0%,#ffffff 62%);
            border:1px solid rgba(148,163,184,.28);box-shadow:0 4px 10px rgba(15,23,42,.04)
        }
        .inputs-smkpi-title{font-size:13px;font-weight:800;color:#0f172a;margin-bottom:4px;text-transform:uppercase;letter-spacing:.05em}
        .inputs-smkpi-value{font-size:22px;font-weight:800;color:#0f172a;margin:2px 0 6px}
        .inputs-smkpi-row{font-size:12px;color:#475569;display:flex;gap:6px;flex-wrap:wrap}
        .inputs-smkpi-chip{display:inline-block;padding:2px 8px;border-radius:999px;font-size:12px;border:1px solid rgba(148,163,184,.35);background:#f1f5f9;color:#0f172a}
        .inputs-smkpi-bar{position:relative;width:100%;height:6px;border-radius:999px;background:#eef2f7;margin-top:8px}
        .inputs-smkpi-bar>span{position:absolute;left:0;top:0;height:100%;border-radius:999px}
        </style>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns(min(3, len(tabla_sm)))
    for idx, (_, row) in enumerate(tabla_sm.iterrows()):
        cat = str(row["Categoria"])
        color = FIN_PALETTE_SM.get(cat, "#0E9F6E")
        with cols[idx % len(cols)]:
            st.markdown(
                f"""
                <div class="inputs-smkpi-card">
                  <div class="inputs-smkpi-title">{cat}</div>
                  <div class="inputs-smkpi-value">{format_clp(float(row['Monto']))}</div>
                  <div class="inputs-smkpi-row">
                    <span class="inputs-smkpi-chip">% del total: {float(row['% del total']):.2f}%</span>
                    <span class="inputs-smkpi-chip">Ítems: {int(row['Items'])}</span>
                  </div>
                  <div class="inputs-smkpi-bar"><span style="width:{float(row['% del total']):.6f}%;background:{color};"></span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_inputs_cat_summary_pills(df2: pd.DataFrame, cat_col: str, key_prefix: str = "inputs_fin"):
    agg = (
        df2.groupby(cat_col, as_index=False)
        .agg(Monto=("Monto", "sum"), Items=("Monto", "count"))
    )
    if agg.empty:
        return
    total = float(agg["Monto"].sum() or 0.0)
    agg["pct"] = np.where(total > 0, agg["Monto"] / total * 100.0, 0.0)
    st.markdown(
        """
        <style>
        .inputs-pill-wrap{display:flex;gap:8px;flex-wrap:wrap;margin:6px 0 10px}
        .inputs-pill{
            display:inline-flex;align-items:center;gap:8px;padding:6px 10px;border-radius:999px;
            border:1px solid rgba(148,163,184,.35);background:#fff;box-shadow:0 1px 2px rgba(15,23,42,.05);
            font-size:12px;color:#0f172a
        }
        .inputs-pill-dot{width:10px;height:10px;border-radius:999px}
        .inputs-pill-sub{opacity:.75}
        </style>
        """,
        unsafe_allow_html=True,
    )
    pills = []
    orden = ["Suministro", "I+D", "Montaje"]
    agg["__ord"] = agg[cat_col].apply(lambda x: orden.index(x) if x in orden else 999)
    agg = agg.sort_values(["__ord", "Monto"], ascending=[True, False]).drop(columns="__ord")
    for _, row in agg.iterrows():
        color = FIN_PALETTE_SM.get(str(row[cat_col]), "#334155")
        pills.append(
            f"""<div class="inputs-pill">
                  <span class="inputs-pill-dot" style="background:{color}"></span>
                  <strong>{row[cat_col]}</strong>
                  <span class="inputs-pill-sub">— {format_clp(float(row['Monto']))} ({float(row['pct']):.2f}%) · {int(row['Items'])} ítems</span>
                </div>"""
        )
    st.markdown(f'<div class="inputs-pill-wrap">{"".join(pills)}</div>', unsafe_allow_html=True)


def render_inputs_item_analytics(df_in: pd.DataFrame):
    df = df_in.copy()
    if "item" not in df.columns:
        st.info("La fuente no contiene columna `item` para construir el análisis por categoría.")
        return

    item_col = "item"
    subitem_col = "Sub-item" if "Sub-item" in df.columns else None
    cat_col = "Suministro / montaje" if "Suministro / montaje" in df.columns else None
    if cat_col is None:
        st.info("La fuente no contiene la columna `Suministro / montaje`.")
        return

    df = df[df["Monto"].notna()].copy()
    df[item_col] = df[item_col].astype(str).str.strip().replace({"": np.nan, "nan": np.nan}).fillna("(Vacío)")
    df[cat_col] = df[cat_col].astype(str).str.strip().replace({"": np.nan, "nan": np.nan}).fillna("(Sin categoría)")

    st.markdown("### Desglose de Componentes de Inversión")
    c1, c2 = st.columns([1, 1])
    with c1:
        top_n = st.selectbox("Top N por monto", [5, 10, 15, 20, 30], index=2, key="inputs_fin_top_n")
    with c2:
        modo = st.radio(
            "Visualizar",
            ["Monto CLP", "% por Item (100%)"],
            index=0,
            horizontal=True,
            key="inputs_fin_modo",
        )

    cats_all = sorted(df[cat_col].dropna().astype(str).str.strip().unique().tolist())
    default_cats = st.session_state.get("inputs_fin_cats_sel", cats_all)
    cats_sel = st.multiselect(
        "Categorías S/M",
        cats_all,
        default=default_cats,
        key="inputs_fin_cats_sel",
    )
    render_inputs_cat_summary_pills(df, cat_col)

    df2 = df[df[cat_col].isin(cats_sel)].copy() if cats_sel else df.iloc[0:0].copy()
    if df2.empty:
        st.info("No hay datos para las categorías seleccionadas.")
        return

    resumen_item = (
        df2.groupby(item_col, as_index=False)
        .agg(Monto=("Monto", "sum"), Items=("Monto", "count"), Promedio=("Monto", "mean"))
    )
    resumen_item["% del total"] = np.where(
        float(resumen_item["Monto"].sum() or 0.0) > 0,
        resumen_item["Monto"] / float(resumen_item["Monto"].sum()) * 100.0,
        0.0,
    )
    resumen_item["Promedio"] = resumen_item["Promedio"].round(0)
    resumen_item = resumen_item.sort_values("Monto", ascending=False)
    tabla_show = resumen_item.head(top_n).copy()

    if tabla_show.empty:
        st.info("No hay ítems para mostrar.")
        return

    render_inputs_item_kpi_cards(tabla_show, item_col)

    items_keep = tabla_show[item_col].tolist()
    pivot = (
        df2[df2[item_col].isin(items_keep)]
        .pivot_table(index=item_col, columns=cat_col, values="Monto", aggfunc="sum", fill_value=0.0)
    )

    if not pivot.empty:
        if modo == "Monto CLP":
            plot_df = pivot.reset_index().melt(id_vars=item_col, var_name="Categoría", value_name="Monto")
            fig = px.bar(
                plot_df,
                x="Monto",
                y=item_col,
                color="Categoría",
                orientation="h",
                color_discrete_map=FIN_PALETTE_SM,
                title="Items — Monto por categoría S/M",
            )
            fig.update_traces(hovertemplate="<b>%{y}</b><br>%{trace.name}: $%{x:,.0f}<extra></extra>")
            fig.update_xaxes(separatethousands=True)
        else:
            row_sums = pivot.sum(axis=1).replace(0, np.nan)
            pct = pivot.div(row_sums, axis=0) * 100.0
            plot_df = pct.reset_index().melt(id_vars=item_col, var_name="Categoría", value_name="%")
            fig = px.bar(
                plot_df,
                x="%",
                y=item_col,
                color="Categoría",
                orientation="h",
                color_discrete_map=FIN_PALETTE_SM,
                title="Top Items — % por categoría S/M (100%)",
            )
            fig.update_traces(hovertemplate="<b>%{y}</b><br>%{trace.name}: %{x:.2f}%<extra></extra>")
            fig.update_xaxes(range=[0, 100])

        fig.update_layout(margin=dict(l=80, r=40, t=60, b=40), legend_title="S/M")
        st.plotly_chart(fig, use_container_width=True)


def render_inputs_item_kpi_cards(tabla_show: pd.DataFrame, item_col: str):
    st.markdown(
        """
        <style>
        .inputs-item-card{
            box-sizing:border-box;border-radius:16px;padding:14px 16px;background:linear-gradient(180deg,#f8fafc 0%,#ffffff 62%);
            border:1px solid rgba(148,163,184,.35);box-shadow:0 4px 10px rgba(15,23,42,.05)
        }
        .inputs-item-title{font-size:14px;font-weight:700;color:#0f172a;margin-bottom:4px}
        .inputs-item-value{font-size:26px;font-weight:800;color:#0f172a;margin-top:4px}
        .inputs-item-sub{font-size:13px;color:#475569;margin-top:8px}
        .inputs-item-chip{
            display:inline-block;padding:2px 8px;border-radius:999px;font-size:12px;margin-right:6px;
            border:1px solid rgba(148,163,184,.35);background:#f1f5f9;color:#0f172a
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    items = list(tabla_show.iterrows())
    for start in range(0, len(items), 3):
        cols = st.columns(min(3, len(items) - start))
        for col, (_, rec) in zip(cols, items[start:start + 3]):
            with col:
                st.markdown(
                    f"""
                    <div class="inputs-item-card">
                      <div class="inputs-item-title">{rec[item_col]}</div>
                      <div class="inputs-item-value">{format_clp(float(rec['Monto']))}</div>
                      <div class="inputs-item-sub">
                        <span class="inputs-item-chip">% del total: {float(rec['% del total']):.2f}%</span>
                        <span class="inputs-item-chip">Ítems: {int(rec['Items'])}</span>
                        <span class="inputs-item-chip">Prom: {format_clp(float(rec['Promedio']))}</span>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def render_inputs_factor_chart(df_in: pd.DataFrame):
    if "Factor de costo" not in df_in.columns or "Suministro / montaje" not in df_in.columns:
        return
    fac_cat = (
        df_in.copy()
        .assign(
            Factor=df_in["Factor de costo"].fillna("Sin clasificar").astype(str).str.strip(),
            Categoria=df_in["Suministro / montaje"].fillna("(Sin categoría)").astype(str).str.strip(),
            Monto_num=df_in["Monto"],
            PrecioEsc=df_in["Precio final ec esc"] if "Precio final ec esc" in df_in.columns else np.nan,
        )
    )
    fac_cat = fac_cat[fac_cat["Factor"].str.lower() != "sin clasificar"].copy()
    fac_cat["Categoria"] = fac_cat["Categoria"].replace({"Suministro/montaje": "Montaje", "I + D": "I+D", "i+d": "I+D"})
    if fac_cat.empty:
        return

    agg_fc = (
        fac_cat.groupby(["Factor", "Categoria"], as_index=False)
        .agg(Monto=("Monto_num", "sum"), PrecioEsc=("PrecioEsc", "sum"), Items=("Monto_num", "count"))
    )
    if agg_fc.empty:
        return

    tot_por_factor = agg_fc.groupby("Factor")["Monto"].transform("sum")
    agg_fc["% dentro del Factor"] = np.where(tot_por_factor > 0, agg_fc["Monto"] / tot_por_factor * 100.0, 0.0)
    agg_fc["%_esc"] = np.where(agg_fc["Monto"] > 0, agg_fc["PrecioEsc"] / agg_fc["Monto"] * 100.0, np.nan)

    orden_factor = ["Necesario", "Evitable"]
    orden_cat = ["Suministro", "I+D", "Montaje"]
    agg_fc["__of"] = agg_fc["Factor"].apply(lambda x: orden_factor.index(x) if x in orden_factor else 999)
    agg_fc["__oc"] = agg_fc["Categoria"].apply(lambda x: orden_cat.index(x) if x in orden_cat else 999)
    agg_fc = agg_fc.sort_values(["__of", "__oc"]).drop(columns=["__of", "__oc"])
    agg_fc["label"] = agg_fc.apply(lambda r: f"{format_clp(float(r['Monto']))} · {float(r['% dentro del Factor']):.2f}%", axis=1)

    st.markdown("### Eficiencia del CAPEX Ejecutado")
    fig_fc = px.bar(
        agg_fc,
        x="Monto",
        y="Factor",
        orientation="h",
        color="Categoria",
        color_discrete_map=FIN_PALETTE_SM,
        text="label",
        title="Montos necesarios para un nuevo piloto",
    )
    fig_fc.update_traces(
        textposition="inside",
        insidetextanchor="middle",
        customdata=np.stack([agg_fc["% dentro del Factor"], agg_fc["%_esc"], agg_fc["Items"]], axis=-1),
        hovertemplate=(
            "<b>%{y}</b> · %{trace.name}<br>"
            "Monto seg.: $%{x:,.0f}<br>"
            "Participación en Factor: %{customdata[0]:.2f}%<br>"
            "%_esc seg.: %{customdata[1]:.2f}%<br>"
            "Ítems seg.: %{customdata[2]}<extra></extra>"
        ),
    )
    fig_fc.update_layout(
        barmode="stack",
        margin=dict(l=130, r=40, t=60, b=40),
        plot_bgcolor="white",
        paper_bgcolor="rgba(0,0,0,0)",
        legend_title="Categoría (S/M/I+D)",
    )
    fig_fc.update_xaxes(separatethousands=True, tickprefix="$", gridcolor=FIN_GRID)
    fig_fc.update_yaxes(title="Factor", showgrid=False)
    st.plotly_chart(fig_fc, use_container_width=True)


def gantt_infer_piloto(row):
    val = str(row.get("Piloto", "")).strip()
    if val:
        return val
    texto = " ".join([
        str(row.get("Fase", "")),
        str(row.get("Línea", "")),
        str(row.get("Tarea / Entregable", "")),
        str(row.get("Método", "")),
    ]).lower()
    if "55" in texto or "55kw" in texto or "55 k" in texto:
        return "Piloto 55 kW"
    return "Piloto 10 kW"


def gantt_process_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "ID" in df.columns:
        df["ID"] = pd.to_numeric(df["ID"], errors="coerce").astype("Int64")
    for col in [GANTT_DATE_COL_START, GANTT_DATE_COL_END_PLAN, GANTT_DATE_COL_END_REAL]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
        else:
            df[col] = pd.NaT
    if "%" in df.columns:
        df["%"] = pd.to_numeric(df["%"], errors="coerce")
        try:
            max_val = df["%"].max()
            if pd.notna(max_val) and max_val <= 1:
                df["%"] = df["%"] * 100
        except Exception:
            pass
        if "Estado" in df.columns:
            mask_done = df["Estado"].astype(str).str.contains("complet", case=False, na=False) & df["%"].isna()
            df.loc[mask_done, "%"] = 100
    df["Piloto"] = df.apply(gantt_infer_piloto, axis=1)
    return df


@st.cache_data(show_spinner=True, ttl=300)
def load_project_gantt_data(url: str, refresh_nonce: int = 0) -> pd.DataFrame:
    df = pd.read_csv(url, encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]
    return gantt_process_df(df)


def build_inputs_gantt_figure(df: pd.DataFrame, date_mode: str = "Real", color_by: str = "Estado"):
    dfp = df.copy()
    dfp["_start"] = pd.to_datetime(dfp.get(GANTT_DATE_COL_START), errors="coerce")
    dfp["_end_plan"] = pd.to_datetime(dfp.get(GANTT_DATE_COL_END_PLAN), errors="coerce")
    dfp["_end_real"] = pd.to_datetime(dfp.get(GANTT_DATE_COL_END_REAL), errors="coerce")

    dfp["_start"] = dfp["_start"].fillna(dfp["_end_real"]).fillna(dfp["_end_plan"])
    dfp["_end"] = dfp["_end_real"] if date_mode == "Real" else dfp["_end_plan"]
    dfp["_end"] = dfp["_end"].fillna(dfp["_end_plan"]).fillna(dfp["_end_real"]).fillna(dfp["_start"])
    bad = dfp["_end"] <= dfp["_start"]
    dfp.loc[bad, "_end"] = dfp.loc[bad, "_start"] + pd.Timedelta(days=1)
    dfp = dfp[dfp["_start"].notna() & dfp["_end"].notna()].copy()
    if dfp.empty:
        return go.Figure()

    dfp = dfp.sort_values(["_start", "_end", "ID"], ascending=[False, False, True])
    y_labels = pd.unique(dfp["Tarea / Entregable"].astype(str))
    max_len = max((len(s) for s in y_labels), default=12)
    rows = len(y_labels)
    left_margin = min(56 + max_len * 6, 380)
    height_px = min(max(460, 28 * rows), 1200)

    apple = {
        "Planificado": "#0A84FF",
        "En curso": "#FF3B30",
        "Completado": "#30D158",
        "Pendiente": "#FF9F0A",
        "Recurrente": "#C5C213",
        "default": "#8E8E93",
    }

    def color_estado(s):
        s = (s or "").strip().lower()
        if "plan" in s:
            return apple["Planificado"]
        if "curso" in s:
            return apple["En curso"]
        if "complet" in s:
            return apple["Completado"]
        if "pend" in s:
            return apple["Pendiente"]
        if "recurrente" in s:
            return apple["Recurrente"]
        return apple["default"]

    color_arg = color_by if color_by in dfp.columns else "Estado"
    color_map = None
    if color_by == "Estado" and "Estado" in dfp.columns:
        color_map = {val: color_estado(val) for val in dfp["Estado"].dropna().unique()}
    elif color_by == "Piloto" and "Piloto" in dfp.columns:
        pilotos = dfp["Piloto"].dropna().unique()
        color_map = {p: PX_COLORS[i % len(PX_COLORS)] for i, p in enumerate(pilotos)}

    if GANTT_DATE_COL_END_REAL in dfp.columns:
        dfp["_label_barra"] = dfp[GANTT_DATE_COL_END_REAL].dt.strftime("%d-%m-%Y")
    else:
        dfp["_label_barra"] = ""

    fig = px.timeline(
        dfp,
        x_start="_start",
        x_end="_end",
        y="Tarea / Entregable",
        color=color_arg,
        color_discrete_map=color_map,
        category_orders={"Tarea / Entregable": y_labels},
        text="_label_barra",
        opacity=0.95,
    )
    fig.update_traces(
        texttemplate="%{text}",
        textposition="inside",
        insidetextanchor="middle",
        selector=dict(type="bar"),
    )

    safe_pct = pd.to_numeric(dfp.get("%", 0), errors="coerce").fillna(0).astype(float)
    fig.update_traces(
        customdata=np.stack([
            dfp.get("Fase", pd.Series([""] * len(dfp))).astype(str),
            dfp.get("Estado", pd.Series([""] * len(dfp))).astype(str),
            dfp.get("ID", pd.Series([""] * len(dfp))),
            (dfp["_end"] - dfp["_start"]).dt.days.clip(lower=1).astype(int),
            safe_pct.round(0),
        ], axis=1),
        hovertemplate="<b>%{y}</b><br>"
                      "Estado: %{customdata[1]} · ID: %{customdata[2]}<br>"
                      "Inicio: %{base|%Y-%m-%d}<br>"
                      "Fin: %{x|%Y-%m-%d}<extra>Duración: %{customdata[3]} días · Avance: %{customdata[4]}%</extra>",
    )

    today_ts = pd.Timestamp.today().normalize()
    today_iso = today_ts.strftime("%Y-%m-%d")
    fig.add_shape(
        type="line",
        x0=today_iso, x1=today_iso, y0=0, y1=1, xref="x", yref="paper",
        line=dict(dash="dot", width=2, color="#1C1C1E"),
    )
    fig.add_annotation(
        x=today_iso, y=1.02, xref="x", yref="paper",
        text="Hoy", showarrow=False,
        font=dict(size=12, color="#1C1C1E"),
        bgcolor="rgba(255,255,255,0.7)",
    )

    if "Hito (S/N)" in dfp.columns:
        hitos = dfp[dfp["Hito (S/N)"].astype(str).str.upper().eq("S")].copy()
        if not hitos.empty:
            fig.add_trace(
                go.Scatter(
                    x=hitos["_end"],
                    y=hitos["Tarea / Entregable"],
                    mode="markers",
                    marker=dict(size=10, symbol="diamond", line=dict(width=1, color="#1C1C1E")),
                    name="Hito",
                    hovertext=hitos["Tarea / Entregable"],
                    hoverinfo="text",
                )
            )

    fig.update_xaxes(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=3, label="3m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(step="year", stepmode="todate", label="YTD"),
                dict(count=1, step="year", stepmode="backward", label="1y"),
                dict(step="all", label="All"),
            ])
        ),
        rangeslider=dict(visible=True),
        showgrid=True,
        gridcolor="rgba(60,60,67,0.08)",
    )
    fig.update_yaxes(autorange="reversed", automargin=True, showticklabels=True)
    fig.update_layout(
        height=height_px,
        margin=dict(l=left_margin, r=32, t=36, b=18),
        plot_bgcolor="#FBFBFD",
        paper_bgcolor="#FFFFFF",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.6)",
            borderwidth=0,
        ),
        xaxis_title=None,
        yaxis_title=None,
    )
    return fig


def render_inputs_project_gantt():
    try:
        df_gantt = load_project_gantt_data(GANTT_PROJECT_CSV_URL_DEFAULT, refresh_nonce=data_refresh_nonce)
    except Exception as exc:
        st.warning(f"No se pudo cargar la Carta Gantt del proyecto: {exc}")
        return

    if df_gantt.empty or "Tarea / Entregable" not in df_gantt.columns:
        return

    st.markdown("### Cronograma de Ejecución y Validación")
    st.caption("Vista integrada del cronograma del proyecto antes del análisis financiero por categorías.")

    c1, c2, c3 = st.columns([5, 2, 2])
    with c1:
        fases = sorted(
            df_gantt["Fase"].astype(str).str.strip().replace({"nan": np.nan, "None": np.nan, "": np.nan}).dropna().unique().tolist()
        ) if "Fase" in df_gantt.columns else []
        fase_options = ["Todas"] + fases
        fase_default = "Instalación Turbina" if "Instalación Turbina" in fases else "Todas"
        if "inputs_gantt_fase" not in st.session_state:
            st.session_state["inputs_gantt_fase"] = fase_default
        fase_sel = st.selectbox(
            "Fase",
            fase_options,
            index=fase_options.index(st.session_state.get("inputs_gantt_fase", fase_default)),
            key="inputs_gantt_fase",
        )
    with c2:
        date_mode = st.radio(
            "Fechas",
            ["Plan", "Real"],
            index=1,
            horizontal=True,
            key="inputs_gantt_mode",
        )
    with c3:
        color_by = st.radio(
            "Color por",
            ["Estado", "Piloto"],
            horizontal=True,
            key="inputs_gantt_color",
        )

    plot_df = df_gantt.copy()
    if fase_sel != "Todas" and "Fase" in plot_df.columns:
        plot_df = plot_df[plot_df["Fase"].astype(str).str.strip() == fase_sel].copy()
    if plot_df.empty:
        st.info("No hay tareas para la fase seleccionada.")
        return

    fig_gantt = build_inputs_gantt_figure(plot_df, date_mode=date_mode, color_by=color_by)
    st.plotly_chart(fig_gantt, use_container_width=True, key="inputs_estado_actual_gantt")


def render_inputs_contexto_block():
    try:
        contexto_title, contexto_sections = build_bullet_contexto_10kw_sections(BULLET_CONTEXTO_10KW_CSV_URL_DEFAULT, refresh_nonce=data_refresh_nonce)
    except Exception as exc:
        st.error(f"No se pudo cargar la hoja bullet: {exc}")
        return

    if not contexto_sections:
        st.info("La hoja bullet no contiene secciones para mostrar.")
        return

    contexto_title = contexto_title or "Arquitectura de inversión y creación de valor"
    total_sections = len(contexto_sections)
    total_bullets = sum(len(section["bullets"]) for section in contexto_sections)
    lead_section = contexto_sections[0]["title"] if contexto_sections else "-"

    st.markdown(
        """
        <style>
        .context-hero{
            border-radius:24px;
            padding:20px 22px 18px 22px;
            background:
                radial-gradient(circle at top right, rgba(14,165,164,.14), transparent 28%),
                linear-gradient(90deg,#f8fbff 0%,#e9f6ff 48%,#d7efff 100%);
            border:1px solid rgba(125,211,252,.42);
            box-shadow:0 16px 34px rgba(15,23,42,.08);
            margin-bottom:18px;
        }
        .context-hero-k{
            font-size:11px;
            font-weight:800;
            letter-spacing:.12em;
            text-transform:uppercase;
            color:#0f766e;
            margin-bottom:8px;
        }
        .context-hero-t{
            font-size:34px;
            font-weight:900;
            line-height:1.08;
            color:#0f172a;
            margin-bottom:10px;
            max-width:920px;
        }
        .context-hero-s{
            font-size:15px;
            line-height:1.6;
            color:#475569;
            max-width:980px;
            margin-bottom:14px;
        }
        .context-hero-band{
            display:flex;
            gap:10px;
            flex-wrap:wrap;
        }
        .context-hero-chip{
            display:inline-flex;
            align-items:center;
            gap:8px;
            padding:8px 12px;
            border-radius:999px;
            background:rgba(255,255,255,.82);
            border:1px solid rgba(148,163,184,.22);
            color:#334155;
            font-size:12px;
            font-weight:700;
        }
        .context-stat{
            border-radius:18px;
            padding:16px 18px;
            background:linear-gradient(180deg,#ffffff 0%,#f8fafc 100%);
            border:1px solid rgba(148,163,184,.24);
            box-shadow:0 8px 18px rgba(15,23,42,.05);
            margin-bottom:14px;
        }
        .context-stat-k{
            font-size:11px;
            font-weight:800;
            letter-spacing:.12em;
            text-transform:uppercase;
            color:#64748b;
            margin-bottom:8px;
        }
        .context-stat-v{
            font-size:30px;
            font-weight:800;
            line-height:1;
            color:#0f172a;
            margin-bottom:8px;
        }
        .context-stat-s{
            font-size:13px;
            color:#475569;
            line-height:1.45;
        }
        .context-card{
            border-radius:20px;
            padding:18px 18px 16px 18px;
            background:linear-gradient(180deg,#ffffff 0%,#f8fafc 100%);
            border:1px solid rgba(148,163,184,.22);
            box-shadow:0 10px 24px rgba(15,23,42,.05);
            margin-bottom:16px;
            height:100%;
        }
        .context-card:hover{
            transform:translateY(-1px);
            box-shadow:0 16px 30px rgba(15,23,42,.08);
            transition:all .18s ease;
        }
        .context-card-h{
            font-size:17px;
            font-weight:800;
            line-height:1.25;
            color:#0f172a;
            margin-bottom:12px;
        }
        .context-chip{
            display:inline-block;
            padding:4px 10px;
            border-radius:999px;
            font-size:12px;
            font-weight:700;
            background:#ecfeff;
            border:1px solid rgba(14,165,164,.20);
            color:#0f766e;
            margin-bottom:10px;
        }
        .context-bullet{
            display:flex;
            gap:10px;
            align-items:flex-start;
            margin-bottom:10px;
        }
        .context-dot{
            width:9px;
            height:9px;
            margin-top:7px;
            border-radius:999px;
            background:linear-gradient(180deg,#10b981 0%,#0ea5a4 100%);
            flex:0 0 auto;
        }
        .context-bullet-t{
            font-size:14px;
            line-height:1.5;
            color:#334155;
        }
        .context-grid-head{
            font-size:11px;
            font-weight:800;
            letter-spacing:.12em;
            text-transform:uppercase;
            color:#64748b;
            margin:2px 0 12px 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    stage_counts = {
        "Diseño y Optimización del Sistema": min(total_sections, 3),
        "Ingeniería Aplicada y Manufactura": max(0, min(total_sections - 3, 3)),
        "Integración y Validación del Activo": max(0, min(total_sections - 6, 2)),
    }
    st.markdown(
        f"""
        <div class="context-hero">
          <div class="context-hero-k">Ruta de Validación</div>
          <div class="context-hero-t">Validación Integrada del Sistema</div>
          <div class="context-hero-s">
            Lectura técnica estructurada para consolidar la validación del activo, conectando diseño, manufactura e integración como base habilitante para el escalamiento.
          </div>
          <div class="context-hero-band">
            <div class="context-hero-chip">Hitos técnicos: {total_sections}</div>
            <div class="context-hero-chip">Frentes documentados: {total_bullets}</div>
            <div class="context-hero-chip">Diseño: {stage_counts["Diseño y Optimización del Sistema"]}</div>
            <div class="context-hero-chip">Manufactura: {stage_counts["Ingeniería Aplicada y Manufactura"]}</div>
            <div class="context-hero-chip">Integración: {stage_counts["Integración y Validación del Activo"]}</div>
          </div>
        </div>
        <div class="context-grid-head">Matriz técnica de hitos</div>
        """,
        unsafe_allow_html=True,
    )
    rows = [contexto_sections[i:i + 2] for i in range(0, len(contexto_sections), 2)]
    for row_sections in rows:
        cols = st.columns(len(row_sections))
        for idx, section in enumerate(row_sections):
            absolute_idx = contexto_sections.index(section)
            if absolute_idx in (0, 1, 2):
                chip_label = "Diseño y Optimización del Sistema"
            elif absolute_idx in (3, 4, 5):
                chip_label = "Ingeniería Aplicada y Manufactura"
            elif absolute_idx in (6, 7):
                chip_label = "Integración y Validación del Activo"
            else:
                chip_label = "Narrativa estratégica"
            bullets_html = "".join(
                f'<div class="context-bullet"><span class="context-dot"></span><div class="context-bullet-t">{bullet}</div></div>'
                for bullet in section["bullets"]
            )
            with cols[idx]:
                st.markdown(
                    f"""
                    <div class="context-card">
                      <div class="context-chip">{chip_label}</div>
                      <div class="context-card-h">{section["title"]}</div>
                      {bullets_html}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def get_valor_activo_tecnologico_construido() -> tuple[float, float, float, float]:
    monto_total = 0.0
    capacidades_externo = 0.0
    know_how_fw = 0.0

    try:
        df_fin = load_dashboard_financiero_data(DASHBOARD_FINANCIERO_CSV_URL_DEFAULT, refresh_nonce=data_refresh_nonce)
        if not df_fin.empty and "Monto" in df_fin.columns:
            monto_total = float(df_fin["Monto"].dropna().sum() or 0.0)
    except Exception:
        monto_total = 0.0

    try:
        df_val_raw = load_valorizacion_raw_data(VALORIZACION_CSV_URL_DEFAULT, refresh_nonce=data_refresh_nonce)
        if df_val_raw.shape[0] > 6 and df_val_raw.shape[1] > 6:
            capacidades_externo = float(parse_money_clp_robusto(clean_sheet_cell(df_val_raw.iloc[5, 6])) or 0.0)
            know_how_fw = float(parse_money_clp_robusto(clean_sheet_cell(df_val_raw.iloc[6, 6])) or 0.0)
    except Exception:
        capacidades_externo = 0.0
        know_how_fw = 0.0

    return monto_total + capacidades_externo + know_how_fw, monto_total, capacidades_externo, know_how_fw


def render_inputs_capex_10kw_detail():
    try:
        df_10kw = build_restante_piloto_10kw_view(RESTANTE_PILOTO_10KW_CSV_URL_DEFAULT, refresh_nonce=data_refresh_nonce)
    except Exception as exc:
        st.error(f"No se pudo cargar la hoja Restante piloto 10kw: {exc}")
        return

    if df_10kw.empty:
        st.info("La hoja Restante piloto 10kw no contiene datos para mostrar.")
        return

    tabla_10kw = df_10kw[["Columna A", "Columna B", "Columna C"]].copy()
    resumen_10kw = (
        df_10kw[df_10kw["Valor C"] > 0]
        .groupby("Columna A", as_index=False)
        .agg(Monto_CLP=("Valor C", "sum"), Items=("Columna B", "count"))
        .sort_values("Monto_CLP", ascending=False)
        .reset_index(drop=True)
    )
    total_10kw = float(resumen_10kw["Monto_CLP"].sum() or 0.0)
    resumen_10kw["Pct_total"] = np.where(total_10kw > 0, resumen_10kw["Monto_CLP"] / total_10kw * 100.0, 0.0)
    resumen_10kw["Monto_fmt"] = resumen_10kw["Monto_CLP"].apply(format_clp)

    st.markdown("### Brecha de Inversión – Piloto 10 kW")

    st.markdown("#### Distribución relativa por componente")
    if resumen_10kw.empty:
        st.info("La columna C no contiene valores numéricos válidos para el gráfico de torta.")
    else:
        fig_10kw = px.pie(
            resumen_10kw,
            values="Monto_CLP",
            names="Columna A",
            hole=0.64,
            color_discrete_sequence=[
                "#0F766E", "#1D4ED8", "#F59E0B", "#7C3AED", "#DC2626", "#0891B2",
                "#65A30D", "#C2410C", "#BE185D", "#4F46E5", "#0EA5E9", "#EAB308",
                "#10B981", "#64748B", "#2563EB", "#14B8A6", "#9333EA", "#B45309",
            ],
        )
        fig_10kw.update_traces(
            textposition="inside",
            textinfo="percent",
            hovertemplate="<b>%{label}</b><br>Monto: $%{value:,.0f}<br>Participación: %{percent}<extra></extra>",
            marker=dict(line=dict(color="white", width=2)),
            sort=False,
        )
        fig_10kw.add_annotation(
            text="CAPEX<br>10kW",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=18, color="#0F172A"),
        )
        fig_10kw.update_layout(
            title="Participación por componente (Columna A vs Columna C)",
            margin=dict(l=10, r=10, t=56, b=130),
            height=560,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.10,
                xanchor="left",
                x=0,
                font=dict(size=11),
            ),
        )
        resumen_show = resumen_10kw[["Columna A", "Monto_fmt", "Items", "Pct_total"]].rename(
            columns={
                "Columna A": "Componente",
                "Monto_fmt": "Monto CLP",
                "Items": "Ítems",
                "Pct_total": "% del total",
            }
        )
        resumen_show["% del total"] = resumen_show["% del total"].map(lambda v: f"{v:.1f}%")
        pie_col, table_col = st.columns([1.15, 0.85])
        with pie_col:
            st.plotly_chart(fig_10kw, use_container_width=True)
        with table_col:
            st.markdown("##### Resumen consolidado")
            st.dataframe(
                style_engineering_table(resumen_show, header_color="#0F766E", row_color="#ECFDF5"),
                hide_index=True,
                use_container_width=True,
                height=480,
            )

    st.markdown("#### Tabla base de ingeniería")
    st.dataframe(
        style_engineering_table(tabla_10kw, header_color="#2C5783", row_color="#EAF6FF"),
        hide_index=True,
        use_container_width=True,
        height=360,
    )


def render_inputs_estado_actual_dashboard():
    try:
        df_fin = load_dashboard_financiero_data(DASHBOARD_FINANCIERO_CSV_URL_DEFAULT, refresh_nonce=data_refresh_nonce)
    except Exception as exc:
        st.error(f"No se pudo cargar la fuente de Dashboard Financiero Proyecto: {exc}")
        return

    base = df_fin[df_fin["Monto"].notna()].copy()
    if base.empty:
        st.info("La fuente de Dashboard Financiero Proyecto no contiene registros válidos.")
        return

    estado_subblock_key = "inputs_estado_actual_subbloque_sel"

    def _set_estado_subblock(value: str):
        st.session_state[estado_subblock_key] = value

    estado_subblocks = [
        ("contexto", "Construcción del Activo Tecnológico (en ejecución)"),
        ("financiero", "Materialización del Activo – Piloto 10 kW"),
    ]
    estado_subblock_icons = {
        "contexto": "🛠️",
        "financiero": "🧱",
    }
    if estado_subblock_key not in st.session_state:
        st.session_state[estado_subblock_key] = None

    subnav_cols = st.columns(2)
    for idx, (block_value, block_title) in enumerate(estado_subblocks):
        is_active = st.session_state.get(estado_subblock_key) == block_value
        with subnav_cols[idx]:
            st.markdown(
                f"""
                <div class="inputs-nav-card {'active' if is_active else ''}">
                    <div class="inputs-nav-k">Sub-bloque {idx + 1}</div>
                    <div class="inputs-nav-title-row">
                        <div class="inputs-nav-ico">{estado_subblock_icons.get(block_value, '📁')}</div>
                        <div class="inputs-nav-title-wrap">
                            <div class="inputs-nav-t">{block_title}</div>
                        </div>
                    </div>
                    <div class="inputs-nav-s">{'Seleccionado para análisis' if is_active else 'Haz clic para abrir este sub-bloque'}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.button(
                "Seleccionado" if is_active else "Abrir bloque",
                key=f"inputs_estado_subnav_{idx}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
                on_click=_set_estado_subblock,
                args=(block_value,),
            )

    st.markdown("---")
    selected_estado_subblock = st.session_state.get(estado_subblock_key)

    if selected_estado_subblock == "contexto":
        render_inputs_contexto_block()
        return
    if selected_estado_subblock != "financiero":
        st.info("Selecciona uno de los sub-bloques para abrir su contenido.")
        return

    valor_activo_tecnologico, monto_total, capacidades_externo, know_how_fw = get_valor_activo_tecnologico_construido()

    st.markdown(
        """
        <style>
        .asset-hero{
            border-radius:24px;
            padding:22px 24px;
            background:
                radial-gradient(circle at top right, rgba(14,165,164,.18), transparent 24%),
                linear-gradient(90deg,#f8fbff 0%,#e7f5ff 48%,#d4efff 100%);
            border:1px solid rgba(125,211,252,.45);
            box-shadow:0 16px 36px rgba(15,23,42,.08);
            margin-bottom:20px;
        }
        .asset-hero-grid{
            display:grid;
            grid-template-columns:1.3fr .9fr;
            gap:18px;
            align-items:stretch;
        }
        @media (max-width:1100px){.asset-hero-grid{grid-template-columns:1fr;}}
        .asset-hero-k{
            font-size:11px;
            font-weight:800;
            letter-spacing:.14em;
            text-transform:uppercase;
            color:#0f766e;
            margin-bottom:8px;
        }
        .asset-hero-t{
            font-size:18px;
            font-weight:800;
            line-height:1.2;
            color:#0f172a;
            margin-bottom:10px;
        }
        .asset-hero-v{
            font-size:58px;
            font-weight:900;
            line-height:1;
            color:#0f172a;
            margin-bottom:12px;
            letter-spacing:-.03em;
        }
        .asset-hero-p{
            font-size:15px;
            line-height:1.6;
            color:#475569;
            max-width:780px;
        }
        .asset-hero-panel{
            border-radius:18px;
            padding:16px 18px;
            background:rgba(255,255,255,.75);
            border:1px solid rgba(148,163,184,.24);
            backdrop-filter:blur(6px);
        }
        .asset-hero-panel-h{
            font-size:12px;
            font-weight:800;
            letter-spacing:.10em;
            text-transform:uppercase;
            color:#64748b;
            margin-bottom:10px;
        }
        .asset-hero-row{
            display:flex;
            justify-content:space-between;
            gap:12px;
            align-items:flex-start;
            padding:10px 0;
            border-bottom:1px solid rgba(226,232,240,.9);
        }
        .asset-hero-row:last-child{border-bottom:none;padding-bottom:0}
        .asset-hero-label{
            font-size:14px;
            font-weight:700;
            color:#0f172a;
            line-height:1.35;
        }
        .asset-hero-value{
            font-size:16px;
            font-weight:800;
            color:#0f172a;
            white-space:nowrap;
        }
        .asset-hero-total{
            margin-top:10px;
            padding-top:12px;
            border-top:1px dashed rgba(14,165,164,.32);
            display:flex;
            justify-content:space-between;
            gap:12px;
            align-items:center;
        }
        .asset-hero-total .asset-hero-label{color:#0f766e;}
        .asset-hero-total .asset-hero-value{font-size:20px;color:#0f766e;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="asset-hero">
          <div class="asset-hero-grid">
            <div>
              <div class="asset-hero-k">Resumen patrimonial técnico</div>
              <div class="asset-hero-t">Costo ejecutado en Activo Tecnológico Construido</div>
              <div class="asset-hero-v">{format_clp(valor_activo_tecnologico)}</div>
              <div class="asset-hero-p">
                Valor consolidado del activo construido considerando inversión ejecutada,
                capacidades externas incorporadas y know-how técnico valorizado dentro del proceso de desarrollo.
              </div>
            </div>
            <div class="asset-hero-panel">
              <div class="asset-hero-panel-h">Descomposición del valor</div>
              <div class="asset-hero-row">
                <div class="asset-hero-label">Inversión ejecutada en el activo</div>
                <div class="asset-hero-value">{format_clp(monto_total)}</div>
              </div>
              <div class="asset-hero-row">
                <div class="asset-hero-label">Capacidades externo</div>
                <div class="asset-hero-value">{format_clp(capacidades_externo)}</div>
              </div>
              <div class="asset-hero-row">
                <div class="asset-hero-label">Know-how FW</div>
                <div class="asset-hero-value">{format_clp(know_how_fw)}</div>
              </div>
              <div class="asset-hero-total">
                <div class="asset-hero-label">Valor construido total</div>
                <div class="asset-hero-value">{format_clp(valor_activo_tecnologico)}</div>
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Estado Técnico-Financiero del Piloto 10 kW")
    financial_kpis = render_inputs_financial_main_kpis(base)
    selected_financial_asset = financial_kpis.get("selected", "costo_ejecutado")

    if selected_financial_asset == "costo_ejecutado":
        st.markdown("### Distribución de Inversión por Categoría Técnica")
        fig_sm, tabla_sm = make_inputs_suministro_chart(base)
        if fig_sm is not None and tabla_sm is not None and not tabla_sm.empty:
            render_inputs_sm_kpi_cards(tabla_sm)
            st.plotly_chart(fig_sm, use_container_width=True)
        else:
            st.info("No hay datos válidos para graficar Suministro / Montaje.")

        render_inputs_project_gantt()
        render_inputs_item_analytics(base)
        render_inputs_factor_chart(base)


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


def render_pagos_hitos(
    capex_url: str,
    fx_used: float,
    pagos_scale: float,
    key_prefix: str = "",
    include_direction_salaries: bool = True,
):
    st.markdown("---")
    st.subheader("Estructura de Desembolso de Capital por Hitos del Proyecto")

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
        if include_direction_salaries:
            df_dir_mensual = build_direccion_mensual(df_direccion, horizonte_meses=int(df_meses["Mes"].max()))
            if not df_dir_mensual.empty:
                df_dir_mes = (
                    df_dir_mensual.groupby("Mes", as_index=False)
                    .agg(Pago_CLP_Sueldos=("Pago_CLP", "sum"))
                )
                df_consolidado = df_consolidado.merge(df_dir_mes, on="Mes", how="left")
            else:
                df_consolidado["Pago_CLP_Sueldos"] = 0.0
        else:
            df_consolidado["Pago_CLP_Sueldos"] = 0.0
        df_consolidado["Pago_CLP_Sueldos"] = df_consolidado["Pago_CLP_Sueldos"].fillna(0.0)
        df_consolidado["Pago_USD_Sueldos"] = np.where(
            np.isfinite(fx_used) and fx_used > 0,
            df_consolidado["Pago_CLP_Sueldos"] / fx_used,
            0.0,
        )
        df_consolidado["Total_USD"] = (
            df_consolidado["Pago_USD_Anticipo"]
            + df_consolidado["Pago_USD_Entrega"]
            + df_consolidado["Pago_USD_SAT"]
            + df_consolidado["Pago_USD_Sueldos"]
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
            key=f"{key_prefix}pay_currency_selector",
        )
        if unit_sel.startswith("USD"):
            scale_factor = 1.0 / 1_000.0
            axis_unit = "miles USD"
            line_label = "Acumulado (miles USD)"
            def fmt_bar_value(val: float) -> str:
                return f"{val:,.0f}k".replace(",", ".")
            def scale_clp(series: pd.Series) -> pd.Series:
                return (series / fx_used) / 1_000.0 if np.isfinite(fx_used) and fx_used > 0 else series * 0.0
        else:
            scale_factor = fx_used / 1_000_000.0
            axis_unit = "MM CLP"
            line_label = "Acumulado (MM CLP)"
            def fmt_bar_value(val: float) -> str:
                return f"{val:.1f} MM"
            def scale_clp(series: pd.Series) -> pd.Series:
                return series / 1_000_000.0

        def scale_usd(series: pd.Series) -> pd.Series:
            return series * scale_factor

        view_sel = st.selectbox(
            "Selecciona vista",
            [
                "1) Inyección por hito (Anticipo/FAT/SAT)",
                "2) Inyección por ítem",
                "3) Total por período + categoría",
            ],
            index=1,
            key=f"{key_prefix}pay_view_selector",
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
            df_sueldos_plot = df_consolidado[["Mes", "Pago_CLP_Sueldos"]].copy()
            df_sueldos_plot["Pago_plot"] = scale_clp(df_sueldos_plot["Pago_CLP_Sueldos"])
            df_sueldos_plot["Tipo"] = "Sueldos dirección"
            df_sueldos_plot = df_sueldos_plot[["Mes", "Tipo", "Pago_plot"]]
            df_flujo_long = pd.concat(
                [df_flujo_long[["Mes", "Tipo", "Pago_plot"]], df_sueldos_plot],
                ignore_index=True,
            )

            fig_iny = px.bar(
                df_flujo_long,
                x="Mes",
                y="Pago_plot",
                color="Tipo",
                color_discrete_map={
                    "Anticipo": "#0EA5A4",
                    "Entrega FAT": "#6366F1",
                    "SAT": "#F59E0B",
                    "Sueldos dirección": "#64748B",
                },
                labels={
                    "Mes": "Mes del proyecto",
                    "Pago_plot": f"Pago mensual ({axis_unit})",
                    "Tipo": "Hito de pago",
                },
                title=(
                    "Inyección por hito + sueldos de dirección"
                    if include_direction_salaries
                    else "Inyección por hito (Anticipo/FAT/SAT)"
                ),
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
capex_url = CAPEX_CSV_URL_DEFAULT

st.sidebar.markdown("---")
st.sidebar.caption(
    "El dashboard se alimenta directamente de Google Sheets y recalcula "
    "montos, tipo de cambio y gráficos en tiempo real."
)
if "data_refresh_nonce" not in st.session_state:
    st.session_state["data_refresh_nonce"] = 0
if st.sidebar.button("🔁 Actualizar datos desde URL"):
    st.session_state["data_refresh_nonce"] += 1
    st.rerun()

# =========================
# DATOS
# =========================
data_refresh_nonce = int(st.session_state.get("data_refresh_nonce", 0))

df_capex_base = load_capex_data(capex_url, refresh_nonce=data_refresh_nonce)
capex_total_usd_base = float(df_capex_base["Monto_USD"].sum() or 0.0) if "Monto_USD" in df_capex_base.columns else 0.0
fx_base = 909.0
global_fx_value = st.sidebar.number_input(
    "Tipo de cambio",
    min_value=100.0,
    max_value=5000.0,
    value=909.0,
    step=10.0,
    help="Valor base global CLP/USD. Se usa como referencia inicial para los cálculos y parámetros ligados al dólar.",
)
fx_used = global_fx_value if np.isfinite(global_fx_value) and global_fx_value > 0 else fx_base
df_capex = df_capex_base.copy()
if np.isfinite(fx_used) and fx_used > 0:
    df_capex["Monto_CLP"] = df_capex["Monto_USD"] * fx_used
capex_total_usd = capex_total_usd_base
capex_total_clp = float(df_capex["Monto_CLP"].sum() or 0.0) if "Monto_CLP" in df_capex.columns else 0.0
tipo_cambio_implicito = fx_used
pagos_scale = 1.0

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

try:
    df_direccion = load_director_general_data(capex_url, refresh_nonce=data_refresh_nonce)
    direccion_error = None
except Exception as exc:
    df_direccion = pd.DataFrame(columns=["Cargo", "Meses", "Costo empresa mensual", "Total"])
    direccion_error = str(exc)

direccion_total_clp = float(df_direccion["Total"].sum() or 0.0) if "Total" in df_direccion.columns else 0.0
capex_total_integrado_clp = capex_total_clp + direccion_total_clp
capex_total_real_clp = load_capex_total_real_clp(capex_url, refresh_nonce=data_refresh_nonce)
capex_total_integrado_real_clp = float(capex_total_real_clp or capex_total_clp) + direccion_total_clp

# =========================
# HEADER
# =========================
render_inputs_main_hero()

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
        min-height: 160px;
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
    .kpi-card.kpi-card-sky {
        background: linear-gradient(90deg, #EFF8FF 0%, #DFF4FF 42%, #C6ECFF 100%);
    }
    .kpi-card.kpi-card-green {
        background: linear-gradient(90deg, #ECFDF5 0%, #D1FAE5 42%, #A7F3D0 100%);
        border: 1px solid rgba(22,163,74,.28);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def kpi_card(title: str, value: str, subtitle: str = "", variant: str = "default"):
    """Renderiza una tarjeta KPI con título, valor y subtítulo."""
    card_class = "kpi-card"
    if variant == "sky":
        card_class += " kpi-card-sky"
    elif variant == "green":
        card_class += " kpi-card-green"
    html = f"""
    <div class="{card_class}">
        <div class="kpi-label">{title}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{subtitle}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_resumen_content(
    key_prefix: str = "",
    include_export: bool = True,
    include_direction_item: bool = False,
):
    st.subheader("Estructura de Inversión y Ejecución de CAPEX – Piloto 80 kW")

    df_items_tot = (
        df_capex
        .groupby("Item", as_index=False)
        .agg(Total_CLP=("Monto_CLP", "sum"))
    )
    df_items_tot = df_items_tot.merge(
        pd.DataFrame(list(item_category_lookup.items()), columns=["Item", "Categoria"]),
        on="Item",
        how="left",
    )
    if include_direction_item and direccion_total_clp > 0:
        df_items_tot = pd.concat(
            [
                df_items_tot,
                pd.DataFrame(
                    [
                        {
                            "Item": "Capital Humano",
                            "Total_CLP": direccion_total_clp,
                            "Categoria": "Dirección técnica",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    capex_total_clp_calc = (
        capex_total_integrado_clp if include_direction_item else df_items_tot["Total_CLP"].sum()
    )
    df_items_tot["Pct_total"] = df_items_tot["Total_CLP"] / capex_total_clp_calc
    df_items_tot["Total_MM"] = df_items_tot["Total_CLP"] / 1e6
    df_items_tot["Texto"] = df_items_tot.apply(
        lambda r: f"{r['Total_MM']:.1f} MM / {r['Pct_total']*100:.1f}%",
        axis=1
    )
    df_items_tot = df_items_tot.sort_values("Total_CLP", ascending=False)

    fig_item_total = px.bar(
        df_items_tot,
        x="Total_MM",
        y="Item",
        orientation="h",
        text="Texto",
        color="Item",
        color_discrete_map={**item_color_map, "Capital Humano": "#0F766E"},
        labels={"Total_MM": "Monto (millones de CLP)", "Item": "Ítem"},
        title=(
            "CAPEX + Capital Humano por ítem (monto total y % del total integrado)"
            if include_direction_item
            else "CAPEX por ítem (monto total y % del CAPEX)"
        ),
    )
    fig_item_total.update_traces(textposition="inside", insidetextanchor="middle", textfont_size=11)
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

    st.markdown("### Secuencia de Ejecución del CAPEX y Desarrollo del Proyecto")
    if "Mes_inicio" in df_capex.columns and "Mes_termino" in df_capex.columns:
        df_timeline = df_capex.dropna(subset=["Mes_inicio", "Mes_termino"]).copy()
        if not df_timeline.empty:
            df_timeline["Mes_inicio"] = pd.to_numeric(df_timeline["Mes_inicio"], errors="coerce")
            df_timeline["Mes_termino"] = pd.to_numeric(df_timeline["Mes_termino"], errors="coerce")
            df_timeline = df_timeline.dropna(subset=["Mes_inicio", "Mes_termino"])
            if not df_timeline.empty:
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
                    key=f"{key_prefix}timeline_radio_item_cat",
                    format_func=_fmt_item,
                )
                df_tl_plot = (
                    df_timeline_cat.copy()
                    if item_sel == "Todas"
                    else df_timeline_cat[df_timeline_cat["Item"] == item_sel].copy()
                )
                if not df_tl_plot.empty:
                    base_date = pd.to_datetime("2025-01-01")
                    df_tl_plot["Fecha_inicio"] = base_date + pd.to_timedelta((df_tl_plot["Mes_inicio"] - 1) * 30, unit="D")
                    df_tl_plot["Fecha_termino"] = base_date + pd.to_timedelta((df_tl_plot["Mes_termino"] - 1) * 30, unit="D")
                    df_tl_plot = df_tl_plot.sort_values(by=["Fecha_inicio", "Fecha_termino"], ascending=[False, False])
                    df_tl_plot["Categoria"] = pd.Categorical(
                        df_tl_plot["Categoria"],
                        categories=df_tl_plot["Categoria"].tolist(),
                        ordered=True,
                    )
                    fig_timeline_cat = px.timeline(
                        df_tl_plot,
                        x_start="Fecha_inicio",
                        x_end="Fecha_termino",
                        y="Categoria",
                        color="Item",
                        color_discrete_map=item_color_map,
                        hover_data={
                            "Categoria": True,
                            "Item": True,
                            "Mes_inicio": True,
                            "Mes_termino": True,
                            "Monto_CLP": ":,.0f",
                            "Monto_USD": ":,.0f",
                        },
                    )
                    fig_timeline_cat.update_yaxes(categoryorder="array", title="Categoría / Tarea")
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

    render_pagos_hitos(
        capex_url,
        fx_used,
        pagos_scale,
        key_prefix=key_prefix,
        include_direction_salaries=include_direction_item,
    )

    if include_export:
        st.markdown("---")
        st.subheader("📄 Exportar informe técnico")
        if REPORTLAB_AVAILABLE:
            pdf_bytes = build_pdf_report()
            st.download_button(
                label="📥 Descargar reporte PDF técnico (CAPEX Piloto 80 kW)",
                data=pdf_bytes,
                file_name="Reporte_CAPEX_Piloto_80kW.pdf",
                mime="application/pdf",
                key=f"{key_prefix}download_pdf_report",
            )
        else:
            st.info("La exportación PDF está deshabilitada porque `reportlab` no está instalado en este entorno.")

# =========================
# KPI CARDS – DISEÑO PRO
# =========================
def render_top_summary_kpis():
    st.markdown('<div class="kpi-row"></div>', unsafe_allow_html=True)
    k1, k2, k3 = st.columns(3)

    with k1:
        kpi_card(
            "CAPEX",
            format_clp(capex_total_clp),
            "Inversión piloto 80 kW, incluye I+D, componentes, montaje y contingencias."
        )

    with k2:
        kpi_card(
            "Dirección",
            format_clp(direccion_total_clp),
            "Fondos de dirección técnica separados del CAPEX base."
        )

    with k3:
        kpi_card(
            "CAPEX total",
            format_clp(capex_total_integrado_clp),
            "Suma referencial de CAPEX base + dirección técnica."
        )


def render_capex_categoria_content():
    st.subheader("Análisis técnico por categoría")

    df_cat_filtrado = df_cat.copy()
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

                    total_item = df_item_cat["Monto_CLP"].sum()
                    total_capex_visible = df_capex_filtrado["Monto_CLP"].sum()
                    pct_item_total = (total_item / total_capex_visible) if total_capex_visible > 0 else 0

                    fig_donut_item = px.pie(df_item_cat, values="Monto_CLP", names="Categoria", hole=0.70)
                    fig_donut_item.update_traces(
                        textinfo="percent",
                        textposition="inside",
                        hovertemplate="<b>%{label}</b><br>Participación dentro del ítem: %{percent:.1%}<br>Monto CLP: %{value:,.0f}<br><extra></extra>",
                        insidetextorientation="horizontal"
                    )
                    fig_donut_item.add_annotation(
                        x=0.5, y=0.5, text=f"{pct_item_total*100:.1f}%", showarrow=False,
                        font=dict(size=22, color="black"), xanchor="center", yanchor="middle"
                    )
                    fig_donut_item.update_layout(
                        showlegend=True,
                        legend=dict(orientation="v", x=1.25, y=0.5, xanchor="left", font=dict(size=11)),
                        margin=dict(l=0, r=120, t=10, b=10),
                        height=280,
                    )
                    st.plotly_chart(fig_donut_item, use_container_width=True)
    else:
        st.info("No hay ítems para mostrar en los dónuts según las categorías seleccionadas.")

    st.markdown("### Participación porcentual por categoría")
    total_clp_cat = df_cat_filtrado["Monto_CLP"].sum()
    df_cat_plot = df_cat_filtrado.copy().sort_values("Monto_CLP", ascending=False)
    df_cat_plot["Pct_cat"] = df_cat_plot["Monto_CLP"] / total_clp_cat if total_clp_cat > 0 else 0.0

    df_cat_item = (
        df_capex.groupby(["Categoria", "Item"], as_index=False)
        .agg(Monto_CLP=("Monto_CLP", "sum"))
        .sort_values(["Categoria", "Monto_CLP"], ascending=[True, False])
    )
    top_item_by_cat = df_cat_item.drop_duplicates(subset=["Categoria"], keep="first")
    cat_item_color_map = {
        row["Categoria"]: item_color_map.get(row["Item"], CAT_COLOR_MAP.get(row["Categoria"], "#2563EB"))
        for _, row in top_item_by_cat.iterrows()
    }

    fig_cat = px.bar(
        df_cat_plot,
        x="Categoria",
        y="Pct_cat",
        color="Categoria",
        color_discrete_map=cat_item_color_map,
        text="Pct_cat",
        labels={"Categoria": "Categoría", "Pct_cat": "Participación"},
        title="Distribución porcentual del CAPEX por categoría",
    )
    fig_cat.update_traces(texttemplate="%{text:.1%}", textposition="outside")
    max_part = float(df_cat_plot["Pct_cat"].max() or 0)
    fig_cat.update_yaxes(tickformat=".0%", range=[0, max_part * 1.15 if max_part > 0 else 1])
    fig_cat.update_layout(
        xaxis_title="",
        yaxis_title="Participación (%)",
        margin=dict(l=10, r=10, t=80, b=120),
        height=460,
        bargap=0.25,
        showlegend=False,
    )
    st.plotly_chart(fig_cat, use_container_width=True)
    st.session_state["fig_cat_categoria"] = fig_cat

    legend_css = """
    <style>
    .item-legend { display:flex; flex-wrap:wrap; gap:0.45rem 0.75rem; margin-top:0.4rem; margin-bottom:0.8rem; }
    .item-legend-title { font-size:0.82rem; font-weight:700; color:#6B7280; text-transform:uppercase; letter-spacing:.08em; margin-top:0.4rem; margin-bottom:0.25rem; }
    .item-legend-chip { display:inline-flex; align-items:center; gap:0.4rem; font-size:0.78rem; font-weight:600; color:#111827; }
    .item-legend-swatch { width:12px; height:12px; border-radius:2px; border:1px solid rgba(17, 24, 39, 0.2); }
    </style>
    """
    legend_items = []
    legend_order = [
        "Desarrollo Tecnológico", "Componentes Mecánicos", "Sistema Eléctrico y Control",
        "Obras Civiles", "Montaje y Logística", "Ensayos y Certificación", "Contingencias y Administración",
    ]
    for item in legend_order:
        color = CAT_COLOR_MAP.get(item, "#2563EB")
        legend_items.append(
            f'<span class="item-legend-chip"><span class="item-legend-swatch" style="background:{color}"></span>{item}</span>'
        )
    st.markdown(
        legend_css + '<div class="item-legend-title">Ítem</div>' + f'<div class="item-legend">{"".join(legend_items)}</div>',
        unsafe_allow_html=True,
    )

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
        df_show[["Categoria", "Participación (%)", "Monto_CLP_fmt", "Monto_USD_fmt", "Bullet_cat"]],
        hide_index=True,
        use_container_width=True,
    )


def render_capex_items_content():
    st.subheader("Top ítems por monto")
    render_category_palette()

    top_n = st.slider("Número de ítems a mostrar (Top N):", 5, 30, 15, step=1, key="capex_items_top_n")
    df_top = df_capex.sort_values("Monto_CLP", ascending=False).head(top_n).copy()
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
        labels={"Monto_CLP_MM": "Monto (MM CLP)", "Item": "Ítem", "Categoria": "Categoría"},
        title=f"Top {top_n} ítems por monto (millones de CLP)",
    )
    fig_top.update_traces(text=df_top["Monto_CLP_MM"].apply(lambda v: f"{v:.1f} MM"), textposition="outside")
    fig_top.update_layout(xaxis_title="Monto (millones de CLP)", yaxis_title="", margin=dict(l=10, r=10, t=60, b=10))
    st.plotly_chart(fig_top, use_container_width=True)
    st.session_state["fig_top_items"] = fig_top

    st.markdown("#### Tabla detallada")
    st.dataframe(
        df_top[["Item", "Categoria", "Participación (%)", "Monto_CLP_fmt", "Monto_USD_fmt", "Bullet"]],
        hide_index=True,
        use_container_width=True,
    )


def render_capex_module_content(selector_key: str = "capex_internal_selector"):
    st.markdown(
        """
        <div style="
            display:flex;
            align-items:center;
            gap:12px;
            margin:6px 0 14px 0;
            padding:10px 14px;
            border:1px solid rgba(229,231,235,1);
            border-radius:16px;
            background:linear-gradient(180deg,#ffffff 0%,#f8fafc 100%);
            width:fit-content;
            box-shadow:0 6px 14px rgba(15,23,42,.06);
        ">
            <div style="
                width:42px;
                height:42px;
                border-radius:12px;
                display:flex;
                align-items:center;
                justify-content:center;
                background:#0f172a;
                color:white;
                font-size:22px;
                font-weight:700;
            ">C</div>
            <div>
                <div style="font-size:11px;font-weight:700;letter-spacing:.12em;color:#64748b;text-transform:uppercase;">
                    Modulo
                </div>
                <div style="font-size:22px;font-weight:800;color:#111827;line-height:1.1;">
                    Capex
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    capex_subview = st.radio(
        "Visualización dentro de Capex",
        ["📂 Por categoría", "📄 Detalle de ítems"],
        index=0,
        horizontal=True,
        key=selector_key,
    )
    if capex_subview == "📂 Por categoría":
        render_capex_categoria_content()
    else:
        render_capex_items_content()


def render_direccion_module_content():
    st.subheader("Fondos de Dirección / Director General Técnico")
    st.info(
        "Esta pestaña muestra fondos de estructura técnica y dirección que se leen desde la hoja "
        "`Director General Técnico`. Estos montos no se suman al CAPEX base de 480 MM CLP."
    )

    if direccion_error:
        st.error(direccion_error)
    elif df_direccion.empty:
        st.warning("La hoja `Director General Técnico` no tiene registros válidos para mostrar.")
    else:
        total_direccion = float(df_direccion["Total"].sum() or 0.0)
        total_meses = float(df_direccion["Meses"].sum() or 0.0)
        costo_mensual_prom = (
            float(df_direccion["Costo empresa mensual"].mean() or 0.0)
            if not df_direccion["Costo empresa mensual"].empty else 0.0
        )
        capex_mas_direccion = capex_total_clp + total_direccion

        dk1, dk2, dk3, dk4 = st.columns(4)
        with dk1:
            kpi_card(
                "Fondos dirección (CLP)",
                format_clp(total_direccion),
                "Monto total separado del CAPEX técnico base."
            )
        with dk2:
            kpi_card(
                "Cargos cubiertos",
                f"{len(df_direccion):,}".replace(",", "."),
                "Roles leídos desde la hoja Dirección General Técnico."
            )
        with dk3:
            kpi_card(
                "Meses acumulados",
                f"{total_meses:,.0f}".replace(",", "."),
                "Suma de meses reportados por cargo."
            )
        with dk4:
            kpi_card(
                "CAPEX + Dirección",
                format_clp(capex_mas_direccion),
                "Vista referencial si quisieras observar ambos bloques juntos."
            )

        df_direccion["Total_MM"] = df_direccion["Total"] / 1e6
        df_direccion["Costo_mensual_fmt"] = df_direccion["Costo empresa mensual"].apply(format_clp)
        df_direccion["Total_fmt"] = df_direccion["Total"].apply(format_clp)

        df_dir_plot = df_direccion.sort_values("Total", ascending=True).copy()
        direccion_color_map = {
            "Ingeniero Eléctrico": "#7C3AED",
            "Ingeniero Mecánico": "#0F766E",
            "Ingeniero Proyecto (PMO)": "#C2410C",
            "Director General Técnico": "#1D4ED8",
        }
        fig_direccion = px.bar(
            df_dir_plot,
            x="Total_MM",
            y="Cargo",
            orientation="h",
            text=df_dir_plot["Total_MM"].map(lambda v: f"{v:.1f} MM"),
            color="Cargo",
            color_discrete_map=direccion_color_map,
            title="Fondos por cargo de dirección técnica",
            labels={"Total_MM": "Monto total (MM CLP)", "Cargo": ""},
        )
        fig_direccion.update_traces(
            textposition="outside",
            marker=dict(line=dict(color="rgba(255,255,255,0.85)", width=1.2)),
            hovertemplate="<b>%{y}</b><br>Total: %{x:.1f} MM CLP<extra></extra>",
        )
        fig_direccion.update_layout(
            showlegend=False,
            margin=dict(l=10, r=32, t=70, b=24),
            height=430,
            plot_bgcolor="white",
            paper_bgcolor="rgba(0,0,0,0)",
            bargap=0.20,
            title=dict(
                text="Fondos por cargo de direccion tecnica",
                font=dict(size=22, color="#0f172a"),
                x=0.02,
            ),
            font=dict(color="#334155", size=13),
        )
        fig_direccion.update_xaxes(
            showgrid=True,
            gridcolor="rgba(148,163,184,0.25)",
            zeroline=False,
            ticksuffix=" MM",
        )
        fig_direccion.update_yaxes(showgrid=False)
        st.plotly_chart(fig_direccion, use_container_width=True)

        col_dir_1, col_dir_2 = st.columns([1.2, 1])
        with col_dir_1:
            st.markdown("#### Tabla base")
            st.dataframe(
                df_direccion[["Cargo", "Meses", "Costo_mensual_fmt", "Total_fmt"]].rename(columns={
                    "Costo_mensual_fmt": "Costo empresa mensual",
                    "Total_fmt": "Total",
                }),
                hide_index=True,
                use_container_width=True,
            )
        with col_dir_2:
            st.markdown("#### Lectura ejecutiva")
            st.markdown(
                f"- Fondos de dirección identificados: **{format_clp(total_direccion)}**.\n"
                f"- Costo empresa mensual promedio: **{format_clp(costo_mensual_prom)}**.\n"
                f"- Si se observa junto al CAPEX técnico, la referencia total sería **{format_clp(capex_mas_direccion)}**.\n"
                f"- Este bloque se mantiene deliberadamente separado para no contaminar el desglose del CAPEX de ingeniería."
            )


def render_valorizacion_module_content(key_prefix: str = "val_"):
    state_block_key = f"{key_prefix}bloque_sel"
    state_bootstrap_key = f"{key_prefix}default_bootstrapped"

    def widget_key(name: str) -> str:
        return f"{key_prefix}{name}"

    def _set_valorizacion_bloque(value: str):
        st.session_state[state_block_key] = value

    bloque_cards = [
        ("1. Fundamentos de Creación de Valor", "Fundamentos de Creación de Valor"),
        ("2. Serie A: Inversión Inicial y Validación", "Serie A: Inversión Inicial y Validación"),
        ("3. Valorización Post-Validación", "Valorización Post-Validación"),
        ("4. Serie B: Escalamiento Comercial", "Serie B: Escalamiento Comercial"),
    ]
    if state_bootstrap_key not in st.session_state:
        st.session_state[state_block_key] = None
        st.session_state[state_bootstrap_key] = True
    elif state_block_key not in st.session_state:
        st.session_state[state_block_key] = None

    st.markdown(
        """
        <style>
        .val-nav-shell{
            border-radius:24px;
            padding:16px 18px 12px 18px;
            border:1px solid rgba(148,163,184,.18);
            background:linear-gradient(180deg,#ffffff 0%,#fbfdff 100%);
            box-shadow:0 12px 28px rgba(15,23,42,.05);
            margin-bottom:18px;
        }
        .val-nav-head{
            font-size:12px;
            font-weight:800;
            letter-spacing:.12em;
            text-transform:uppercase;
            color:#64748b;
            margin-bottom:10px;
        }
        .val-nav-guide{
            font-size:14px;
            line-height:1.55;
            color:#475569;
            margin-bottom:4px;
        }
        .val-nav-card{
            min-height:188px;
            display:flex;
            flex-direction:column;
            justify-content:flex-start;
            position:relative;
            overflow:hidden;
            border-radius:20px;
            padding:16px 18px 14px 18px;
            border:1px solid rgba(203,213,225,.75);
            background:
                radial-gradient(circle at top right, rgba(219,234,254,.35), transparent 26%),
                linear-gradient(180deg,#f8fafc 0%,#ffffff 74%);
            box-shadow:0 10px 24px rgba(15,23,42,.045);
            margin-bottom:10px;
        }
        .val-nav-card:before{
            content:"";
            position:absolute;
            left:0;
            top:0;
            bottom:0;
            width:5px;
            background:linear-gradient(180deg,#cbd5e1 0%,#e2e8f0 100%);
        }
        .val-nav-card.active{
            border:1px solid rgba(16,185,129,.34);
            box-shadow:0 18px 36px rgba(16,185,129,.14);
            background:
                radial-gradient(circle at top right, rgba(52,211,153,.20), transparent 24%),
                linear-gradient(90deg,#ecfdf5 0%,#d1fae5 40%,#a7f3d0 100%);
        }
        .val-nav-card.active:before{
            background:linear-gradient(180deg,#059669 0%,#34d399 100%);
        }
        .val-nav-k{
            font-size:11px;
            font-weight:800;
            letter-spacing:.10em;
            text-transform:uppercase;
            color:#64748B;
            margin-bottom:10px;
        }
        .val-nav-t{
            font-size:18px;
            font-weight:900;
            line-height:1.18;
            color:#0f172a;
            margin-bottom:10px;
            max-width:22ch;
        }
        .val-nav-title-row{
            display:flex;
            align-items:flex-start;
            gap:10px;
            margin-bottom:10px;
        }
        .val-nav-ico{
            font-size:22px;
            line-height:1;
            flex:0 0 auto;
            margin-top:1px;
        }
        .val-nav-title-wrap{
            min-width:0;
        }
        .val-nav-s{
            font-size:12px;
            line-height:1.5;
            color:#475569;
        }
        .val-nav-card.active .val-nav-k{color:#166534;}
        .val-nav-card.active .val-nav-t{color:#064e3b;}
        .val-nav-card.active .val-nav-s{color:#065f46;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="val-nav-shell">
          <div class="val-nav-head">Ruta de análisis</div>
          <div class="val-nav-guide">Avanza desde fundamentos, hacia Serie A, valorización post-validación y escalamiento comercial.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    nav_cols = st.columns(4)
    block_icons = {
        "1. Fundamentos de Creación de Valor": "📊",
        "2. Serie A: Inversión Inicial y Validación": "🏗️",
        "3. Valorización Post-Validación": "📈",
        "4. Serie B: Escalamiento Comercial": "🚀",
    }
    for idx, (block_value, block_title) in enumerate(bloque_cards):
        is_active = st.session_state.get(state_block_key) == block_value
        with nav_cols[idx]:
            st.markdown(
                f"""
                <div class="val-nav-card {'active' if is_active else ''}">
                    <div class="val-nav-k">Sub-bloque {idx + 1}</div>
                    <div class="val-nav-title-row">
                      <div class="val-nav-ico">{block_icons.get(block_value, '📁')}</div>
                      <div class="val-nav-title-wrap">
                        <div class="val-nav-t">{block_title}</div>
                      </div>
                    </div>
                    <div class="val-nav-s">{'Escenario activo para análisis' if is_active else 'Abrir escenario financiero'}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.button(
                "Seleccionado" if is_active else "Abrir bloque",
                key=widget_key(f"nav_{idx}"),
                use_container_width=True,
                type="primary" if is_active else "secondary",
                on_click=_set_valorizacion_bloque,
                args=(block_value,),
            )

    if st.session_state.get(state_block_key) is None:
        st.info("Selecciona uno de los sub-bloques de valorización para abrir su contenido.")
        return

    try:
        df_valorizacion = load_valorizacion_data(VALORIZACION_CSV_URL_DEFAULT, refresh_nonce=data_refresh_nonce)
        valorizacion_error = None
    except Exception as exc:
        df_valorizacion = pd.DataFrame()
        valorizacion_error = str(exc)

    try:
        df_eerrv2 = load_eerrv2_data(EERRV2_CSV_URL_DEFAULT, refresh_nonce=data_refresh_nonce)
        eerrv2_error = None
    except Exception as exc:
        df_eerrv2 = pd.DataFrame()
        eerrv2_error = str(exc)

    if valorizacion_error:
        st.error(f"No se pudo cargar la valorización: {valorizacion_error}")
        return
    if df_valorizacion.empty:
        st.warning("La hoja de valorización no contiene registros para mostrar.")
        return

    monto_col = find_best_column(
        df_valorizacion,
        ["total", "monto", "montoclp", "valor", "costo", "costototal"],
    )
    label_col = find_best_column(
        df_valorizacion,
        ["item", "concepto", "descripcion", "detalle", "categoria", "cargo"],
    )

    df_val_plot = pd.DataFrame()
    if monto_col:
        df_valorizacion[f"{monto_col}__num"] = df_valorizacion[monto_col].apply(parse_money_clp_robusto)
        if label_col:
            df_val_plot = (
                df_valorizacion[[label_col, f"{monto_col}__num"]]
                .rename(columns={label_col: "Label", f"{monto_col}__num": "Monto"})
                .groupby("Label", as_index=False)["Monto"]
                .sum()
                .sort_values("Monto", ascending=False)
                .head(15)
            )

    if not df_val_plot.empty:
        df_val_plot["Monto_MM"] = df_val_plot["Monto"] / 1e6
        fig_val = px.bar(
            df_val_plot.sort_values("Monto", ascending=True),
            x="Monto_MM",
            y="Label",
            orientation="h",
            text=df_val_plot.sort_values("Monto", ascending=True)["Monto_MM"].map(lambda v: f"{v:.1f} MM"),
            title="Top conceptos de valorización",
            labels={"Monto_MM": "Monto (MM CLP)", "Label": ""},
            color="Monto_MM",
            color_continuous_scale=["#DBEAFE", "#60A5FA", "#2563EB"],
        )
        fig_val.update_traces(textposition="outside")
        fig_val.update_layout(showlegend=False, coloraxis_showscale=False, margin=dict(l=10, r=20, t=60, b=10), height=460)
        st.plotly_chart(fig_val, use_container_width=True)

    df_model, model_map = get_valorizacion_model_map(df_valorizacion)
    fx_default = float(fx_used) if np.isfinite(fx_used) and fx_used > 0 else parse_model_number(model_map.get("fxclpusd", 915))
    total_base_knowhow_clp, _, _, _ = get_valor_activo_tecnologico_construido()
    pre_money_actual_default = total_base_knowhow_clp / fx_default if fx_default > 0 and total_base_knowhow_clp > 0 else 0.0
    capex_10kw_default = 0.0
    try:
        df_restante_10kw = build_restante_piloto_10kw_view(RESTANTE_PILOTO_10KW_CSV_URL_DEFAULT, refresh_nonce=data_refresh_nonce)
        if not df_restante_10kw.empty:
            capex_10kw_default = float(df_restante_10kw["Valor C"].sum() or 0.0)
    except Exception:
        capex_10kw_default = 0.0
    inversion_clp_default = capex_10kw_default + float(capex_total_integrado_clp or 0.0)
    ebitda_unit_default = parse_model_number(model_map.get("ebitdaunitariodereferencia", 0))
    volumen_default = parse_model_number(model_map.get("volumencomercialdereferencia", 0))
    multiple_default = 1.0
    multiple_post_default = 5.0
    captura_default = parse_model_percent(model_map.get("capturadelvalorpotencialpostpiloto", "100%"))
    ronda_pct_default = parse_model_percent(model_map.get("participacionobjetivoparanuevosinversionistas", "70%"))
    widget_defaults = {
        "fx": int(round(fx_default or 915)),
        "pre_money": int(round(pre_money_actual_default)),
        "inv_clp": int(round(inversion_clp_default)),
        "investment_currency": "CLP",
        "volume": int(round(volumen_default)),
        "ebitda_unit": int(round(ebitda_unit_default)),
        "multiple": float(multiple_default or 1.0),
        "ronda_pct": float((ronda_pct_default or 0.70) * 100.0),
        "valuation_basis": "BASE INVERSION + KNOW-HOW",
        "alloc_manual": False,
        "fluxial_pct_manual": 50.0,
        "imelsa_pct_manual": 50.0,
    }
    bloque_sel = st.session_state.get(state_block_key)
    shared_group = "base" if bloque_sel in {"1. Fundamentos de Creación de Valor", "2. Serie A: Inversión Inicial y Validación"} else "post"

    def shared_state_key(name: str, group: str | None = None) -> str:
        active_group = group or shared_group
        return widget_key(f"state_{active_group}_{name}")

    def shared_widget_key(name: str, group: str | None = None) -> str:
        active_group = group or shared_group
        return widget_key(f"widget_{active_group}_{name}")

    def prime_widget(name: str, group: str | None = None) -> str:
        active_group = group or shared_group
        state_key = shared_state_key(name, active_group)
        widget_state_key = shared_widget_key(name, active_group)
        if widget_state_key not in st.session_state:
            st.session_state[widget_state_key] = st.session_state[state_key]
        return widget_state_key

    def sync_widget_to_state(name: str, group: str | None = None):
        active_group = group or shared_group
        st.session_state[shared_state_key(name, active_group)] = st.session_state[shared_widget_key(name, active_group)]

    def sync_investment_currency(group: str | None = None):
        active_group = group or shared_group
        currency = st.session_state[shared_widget_key("investment_currency", active_group)]
        st.session_state[shared_state_key("investment_currency", active_group)] = currency
        if currency == "USD":
            fx_val = float(st.session_state.get(shared_state_key("fx", active_group), 1) or 1)
            clp_val = float(st.session_state.get(shared_state_key("inv_clp", active_group), 0) or 0)
            st.session_state[shared_widget_key("inv_usd_display", active_group)] = int(round(clp_val / fx_val)) if fx_val > 0 else 0

    def sync_investment_usd_to_clp(group: str | None = None):
        active_group = group or shared_group
        fx_val = float(st.session_state.get(shared_state_key("fx", active_group), 1) or 1)
        usd_val = float(st.session_state.get(shared_widget_key("inv_usd_display", active_group), 0) or 0)
        st.session_state[shared_state_key("inv_clp", active_group)] = int(round(usd_val * fx_val))
        st.session_state.pop(shared_widget_key("inv_clp", active_group), None)

    def resolve_fluxial_pre_money(valuation_basis: str, base_usd: float, ebitda_value: float) -> float:
        return base_usd if valuation_basis == "BASE INVERSION + KNOW-HOW" else ebitda_value

    group_widget_defaults = {
        "base": widget_defaults,
        "post": {
            **widget_defaults,
            "multiple": float(multiple_post_default),
        },
    }

    for group_name in ("base", "post"):
        for name, default_value in group_widget_defaults[group_name].items():
            if shared_state_key(name, group_name) not in st.session_state:
                st.session_state[shared_state_key(name, group_name)] = default_value

    # Keep block 2 investment aligned with the "Capital a recaudar" KPI.
    st.session_state[shared_state_key("inv_clp", "base")] = int(round(inversion_clp_default))
    st.session_state.pop(shared_widget_key("inv_clp", "base"), None)

    st.markdown(
        """
        <style>
        .val-summary-hero{
            border-radius:24px;
            padding:22px 24px;
            background:
                radial-gradient(circle at top right, rgba(14,165,164,.16), transparent 24%),
                linear-gradient(90deg,#f8fbff 0%,#e7f5ff 48%,#d4efff 100%);
            border:1px solid rgba(125,211,252,.42);
            box-shadow:0 16px 36px rgba(15,23,42,.08);
            margin-bottom:8px;
        }
        .val-summary-grid{
            display:grid;
            grid-template-columns:1.25fr .95fr;
            gap:18px;
            align-items:stretch;
        }
        @media (max-width:1100px){.val-summary-grid{grid-template-columns:1fr;}}
        .val-summary-k{
            font-size:11px;font-weight:800;letter-spacing:.14em;text-transform:uppercase;color:#0f766e;margin-bottom:8px;
        }
        .val-summary-t{
            font-size:18px;font-weight:800;line-height:1.2;color:#0f172a;margin-bottom:10px;
        }
        .val-summary-v{
            font-size:52px;font-weight:900;line-height:1;color:#0f172a;margin-bottom:12px;letter-spacing:-.03em;
        }
        .val-summary-p{
            font-size:15px;line-height:1.6;color:#475569;max-width:720px;
        }
        .val-summary-panel{
            border-radius:18px;padding:16px 18px;background:rgba(255,255,255,.76);border:1px solid rgba(148,163,184,.24);backdrop-filter:blur(6px);
        }
        .val-summary-panel-h{
            font-size:12px;font-weight:800;letter-spacing:.10em;text-transform:uppercase;color:#64748b;margin-bottom:10px;
        }
        .val-summary-row{
            display:flex;justify-content:space-between;gap:12px;align-items:flex-start;padding:10px 0;border-bottom:1px solid rgba(226,232,240,.9);
        }
        .val-summary-row:last-child{border-bottom:none;padding-bottom:0}
        .val-summary-label{
            font-size:14px;font-weight:700;color:#0f172a;line-height:1.35;
        }
        .val-summary-value{
            font-size:16px;font-weight:800;color:#0f172a;white-space:nowrap;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if bloque_sel == "1. Fundamentos de Creación de Valor":
        base_fx_preview = float(st.session_state.get(shared_state_key("fx", "base"), fx_default or 1))
        base_currency_preview = str(st.session_state.get(shared_state_key("investment_currency", "base"), "CLP"))
        volume_preview = float(st.session_state.get(shared_state_key("volume", "base"), volumen_default))
        ebitda_unit_preview = float(st.session_state.get(shared_state_key("ebitda_unit", "base"), ebitda_unit_default))
        multiple_preview = float(st.session_state.get(shared_state_key("multiple", "base"), multiple_default or 1))
        valuation_basis_preview = str(st.session_state.get(shared_state_key("valuation_basis", "base"), "BASE INVERSION + KNOW-HOW"))
        base_en_usd_preview = (total_base_knowhow_clp / base_fx_preview) if base_fx_preview > 0 else 0.0
        ebitda_preview = volume_preview * ebitda_unit_preview * multiple_preview
        valorizacion_fluxial_preview = resolve_fluxial_pre_money(valuation_basis_preview, base_en_usd_preview, ebitda_preview)
        if valuation_basis_preview == "BASE INVERSION + KNOW-HOW":
            if base_currency_preview == "CLP":
                composition_rows_preview = (
                    f'<div class="val-summary-row">'
                    f'<div class="val-summary-label">BASE INVERSION + KNOW-HOW</div>'
                    f'<div class="val-summary-value">{format_clp(total_base_knowhow_clp)}</div>'
                    f'</div>'
                )
                valorizacion_fluxial_preview_display = format_clp(valorizacion_fluxial_preview * base_fx_preview)
            else:
                composition_rows_preview = (
                    f'<div class="val-summary-row">'
                    f'<div class="val-summary-label">BASE INVERSION + KNOW-HOW</div>'
                    f'<div class="val-summary-value">{format_clp(total_base_knowhow_clp)}</div>'
                    f'</div>'
                    f'<div class="val-summary-row">'
                    f'<div class="val-summary-label">BASE INVERSION + KNOW-HOW EN USD</div>'
                    f'<div class="val-summary-value">{format_usd(base_en_usd_preview)}</div>'
                    f'</div>'
                )
                valorizacion_fluxial_preview_display = format_usd(valorizacion_fluxial_preview)
        else:
            composition_rows_preview = (
                f'<div class="val-summary-row">'
                f'<div class="val-summary-label">Múltiplo EBITDA</div>'
                f'<div class="val-summary-value">{multiple_preview:.2f}x</div>'
                f'</div>'
                f'<div class="val-summary-row">'
                f'<div class="val-summary-label">EBITDA potencial multiplicado</div>'
                f'<div class="val-summary-value">{format_clp(ebitda_preview * base_fx_preview) if base_currency_preview == "CLP" else format_usd(ebitda_preview)}</div>'
                f'</div>'
            )
            valorizacion_fluxial_preview_display = format_clp(valorizacion_fluxial_preview * base_fx_preview) if base_currency_preview == "CLP" else format_usd(valorizacion_fluxial_preview)
        st.markdown("---")
        st.markdown(
            f"""
            <div class="val-summary-hero">
              <div class="val-summary-grid">
                <div>
                  <div class="val-summary-k">Resumen de valorización</div>
                  <div class="val-summary-t">Valorización Fluxial Hoy (Pre-money)</div>
                  <div class="val-summary-v">{valorizacion_fluxial_preview_display}</div>
                  <div class="val-summary-p">
                    Valor pre-money estimado segun la metodologia seleccionada para valorizacion,
                    expresado en la moneda activa del sub-bloque.
                  </div>
                </div>
                <div class="val-summary-panel">
                  <div class="val-summary-panel-h">Composición del valor</div>
                  {composition_rows_preview}
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif bloque_sel == "4. Serie B: Escalamiento Comercial":
        base_fx_preview = float(st.session_state.get(shared_state_key("fx", "base"), fx_default or 1))
        base_volume_preview = float(st.session_state.get(shared_state_key("volume", "base"), volumen_default))
        base_ebitda_unit_preview = float(st.session_state.get(shared_state_key("ebitda_unit", "base"), ebitda_unit_default))
        post_multiple_preview = float(st.session_state.get(shared_state_key("multiple", "post"), multiple_post_default))
        post_ronda_pct_preview = float(st.session_state.get(shared_state_key("ronda_pct", "post"), (ronda_pct_default or 0.70) * 100.0)) / 100.0

        base_ebitda_preview = base_volume_preview * base_ebitda_unit_preview * post_multiple_preview
        valuation_basis_preview = str(st.session_state.get(shared_state_key("valuation_basis", "base"), "BASE INVERSION + KNOW-HOW"))
        base_base_en_usd_preview = (total_base_knowhow_clp / base_fx_preview) if base_fx_preview > 0 else 0.0
        base_pre_money_preview = resolve_fluxial_pre_money(
            valuation_basis_preview,
            base_base_en_usd_preview,
            (
                float(st.session_state.get(shared_state_key("volume", "base"), volumen_default))
                * float(st.session_state.get(shared_state_key("ebitda_unit", "base"), ebitda_unit_default))
                * float(st.session_state.get(shared_state_key("multiple", "base"), multiple_default or 1))
            ),
        )
        base_inv_preview = float(st.session_state.get(shared_state_key("inv_clp", "base"), inversion_clp_default)) / base_fx_preview if base_fx_preview > 0 else 0.0
        post_money_a_preview = base_pre_money_preview + base_inv_preview
        valor_post_piloto_preview = base_ebitda_preview
        capital_raise_preview = valor_post_piloto_preview * post_ronda_pct_preview
        post_money_b_preview = valor_post_piloto_preview + capital_raise_preview

        st.markdown("---")
        st.markdown(
            f"""
            <div class="val-summary-hero">
              <div class="val-summary-grid">
                <div>
                  <div class="val-summary-k">Resumen de valorización</div>
                  <div class="val-summary-t">Post-money Serie B</div>
                  <div class="val-summary-v">{format_usd(post_money_b_preview)}</div>
                  <div class="val-summary-p">
                    Valorización posterior a la nueva ronda, integrando la base post piloto y el capital a levantar en Serie B.
                  </div>
                </div>
                <div class="val-summary-panel">
                  <div class="val-summary-panel-h">Composición del valor</div>
                  <div class="val-summary-row">
                    <div class="val-summary-label">Valorización base Serie B</div>
                    <div class="val-summary-value">{format_usd(valor_post_piloto_preview)}</div>
                  </div>
                  <div class="val-summary-row">
                    <div class="val-summary-label">Capital Serie B</div>
                    <div class="val-summary-value">{format_usd(capital_raise_preview)}</div>
                  </div>
                  <div class="val-summary-row">
                    <div class="val-summary-label">Post-money Serie B</div>
                    <div class="val-summary-value">{format_usd(post_money_b_preview)}</div>
                  </div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    supuestos_title_map = {
        "1. Fundamentos de Creación de Valor": "Supuestos Base de Creación de Valor",
        "2. Serie A: Inversión Inicial y Validación": "Supuestos Económicos de Serie A",
        "3. Valorización Post-Validación": "Supuestos de Valorización Post-Validación",
        "4. Serie B: Escalamiento Comercial": "Supuestos de Ronda y Expansión Comercial",
    }
    supuestos_subtitle_map = {
        "1. Fundamentos de Creación de Valor": "Parámetros técnicos y económicos que explican la base de creación de valor del modelo.",
        "2. Serie A: Inversión Inicial y Validación": "Entradas de inversión y validación para estructurar la etapa inicial y la cesión asociada.",
        "3. Valorización Post-Validación": "Variables de volumen y múltiplo para proyectar el valor del activo tras validación.",
        "4. Serie B: Escalamiento Comercial": "Supuestos de ronda, expansión comercial y dilución para la etapa de escalamiento.",
    }

    st.markdown(
        """
        <style>
        .sup-shell{
            border-radius:24px;
            padding:18px 20px 14px 20px;
            border:1px solid rgba(148,163,184,.18);
            background:linear-gradient(180deg,#ffffff 0%,#f8fbff 100%);
            box-shadow:0 12px 28px rgba(15,23,42,.05);
            margin-bottom:18px;
        }
        .sup-head-k{
            font-size:11px;
            font-weight:800;
            letter-spacing:.12em;
            text-transform:uppercase;
            color:#64748B;
            margin-bottom:6px;
        }
        .sup-head-t{
            font-size:16px;
            line-height:1.55;
            color:#475569;
            margin-top:4px;
            margin-bottom:0;
            max-width:980px;
        }
        .sup-input-shell{
            border-radius:22px;
            padding:16px 16px 8px 16px;
            border:1px solid rgba(203,213,225,.72);
            background:
                radial-gradient(circle at top right, rgba(219,234,254,.35), transparent 24%),
                linear-gradient(180deg,#f8fafc 0%,#ffffff 78%);
            box-shadow:0 10px 22px rgba(15,23,42,.04);
            margin:10px 0 18px 0;
        }
        .sup-input-shell-title{
            font-size:11px;
            font-weight:800;
            letter-spacing:.12em;
            text-transform:uppercase;
            color:#64748B;
            margin-bottom:10px;
        }
        .sup-kpi-shell{
            margin-top:8px;
            margin-bottom:18px;
        }
        .eng-section-label{
            font-size:11px;
            font-weight:800;
            letter-spacing:.12em;
            text-transform:uppercase;
            color:#64748B;
            margin-bottom:8px;
        }
        .eng-transition{
            border-top:1px solid rgba(203,213,225,.75);
            padding-top:16px;
            margin-top:10px;
            margin-bottom:18px;
        }
        .eng-transition-k{
            font-size:11px;
            font-weight:800;
            letter-spacing:.12em;
            text-transform:uppercase;
            color:#64748B;
            margin-bottom:6px;
        }
        .eng-transition-t{
            font-size:15px;
            line-height:1.55;
            color:#475569;
            max-width:980px;
        }
        .sup-shell div[data-testid="stButton"] > button,
        .sup-shell div[data-testid="stDownloadButton"] > button{
            min-height:74px;
            border-radius:999px;
            padding:0 24px;
            font-size:18px;
            font-weight:800;
            letter-spacing:-0.01em;
            border:1px solid rgba(148,163,184,.18);
            box-shadow:0 14px 28px rgba(15,23,42,.10);
            transition:transform .18s ease, box-shadow .18s ease, filter .18s ease;
        }
        .sup-shell div[data-testid="stButton"] > button:hover,
        .sup-shell div[data-testid="stDownloadButton"] > button:hover{
            transform:translateY(-1px);
            box-shadow:0 18px 30px rgba(15,23,42,.14);
            filter:saturate(1.03);
        }
        .sup-shell div[data-testid="stButton"] > button{
            color:#1E293B;
            background:
                radial-gradient(circle at 20% 20%, rgba(255,255,255,.40), transparent 34%),
                linear-gradient(135deg,#FFFFFF 0%,#F3F8FF 48%,#E0EDFF 100%);
        }
        .sup-shell div[data-testid="stButton"] > button p{
            font-size:18px;
            font-weight:800;
            color:#1E293B;
        }
        .sup-shell div[data-testid="stDownloadButton"] > button{
            color:#ffffff;
            border:1px solid rgba(59,130,246,.18);
            background:
                radial-gradient(circle at 24% 24%, rgba(255,255,255,.28), transparent 28%),
                linear-gradient(135deg,#2563EB 0%,#1D4ED8 40%,#0EA5E9 100%);
        }
        .sup-shell div[data-testid="stDownloadButton"] > button p{
            font-size:18px;
            font-weight:800;
            color:#ffffff;
        }
        .sup-shell div[data-testid="stDownloadButton"] > button::before{
            content:"⬇";
            display:inline-block;
            margin-right:10px;
            font-size:18px;
            font-weight:900;
        }
        .sup-shell div[data-testid="stButton"] > button::before{
            content:"↺";
            display:inline-block;
            margin-right:10px;
            font-size:18px;
            font-weight:900;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown('<div class="sup-shell">', unsafe_allow_html=True)
    sh_col, reset_col, pdf_col = st.columns([1, 0.24, 0.28])
    with sh_col:
        st.subheader(supuestos_title_map.get(bloque_sel, "Supuestos Clave del Modelo de Valorización"))
        st.markdown(
            f'<p class="sup-head-t">{supuestos_subtitle_map.get(bloque_sel, "")}</p>',
            unsafe_allow_html=True,
        )
    with reset_col:
        if st.button("Restablecer supuestos", key=widget_key("reset_supuestos"), use_container_width=True):
            for group_name in ("base", "post"):
                for name, default_value in group_widget_defaults[group_name].items():
                    st.session_state[shared_state_key(name, group_name)] = default_value
                    st.session_state.pop(shared_widget_key(name, group_name), None)
    pdf_export_slot = pdf_col.empty()
    st.markdown('<div class="sup-input-shell"><div class="sup-input-shell-title">Panel de entrada</div>', unsafe_allow_html=True)

    # Seed all calculation inputs from shared state so every branch has valid values.
    fx_input = float(st.session_state[shared_state_key("fx")])
    pre_money_input = float(st.session_state[shared_state_key("pre_money")])
    inversion_clp_input = float(st.session_state[shared_state_key("inv_clp")])
    volume_input = float(st.session_state[shared_state_key("volume")])
    ebitda_unit_input = float(st.session_state[shared_state_key("ebitda_unit")])
    multiple_input = float(st.session_state[shared_state_key("multiple")])
    ronda_pct_input = float(st.session_state[shared_state_key("ronda_pct")]) / 100.0
    captura_input = float(captura_default or 1.0)
    alloc_manual_input = bool(st.session_state[shared_state_key("alloc_manual")])
    fluxial_pct_manual_input = float(st.session_state[shared_state_key("fluxial_pct_manual")]) / 100.0
    imelsa_pct_manual_input = float(st.session_state[shared_state_key("imelsa_pct_manual")]) / 100.0
    investment_currency_input = str(st.session_state.get(shared_state_key("investment_currency"), "CLP"))
    aporte_no_pecuniario_clp = 0.0
    aporte_no_pecuniario_usd = 0.0

    if bloque_sel == "1. Fundamentos de Creación de Valor":
        pcol1, pcol2, pcol3 = st.columns(3)
        with pcol1:
            fx_input = st.number_input("FX CLP/USD", min_value=1, value=int(st.session_state[shared_state_key("fx")]), step=1, format="%d", key=prime_widget("fx"), on_change=sync_widget_to_state, args=("fx",))
            render_input_thousands_hint(fx_input)
        with pcol2:
            investment_currency_input = st.selectbox(
                "Moneda de inversión",
                ["CLP", "USD"],
                key=prime_widget("investment_currency"),
                on_change=sync_investment_currency,
            )
        with pcol3:
            valuation_basis_input = st.selectbox(
                "Base de valorización pre-money",
                ["BASE INVERSION + KNOW-HOW", "EBITDA potencial ciclo inicial"],
                key=prime_widget("valuation_basis"),
                on_change=sync_widget_to_state,
                args=("valuation_basis",),
            )
        if valuation_basis_input == "EBITDA potencial ciclo inicial":
            pcol3, pcol4 = st.columns(2)
            with pcol3:
                volume_input = st.number_input("Volumen comercial", min_value=0, value=int(st.session_state[shared_state_key("volume")]), step=1, format="%d", key=prime_widget("volume"), on_change=sync_widget_to_state, args=("volume",))
                render_input_thousands_hint(volume_input)
            with pcol4:
                ebitda_unit_input = st.number_input("EBITDA unitario (USD)", min_value=0, value=int(st.session_state[shared_state_key("ebitda_unit")]), step=1000, format="%d", key=prime_widget("ebitda_unit"), on_change=sync_widget_to_state, args=("ebitda_unit",))
                render_input_thousands_hint(ebitda_unit_input, "US$")
        with st.columns(1)[0]:
            multiple_input = st.slider("Múltiplo EBITDA", min_value=1.0, max_value=12.0, value=float(st.session_state[shared_state_key("multiple")]), step=0.5, key=prime_widget("multiple"), on_change=sync_widget_to_state, args=("multiple",))
        inversion_clp_input = float(st.session_state[shared_state_key("inv_clp")])
        captura_input = float(captura_default or 1.0)
        ronda_pct_input = float(st.session_state[shared_state_key("ronda_pct")]) / 100.0
    elif bloque_sel == "2. Serie A: Inversión Inicial y Validación":
        pcol1, pcol2, pcol3 = st.columns(3)
        with pcol1:
            fx_input = st.number_input("FX CLP/USD", min_value=1, value=int(st.session_state[shared_state_key("fx")]), step=1, format="%d", key=prime_widget("fx"), on_change=sync_widget_to_state, args=("fx",))
            render_input_thousands_hint(fx_input)
        with pcol2:
            investment_currency_input = st.selectbox(
                "Moneda de inversión",
                ["CLP", "USD"],
                key=prime_widget("investment_currency"),
                on_change=sync_investment_currency,
            )
        with pcol3:
            if investment_currency_input == "USD":
                inv_usd_widget_key = shared_widget_key("inv_usd_display")
                if inv_usd_widget_key not in st.session_state:
                    st.session_state[inv_usd_widget_key] = int(round(float(st.session_state[shared_state_key("inv_clp")]) / fx_input)) if fx_input > 0 else 0
                inversion_usd_input = st.number_input(
                    "Inversión piloto (USD)",
                    min_value=0,
                    value=int(st.session_state[inv_usd_widget_key]),
                    step=10000,
                    format="%d",
                    key=inv_usd_widget_key,
                    on_change=sync_investment_usd_to_clp,
                )
                render_input_thousands_hint(inversion_usd_input, "US$")
                inversion_clp_input = float(inversion_usd_input) * fx_input
                st.session_state[shared_state_key("inv_clp")] = int(round(inversion_clp_input))
                st.session_state.pop(shared_widget_key("inv_clp"), None)
            else:
                inversion_clp_input = st.number_input("Inversión piloto (CLP)", min_value=0, value=int(st.session_state[shared_state_key("inv_clp")]), step=10000000, format="%d", key=prime_widget("inv_clp"), on_change=sync_widget_to_state, args=("inv_clp",))
                render_input_thousands_hint(inversion_clp_input, "$")
        volume_input = float(st.session_state[shared_state_key("volume")])
        ebitda_unit_input = float(st.session_state[shared_state_key("ebitda_unit")])
        multiple_input = float(st.session_state[shared_state_key("multiple")])
        captura_input = float(captura_default or 1.0)
        ronda_pct_input = float(st.session_state[shared_state_key("ronda_pct")]) / 100.0
        auto_ebitda_potencial = volume_input * ebitda_unit_input * multiple_input
        auto_base_en_usd = (total_base_knowhow_clp / fx_input) if fx_input > 0 else 0.0
        auto_valuation_basis = str(st.session_state.get(shared_state_key("valuation_basis", "base"), "BASE INVERSION + KNOW-HOW"))
        auto_valorizacion_fluxial = resolve_fluxial_pre_money(
            auto_valuation_basis,
            auto_base_en_usd,
            auto_ebitda_potencial,
        )
        auto_inversion_usd = inversion_clp_input / fx_input if fx_input > 0 else 0.0
        auto_post_money = auto_valorizacion_fluxial + auto_inversion_usd
        auto_imelsa_pct = (auto_inversion_usd / auto_post_money) if auto_post_money > 0 else 0.0
        auto_fluxial_pct = max(0.0, 1.0 - auto_imelsa_pct)

        if not alloc_manual_input:
            st.session_state[shared_state_key("fluxial_pct_manual")] = auto_fluxial_pct * 100.0
            st.session_state[shared_state_key("imelsa_pct_manual")] = auto_imelsa_pct * 100.0
            st.session_state.pop(shared_widget_key("fluxial_pct_manual"), None)
            st.session_state.pop(shared_widget_key("imelsa_pct_manual"), None)

        with st.columns(1)[0]:
            alloc_manual_input = st.checkbox(
                "Asignar participación manual para Fluxial e IMELSA",
                key=prime_widget("alloc_manual"),
                on_change=sync_widget_to_state,
                args=("alloc_manual",),
            )
        if alloc_manual_input:
            st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
            manual_wrap_left, manual_col_1, manual_col_2, manual_col_3, manual_wrap_right = st.columns([0.18, 1.1, 1, 1, 0.18])
            with manual_col_1:
                imelsa_pct_manual_input = st.slider(
                    "% IMELSA manual",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(st.session_state[shared_state_key("imelsa_pct_manual")]),
                    step=1.0,
                    format="%.0f%%",
                    key=prime_widget("imelsa_pct_manual"),
                    on_change=sync_widget_to_state,
                    args=("imelsa_pct_manual",),
                ) / 100.0
                fluxial_pct_manual_input = max(0.0, 1.0 - imelsa_pct_manual_input)
                st.session_state[shared_state_key("fluxial_pct_manual")] = fluxial_pct_manual_input * 100.0
                st.session_state.pop(shared_widget_key("fluxial_pct_manual"), None)
                aporte_no_pecuniario_usd_manual = max(0.0, (auto_post_money * imelsa_pct_manual_input) - auto_inversion_usd) if auto_post_money > 0 else 0.0
                aporte_no_pecuniario_usd = aporte_no_pecuniario_usd_manual
                aporte_no_pecuniario_clp = aporte_no_pecuniario_usd_manual * fx_input
            with manual_col_2:
                st.markdown(
                    f"""
                    <div style="margin-top:-42px;">
                    <div style="
                        border:1px solid rgba(148,163,184,.24);
                        border-radius:16px;
                        padding:16px 18px;
                        background:linear-gradient(180deg,#ffffff 0%,#f8fafc 100%);
                        box-shadow:0 6px 14px rgba(15,23,42,.04);
                    ">
                        <div style="font-size:12px;font-weight:800;letter-spacing:.08em;text-transform:uppercase;color:#64748B;margin-bottom:8px;">
                            Complemento automático
                        </div>
                        <div style="font-size:34px;font-weight:800;line-height:1;color:#0f172a;margin-bottom:8px;">
                            {fluxial_pct_manual_input:.0%}
                        </div>
                        <div style="font-size:13px;line-height:1.45;color:#475569;">
                            Se calcula como 100% menos la participación manual asignada a IMELSA.
                        </div>
                    </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with manual_col_3:
                st.markdown(
                    f"""
                    <div style="margin-top:-42px;">
                    <div style="
                        border:1px solid rgba(191,219,254,.55);
                        border-radius:16px;
                        padding:16px 18px;
                        background:linear-gradient(90deg,#EFF8FF 0%,#DFF4FF 42%,#C6ECFF 100%);
                        box-shadow:0 6px 14px rgba(15,23,42,.04);
                    ">
                        <div style="font-size:12px;font-weight:800;letter-spacing:.08em;text-transform:uppercase;color:#64748B;margin-bottom:8px;">
                            Valor complementario
                        </div>
                        <div style="font-size:34px;font-weight:800;line-height:1;color:#0f172a;margin-bottom:8px;">
                            {format_clp(aporte_no_pecuniario_clp) if investment_currency_input == "CLP" else format_usd(aporte_no_pecuniario_usd)}
                        </div>
                        <div style="font-size:13px;line-height:1.45;color:#475569;">
                            {"Monto adicional reconocido en CLP para complementar la estructura de entrada de la Serie A." if investment_currency_input == "CLP" else "Monto adicional reconocido en USD para complementar la estructura de entrada de la Serie A."}
                        </div>
                    </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.markdown("<div style='height:32px;'></div>", unsafe_allow_html=True)
    elif bloque_sel == "3. Valorización Post-Validación":
        pcol1, pcol2, pcol3 = st.columns(3)
        with pcol1:
            volume_input = st.number_input("Volumen comercial", min_value=0, value=int(st.session_state[shared_state_key("volume")]), step=1, format="%d", key=prime_widget("volume"), on_change=sync_widget_to_state, args=("volume",))
            render_input_thousands_hint(volume_input)
        with pcol2:
            ebitda_unit_input = st.number_input("EBITDA unitario (USD)", min_value=0, value=int(st.session_state[shared_state_key("ebitda_unit")]), step=1000, format="%d", key=prime_widget("ebitda_unit"), on_change=sync_widget_to_state, args=("ebitda_unit",))
            render_input_thousands_hint(ebitda_unit_input, "US$")
        with pcol3:
            multiple_input = st.slider("Múltiplo EBITDA", min_value=1.0, max_value=12.0, value=float(st.session_state[shared_state_key("multiple")]), step=0.5, key=prime_widget("multiple"), on_change=sync_widget_to_state, args=("multiple",))
        pcol4, pcol5 = st.columns(2)
        with pcol4:
            fx_input = st.number_input("FX CLP/USD", min_value=1, value=int(st.session_state[shared_state_key("fx")]), step=1, format="%d", key=prime_widget("fx"), on_change=sync_widget_to_state, args=("fx",))
            render_input_thousands_hint(fx_input)
        with pcol5:
            investment_currency_input = st.selectbox(
                "Moneda de inversión",
                ["CLP", "USD"],
                key=prime_widget("investment_currency"),
                on_change=sync_investment_currency,
            )
        captura_input = float(captura_default or 1.0)
        inversion_clp_input = float(st.session_state[shared_state_key("inv_clp")])
        ronda_pct_input = float(st.session_state[shared_state_key("ronda_pct")]) / 100.0
    else:
        pcol1, pcol2, pcol3 = st.columns(3)
        with pcol1:
            fx_input = st.number_input("FX CLP/USD", min_value=1, value=int(st.session_state[shared_state_key("fx")]), step=1, format="%d", key=prime_widget("fx"), on_change=sync_widget_to_state, args=("fx",))
            render_input_thousands_hint(fx_input)
        with pcol2:
            investment_currency_input = st.selectbox(
                "Moneda de inversión",
                ["CLP", "USD"],
                key=prime_widget("investment_currency"),
                on_change=sync_investment_currency,
            )
        with pcol3:
            ronda_pct_input = st.slider("Nueva cesión Serie B", min_value=5.0, max_value=90.0, value=float(st.session_state[shared_state_key("ronda_pct")]), step=1.0, format="%.0f%%", key=prime_widget("ronda_pct"), on_change=sync_widget_to_state, args=("ronda_pct",)) / 100.0
        pcol4, pcol5, pcol6 = st.columns(3)
        with pcol4:
            volume_input = st.number_input("Volumen comercial", min_value=0, value=int(st.session_state[shared_state_key("volume")]), step=1, format="%d", key=prime_widget("volume"), on_change=sync_widget_to_state, args=("volume",))
            render_input_thousands_hint(volume_input)
        with pcol5:
            ebitda_unit_input = st.number_input("EBITDA unitario (USD)", min_value=0, value=int(st.session_state[shared_state_key("ebitda_unit")]), step=1000, format="%d", key=prime_widget("ebitda_unit"), on_change=sync_widget_to_state, args=("ebitda_unit",))
            render_input_thousands_hint(ebitda_unit_input, "US$")
        with pcol6:
            multiple_input = st.slider("Múltiplo EBITDA", min_value=1.0, max_value=12.0, value=float(st.session_state[shared_state_key("multiple")]), step=0.5, key=prime_widget("multiple"), on_change=sync_widget_to_state, args=("multiple",))
        captura_input = float(captura_default or 1.0)
        inversion_clp_input = float(st.session_state[shared_state_key("inv_clp")])

    st.markdown("</div></div>", unsafe_allow_html=True)

    base_en_usd = (total_base_knowhow_clp / fx_input) if fx_input > 0 else 0.0
    ebitda_potencial_ciclo_inicial = volume_input * ebitda_unit_input
    ebitda_potencial_multiplicado = ebitda_potencial_ciclo_inicial * multiple_input
    valuation_basis_input = str(st.session_state.get(shared_state_key("valuation_basis", "base"), "BASE INVERSION + KNOW-HOW"))
    valorizacion_fluxial_hoy = resolve_fluxial_pre_money(
        valuation_basis_input,
        base_en_usd,
        ebitda_potencial_multiplicado,
    )
    inversion_usd = inversion_clp_input / fx_input if fx_input > 0 else 0.0
    post_money_serie_a = valorizacion_fluxial_hoy + inversion_usd
    imelsa_pct = (inversion_usd / post_money_serie_a) if post_money_serie_a > 0 else 0.0
    fluxial_pct = max(0.0, 1.0 - imelsa_pct)
    if bloque_sel == "2. Serie A: Inversión Inicial y Validación" and alloc_manual_input:
        fluxial_pct = fluxial_pct_manual_input
        imelsa_pct = imelsa_pct_manual_input
        aporte_no_pecuniario_usd = aporte_no_pecuniario_clp / fx_input if fx_input > 0 else 0.0
        post_money_serie_a += aporte_no_pecuniario_usd
    else:
        aporte_no_pecuniario_usd = max(0.0, (post_money_serie_a * imelsa_pct) - inversion_usd) if post_money_serie_a > 0 else 0.0
        aporte_no_pecuniario_clp = aporte_no_pecuniario_usd * fx_input

    # Serie B must inherit the ownership mix coming from block 2 / Serie A.
    base_fx_input = float(st.session_state.get(shared_state_key("fx", "base"), fx_default or 1))
    base_pre_money_input = float(st.session_state.get(shared_state_key("pre_money", "base"), pre_money_actual_default))
    base_volume_input = float(st.session_state.get(shared_state_key("volume", "base"), volumen_default))
    base_ebitda_unit_input = float(st.session_state.get(shared_state_key("ebitda_unit", "base"), ebitda_unit_default))
    base_multiple_input = float(st.session_state.get(shared_state_key("multiple", "base"), multiple_default or 1))
    base_inversion_clp_input = float(st.session_state.get(shared_state_key("inv_clp", "base"), inversion_clp_default))

    base_ebitda_potencial = base_volume_input * base_ebitda_unit_input * base_multiple_input
    base_valuation_basis_input = str(st.session_state.get(shared_state_key("valuation_basis", "base"), "BASE INVERSION + KNOW-HOW"))
    base_base_en_usd = (total_base_knowhow_clp / base_fx_input) if base_fx_input > 0 else 0.0
    base_valorizacion_fluxial_hoy = resolve_fluxial_pre_money(
        base_valuation_basis_input,
        base_base_en_usd,
        base_ebitda_potencial,
    )
    base_inversion_usd = base_inversion_clp_input / base_fx_input if base_fx_input > 0 else 0.0
    base_post_money_serie_a = base_valorizacion_fluxial_hoy + base_inversion_usd
    base_imelsa_pct = (base_inversion_usd / base_post_money_serie_a) if base_post_money_serie_a > 0 else 0.0
    base_fluxial_pct = max(0.0, 1.0 - base_imelsa_pct)
    base_alloc_manual = bool(st.session_state.get(shared_state_key("alloc_manual", "base"), False))
    if base_alloc_manual:
        base_fluxial_pct = float(st.session_state.get(shared_state_key("fluxial_pct_manual", "base"), base_fluxial_pct * 100.0)) / 100.0
        base_imelsa_pct = float(st.session_state.get(shared_state_key("imelsa_pct_manual", "base"), base_imelsa_pct * 100.0)) / 100.0
        base_aporte_no_pecuniario_usd = max(0.0, (base_post_money_serie_a * base_imelsa_pct) - base_inversion_usd) if base_post_money_serie_a > 0 else 0.0
        base_post_money_serie_a += base_aporte_no_pecuniario_usd
    else:
        base_aporte_no_pecuniario_usd = max(0.0, (base_post_money_serie_a * base_imelsa_pct) - base_inversion_usd) if base_post_money_serie_a > 0 else 0.0
    ebitda_total = ebitda_potencial_ciclo_inicial
    valor_post_piloto = ebitda_total * multiple_input
    upside_pct = ((valor_post_piloto / base_post_money_serie_a) - 1.0) if base_post_money_serie_a > 0 else 0.0
    valor_imelsa_post = valor_post_piloto * base_imelsa_pct
    capital_raise = valor_post_piloto * ronda_pct_input
    post_money_serie_b = valor_post_piloto + capital_raise
    pct_remanente = 1.0 - ronda_pct_input
    socios_actuales_total_pct = base_fluxial_pct + base_imelsa_pct
    if socios_actuales_total_pct > 0:
        fluxial_share_base = base_fluxial_pct / socios_actuales_total_pct
        imelsa_share_base = base_imelsa_pct / socios_actuales_total_pct
    else:
        fluxial_share_base = 0.0
        imelsa_share_base = 0.0
    fluxial_post_b = pct_remanente * fluxial_share_base
    imelsa_post_b = pct_remanente * imelsa_share_base
    valor_fluxial_post_b = post_money_serie_b * fluxial_post_b
    valor_imelsa_post_b = post_money_serie_b * imelsa_post_b
    base_post_money_serie_a_clp = base_post_money_serie_a * fx_input
    ebitda_total_clp = ebitda_total * fx_input
    valor_post_piloto_clp = valor_post_piloto * fx_input
    valor_imelsa_post_clp = valor_imelsa_post * fx_input
    capital_raise_clp = capital_raise * fx_input
    post_money_serie_b_clp = post_money_serie_b * fx_input
    valor_fluxial_post_b_clp = valor_fluxial_post_b * fx_input
    valor_imelsa_post_b_clp = valor_imelsa_post_b * fx_input

    def build_valorizacion_supuestos_pdf() -> bytes:
        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=1.2 * cm,
            leftMargin=1.2 * cm,
            topMargin=1.2 * cm,
            bottomMargin=1.2 * cm,
        )
        styles = getSampleStyleSheet()
        h1 = styles["Heading1"]
        h2 = styles["Heading2"]
        body = styles["BodyText"]
        body.fontSize = 9
        body.leading = 12
        elements = []

        investment_currency_label = "CLP" if investment_currency_input == "CLP" else "USD"
        valuation_basis_label_pdf = "BASE INVERSION + KNOW-HOW" if valuation_basis_input == "BASE INVERSION + KNOW-HOW" else "EBITDA potencial ciclo inicial"
        post_multiple_assumption = float(st.session_state.get(shared_state_key("multiple", "post"), multiple_post_default))
        post_ronda_assumption = float(st.session_state.get(shared_state_key("ronda_pct", "post"), (ronda_pct_default or 0.70) * 100.0)) / 100.0
        post_money_serie_a_clp_pdf = post_money_serie_a * fx_input
        valor_post_piloto_clp = valor_post_piloto * fx_input
        capital_raise_clp = capital_raise * fx_input
        post_money_serie_b_clp = post_money_serie_b * fx_input

        def add_table(title: str, rows: list[list[str]], col_widths: list[float]):
            elements.append(Paragraph(title, h2))
            table = Table(rows, colWidths=col_widths, repeatRows=1)
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EAF3FF")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, -1), 8),
                        ("LEADING", (0, 0), (-1, -1), 10),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("ALIGN", (1, 1), (-1, -1), "LEFT"),
                        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CBD5E1")),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FBFF")]),
                        ("TOPPADDING", (0, 0), (-1, -1), 5),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ]
                )
            )
            elements.append(table)
            elements.append(Spacer(1, 0.35 * cm))

        elements.append(Paragraph("Supuestos y KPI Clave - Modelo de Valorización", h1))
        elements.append(Paragraph("Resumen técnico de ingeniería financiera con los supuestos activos y los KPI principales por sub-bloque.", body))
        elements.append(Spacer(1, 0.35 * cm))

        assumptions_rows = [
            ["Parámetro", "Valor", "Observación"],
            ["FX CLP/USD", f"{fx_input:,.0f}".replace(",", "."), "Tipo de cambio activo del modelo"],
            ["Base de valorización pre-money", valuation_basis_label_pdf, "Mecanismo activo para valor pre-money"],
            ["Moneda de inversión Serie A", investment_currency_label, "Moneda visible del bloque 2"],
            ["Inversión piloto (CLP)", format_clp(inversion_clp_input), "Capital base de entrada al piloto"],
            ["Inversión piloto (USD)", format_usd(inversion_usd), "Conversión del capital de entrada usando FX"],
            ["Volumen comercial", f"{volume_input:,.0f}".replace(",", "."), "Supuesto operativo del escenario"],
            ["EBITDA unitario", format_usd(ebitda_unit_input), "Margen operativo unitario"],
            ["Múltiplo EBITDA base", f"{multiple_input:.2f}x", "Supuesto usado en bloques 1 a 3"],
            ["Múltiplo EBITDA Serie B", f"{post_multiple_assumption:.2f}x", "Supuesto heredado para escalamiento"],
            ["Nueva cesión Serie B", f"{post_ronda_assumption:.1%}", "Participación objetivo para nuevos inversionistas"],
            ["Asignación manual Serie A", "Sí" if alloc_manual_input else "No", "Activa el complemento de valor para IMELSA"],
        ]
        if alloc_manual_input:
            assumptions_rows.extend(
                [
                    ["% IMELSA manual", f"{imelsa_pct_manual_input:.1%}", "Participación fijada manualmente"],
                    ["% Fluxial manual", f"{fluxial_pct_manual_input:.1%}", "Complemento automático de participación"],
                    ["Valor complementario (USD)", format_usd(aporte_no_pecuniario_usd), "Aporte adicional reconocido en la Serie A"],
                ]
            )
        add_table("Supuestos acordados", assumptions_rows, [5.6 * cm, 4.0 * cm, 7.2 * cm])

        bloque1_rows = [["KPI", "Valor", "Lectura"]]
        if valuation_basis_input == "BASE INVERSION + KNOW-HOW":
            bloque1_rows.extend(
                [
                    ["Base inversión + know-how", format_clp(total_base_knowhow_clp), "Base patrimonial en CLP"],
                    ["Base inversión + know-how USD", format_usd(base_en_usd), "Base patrimonial convertida a USD"],
                ]
            )
        else:
            bloque1_rows.extend(
                [
                    ["Múltiplo EBITDA", f"{multiple_input:.2f}x", "Supuesto activo de múltiplo"],
                    ["EBITDA potencial ciclo inicial", format_usd(ebitda_potencial_multiplicado), "EBITDA potencial multiplicado"],
                ]
            )
        bloque1_rows.append(["Valorización Fluxial Hoy (Pre-money)", format_usd(valorizacion_fluxial_hoy), "Resultado del bloque 1"])
        add_table("Sub-bloque 1 - Fundamentos de Creación de Valor", bloque1_rows, [6.2 * cm, 4.1 * cm, 6.5 * cm])

        bloque2_rows = [
            ["KPI", "Valor", "Lectura"],
            ["Inversión piloto (CLP)", format_clp(inversion_clp_input), "Capital comprometido en pesos"],
            ["Inversión piloto (USD)", format_usd(inversion_usd), "Capital comprometido convertido a USD"],
            ["Post-money Serie A (CLP)", format_clp(post_money_serie_a_clp_pdf), "Post-money expresado en CLP"],
            ["Post-money Serie A (USD)", format_usd(post_money_serie_a), "Post-money expresado en USD"],
            ["% IMELSA", f"{imelsa_pct:.1%}", "Participación post ingreso"],
            ["% Fluxial", f"{fluxial_pct:.1%}", "Participación remanente"],
        ]
        if alloc_manual_input:
            bloque2_rows.append(["Valor complementario (USD)", format_usd(aporte_no_pecuniario_usd), "Aporte adicional incorporado al post-money"])
        add_table("Sub-bloque 2 - Serie A: Inversión Inicial y Validación", bloque2_rows, [6.2 * cm, 4.1 * cm, 6.5 * cm])

        bloque3_rows = [
            ["KPI", "Valor", "Lectura"],
            ["Pre Money actual (USD)", format_usd(base_post_money_serie_a), "Base heredada desde Serie A"],
            ["EBITDA total", format_usd(ebitda_total), "EBITDA unitario multiplicado por volumen"],
            ["Valorización post piloto (USD)", format_usd(valor_post_piloto), "EBITDA total multiplicado por el múltiplo activo"],
            ["Valorización post piloto (CLP)", format_clp(valor_post_piloto_clp), "Referencia equivalente en CLP"],
            ["Upside vs actual", f"{upside_pct:.1%}", "Crecimiento de valorización post piloto respecto de Pre Money actual"],
            ["Valor por 50% post piloto", format_usd(valor_imelsa_post), "Valor implícito de una participación equivalente al 50% post piloto"],
        ]
        add_table("Sub-bloque 3 - Valorización Post-Validación", bloque3_rows, [6.2 * cm, 4.1 * cm, 6.5 * cm])

        bloque4_rows = [
            ["KPI", "Valor", "Lectura"],
            ["Valorización base Serie B", format_usd(valor_post_piloto), "Pre-money sugerido para la segunda ronda"],
            ["Capital Serie B (USD)", format_usd(capital_raise), "Capital implícito a levantar"],
            ["Capital Serie B (CLP)", format_clp(capital_raise_clp), "Referencia equivalente en CLP"],
            ["Post-money Serie B (USD)", format_usd(post_money_serie_b), "Valorización posterior al cierre"],
            ["Post-money Serie B (CLP)", format_clp(post_money_serie_b_clp), "Referencia equivalente en CLP"],
            ["% remanente socios actuales", f"{pct_remanente:.1%}", "Participación conjunta post ronda"],
            ["% Fluxial post ronda", f"{fluxial_post_b:.1%}", "Participación final de Fluxial"],
            ["% IMELSA post ronda", f"{imelsa_post_b:.1%}", "Participación final de IMELSA"],
        ]
        add_table("Sub-bloque 4 - Serie B: Escalamiento Comercial", bloque4_rows, [6.2 * cm, 4.1 * cm, 6.5 * cm])

        doc.build(elements)
        pdf_bytes = buffer.getvalue()
        buffer.close()
        return pdf_bytes

    if bloque_sel == "1. Fundamentos de Creación de Valor":
        st.markdown(
            """
            <div class="eng-transition">
              <div class="eng-transition-k">Capa de análisis financiero</div>
              <div class="eng-transition-t">A continuación se despliega la proyección integrada del modelo, sus drivers unitarios y la lectura consolidada de desempeño económico.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if eerrv2_error:
            st.error(f"No se pudo cargar EERRv2: {eerrv2_error}")
        elif df_eerrv2.empty:
            st.warning("La hoja EERRv2 no tiene datos disponibles.")
        else:
            st.markdown(
                """
                <style>
                .eerr-mini{
                    border-radius:16px;padding:14px 15px;border:1px solid rgba(148,163,184,.22);
                    background:linear-gradient(90deg,#EFF8FF 0%,#DFF4FF 42%,#C6ECFF 100%);
                    min-height:132px;
                }
                .eerr-mini-h{font-size:11px;font-weight:800;letter-spacing:.08em;text-transform:uppercase;color:#64748B;margin-bottom:6px}
                .eerr-mini-v{font-size:28px;font-weight:800;color:#0f172a;line-height:1.05;margin-bottom:4px}
                .eerr-mini-s{font-size:12px;color:#475569}
                </style>
                """,
                unsafe_allow_html=True,
            )
            eerr_headers = [clean_sheet_cell(v) for v in df_eerrv2.iloc[1, 1:8].tolist()]
            eerr_data = df_eerrv2.iloc[2:10, 1:8].copy()
            eerr_data.columns = eerr_headers
            eerr_data = eerr_data.applymap(clean_sheet_cell)
            cash_headers = [clean_sheet_cell(v) for v in df_eerrv2.iloc[1, 1:8].tolist()]
            cash_data = df_eerrv2.iloc[12:23, 1:8].copy()
            cash_data.columns = cash_headers
            cash_data = cash_data.applymap(clean_sheet_cell)
            kpi_headers = [clean_sheet_cell(v) for v in df_eerrv2.iloc[1, 9:11].tolist()]
            kpi_data = df_eerrv2.iloc[2:10, 9:11].copy()
            kpi_data.columns = kpi_headers
            kpi_data = kpi_data.applymap(clean_sheet_cell)
            kpi_map = {clean_sheet_cell(row[kpi_headers[0]]): clean_sheet_cell(row[kpi_headers[1]]) for _, row in kpi_data.iterrows() if clean_sheet_cell(row[kpi_headers[0]])}
            precio_venta_turbina = get_first_model_value(
                model_map,
                [
                    "Precio venta / turbina",
                    "Precio venta/turbina",
                    "Precio venta turbina",
                ],
            )
            costo_estimado_turbina = get_first_model_value(
                model_map,
                [
                    "Costo estimado / turbina",
                    "Costo estimado/turbina",
                    "Costo estimado turbina",
                ],
            )
            ebitda_unitario_val = get_first_model_value(
                model_map,
                [
                    "EBITDA unitario",
                    "EBITDA unitario de referencia",
                ],
                default=ebitda_unit_default,
            )
            capex_inicial_eerr = parse_model_number(clean_sheet_cell(df_eerrv2.iloc[14, 2])) if df_eerrv2.shape[0] > 14 and df_eerrv2.shape[1] > 2 else 0.0

            eerr_numeric = eerr_data.copy()
            series_cols = [c for c in eerr_numeric.columns if c != "Partida"]
            for col in series_cols:
                eerr_numeric[col] = eerr_numeric[col].apply(parse_model_number)
            row_lookup = {normalize_key(clean_sheet_cell(r["Partida"])): r for _, r in eerr_numeric.iterrows()}
            cash_numeric = cash_data.copy()
            cash_series_cols = [c for c in cash_numeric.columns if c != "Partida"]
            for col in cash_series_cols:
                cash_numeric[col] = cash_numeric[col].apply(parse_model_number)
            cash_row_lookup = {normalize_key(clean_sheet_cell(r["Partida"])): r for _, r in cash_numeric.iterrows()}
            years = [c for c in series_cols]
            chart_df = pd.DataFrame({"Año": years})
            ingresos_row = row_lookup.get(normalize_key("Ingresos (USD)"), {})
            ebitda_row = row_lookup.get(normalize_key("EBITDA (USD)"), {})
            caja_row = cash_row_lookup.get(normalize_key("Flujo de caja neto"))
            if not isinstance(caja_row, pd.Series):
                caja_row = cash_row_lookup.get(normalize_key("Flujo caja neto"), {})
            chart_df["Ingresos"] = [ingresos_row.get(y, 0.0) if isinstance(ingresos_row, pd.Series) else 0.0 for y in years]
            chart_df["EBITDA"] = [ebitda_row.get(y, 0.0) if isinstance(ebitda_row, pd.Series) else 0.0 for y in years]
            chart_df["Caja_neta"] = [caja_row.get(y, 0.0) if isinstance(caja_row, pd.Series) else 0.0 for y in years]
            chart_df["Ingresos_MM"] = chart_df["Ingresos"] / 1e6
            chart_df["EBITDA_MM"] = chart_df["EBITDA"] / 1e6
            chart_df["Caja_MM"] = chart_df["Caja_neta"] / 1e6

            eerr_styler = style_engineering_table(eerr_data).apply(
                lambda row: [
                    "font-weight: 800;" if clean_sheet_cell(row.iloc[0]) in {"Margen bruto (USD)", "EBITDA (USD)"} else ""
                    for _ in row
                ],
                axis=1,
            )

            col_eerr_1, col_eerr_2 = st.columns([1.7, 1])
            with col_eerr_1:
                st.markdown('<div class="eng-section-label">Lectura financiera integrada</div>', unsafe_allow_html=True)
                st.markdown('#### Proyección Financiera Integrada " Etapa comercial"')
                st.dataframe(eerr_styler, hide_index=True, use_container_width=True, height=360)
            with col_eerr_2:
                st.markdown('<div class="eng-section-label">Drivers técnicos del modelo</div>', unsafe_allow_html=True)
                st.markdown("#### Drivers unitarios del modelo")
                drv_row_1 = st.columns(2)
                with drv_row_1[0]:
                    kpi_card("Precio venta / turbina", format_usd(precio_venta_turbina), "Supuesto comercial unitario del modelo.", variant="sky")
                with drv_row_1[1]:
                    kpi_card("Costo estimado / turbina", format_usd(costo_estimado_turbina), "Costo directo unitario usado en valorización.", variant="sky")
                drv_row_2 = st.columns(2)
                with drv_row_2[0]:
                    kpi_card("EBITDA unitario", format_usd(ebitda_unitario_val), "Margen operativo unitario por turbina.", variant="sky")
                with drv_row_2[1]:
                    kpi_card("CAPEX inicial", format_usd(capex_inicial_eerr), "Valor base tomado de EERRv2 celda C15.", variant="sky")

            st.markdown("#### Flujo de Caja del Proyecto y Estrategia de Reinversión")
            st.dataframe(style_engineering_table(cash_data), hide_index=True, use_container_width=True, height=420)
            st.markdown('<div class="eng-section-label">Desempeño consolidado</div>', unsafe_allow_html=True)
            st.markdown("### Desempeño Financiero y Operativo del Proyecto")
            mini_cards = [
                ("Ingresos promedio", kpi_map.get("Ingresos promedio (USD)", "-"), "Promedio anual del escenario EERR."),
                ("EBITDA promedio", kpi_map.get("EBITDA promedio (USD)", "-"), "Promedio anual operativo."),
                ("Margen EBITDA", kpi_map.get("Margen EBITDA promedio (%)", "-"), "Margen consolidado del modelo."),
                ("Saldo caja año 5", kpi_map.get("Saldo caja final año 5 (USD)", "-"), "Posición final estimada de caja."),
            ]
            mini_cols = st.columns(4)
            for idx, (title, value, subtitle) in enumerate(mini_cards):
                with mini_cols[idx]:
                    st.markdown(
                        f"""
                        <div class="eerr-mini">
                          <div class="eerr-mini-h">{title}</div>
                          <div class="eerr-mini-v">{value}</div>
                          <div class="eerr-mini-s">{subtitle}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            fig_eerr = go.Figure()
            fig_eerr.add_trace(go.Bar(x=chart_df["Año"], y=chart_df["Ingresos_MM"], name="Ingresos", marker_color="#CFE8DA", hovertemplate="Ingresos %{x}: US$%{customdata:,.0f}<extra></extra>", customdata=chart_df["Ingresos"]))
            fig_eerr.add_trace(go.Bar(x=chart_df["Año"], y=chart_df["EBITDA_MM"], name="EBITDA", marker_color="#0F766E", hovertemplate="EBITDA %{x}: US$%{customdata:,.0f}<extra></extra>", customdata=chart_df["EBITDA"]))
            fig_eerr.add_trace(go.Scatter(x=chart_df["Año"], y=chart_df["Caja_MM"], name="Flujo caja neto", mode="lines+markers", line=dict(color="#1D4ED8", width=3), marker=dict(size=9, color="#1D4ED8"), hovertemplate="Caja neta %{x}: US$%{customdata:,.0f}<extra></extra>", customdata=chart_df["Caja_neta"], yaxis="y2"))
            fig_eerr.update_layout(title="Trayectoria operativa del modelo EERRv2", barmode="group", height=420, margin=dict(l=10, r=10, t=60, b=10), plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0), yaxis=dict(title="Ingresos / EBITDA (MM USD)", showgrid=True, gridcolor="rgba(148,163,184,0.22)", zeroline=False), yaxis2=dict(title="Caja neta (MM USD)", overlaying="y", side="right", showgrid=False, zeroline=False))
            st.plotly_chart(fig_eerr, use_container_width=True)

    elif bloque_sel == "2. Serie A: Inversión Inicial y Validación":
        selected_basis_label = "BASE INVERSION + KNOW-HOW" if valuation_basis_input == "BASE INVERSION + KNOW-HOW" else "EBITDA potencial ciclo inicial"
        post_money_serie_a_clp = post_money_serie_a * fx_input
        if investment_currency_input == "CLP":
            metric_cols = st.columns(3)
            mk2, mk3, mk4 = metric_cols
            with mk2:
                kpi_card("Post-money Serie A", format_clp(post_money_serie_a_clp), "Valorización posterior al ingreso para construir el piloto.")
        else:
            metric_cols = st.columns(4)
            mk1, mk2, mk3, mk4 = metric_cols
            with mk1:
                kpi_card("Inv. convertida a USD", format_usd(inversion_usd), "Capital de entrada del piloto convertido con el FX editable.")
            with mk2:
                kpi_card("Post-money Serie A", format_usd(post_money_serie_a), "Valorización posterior al ingreso para construir el piloto.")
        with mk3:
            kpi_card("% IMELSA", f"{imelsa_pct:.1%}", "Participación posterior al ingreso de capital.")
        with mk4:
            kpi_card("% Fluxial", f"{fluxial_pct:.1%}", "Participación remanente posterior al piloto.")
        st.markdown(
            """
            <style>
            .seriea-panel{
                border-radius:22px;
                padding:16px 18px 14px 18px;
                border:1px solid rgba(148,163,184,.20);
                background:linear-gradient(180deg,#ffffff 0%,#f8fbff 100%);
                box-shadow:0 10px 24px rgba(15,23,42,.05);
                height:100%;
            }
            .seriea-panel-k{
                font-size:11px;
                font-weight:800;
                letter-spacing:.12em;
                text-transform:uppercase;
                color:#64748B;
                margin-bottom:8px;
            }
            .seriea-panel-t{
                font-size:16px;
                font-weight:800;
                color:#0f172a;
                line-height:1.3;
                margin-bottom:10px;
            }
            .seriea-panel-s{
                font-size:13px;
                line-height:1.5;
                color:#475569;
            }
            .seriea-shell{
                border-radius:24px;
                padding:18px 18px 14px 18px;
                border:1px solid rgba(148,163,184,.18);
                background:linear-gradient(180deg,#ffffff 0%,#f8fbff 100%);
                box-shadow:0 12px 26px rgba(15,23,42,.05);
                min-height:100%;
            }
            .seriea-shell-k{
                font-size:11px;
                font-weight:800;
                letter-spacing:.12em;
                text-transform:uppercase;
                color:#64748B;
                margin-bottom:8px;
            }
            .seriea-shell-t{
                font-size:22px;
                font-weight:800;
                color:#0f172a;
                line-height:1.15;
                margin-bottom:8px;
            }
            .seriea-shell-s{
                font-size:13px;
                line-height:1.55;
                color:#475569;
                margin-bottom:12px;
            }
            .seriea-mini-band{
                display:flex;
                gap:10px;
                flex-wrap:wrap;
                margin:10px 0 14px 0;
            }
            .seriea-mini-chip{
                border-radius:999px;
                padding:8px 12px;
                border:1px solid rgba(29,78,216,.12);
                background:#F8FBFF;
                font-size:12px;
                color:#334155;
            }
            .seriea-foot{
                border-top:1px solid rgba(148,163,184,.16);
                padding-top:10px;
                margin-top:10px;
                font-size:12px;
                color:#64748B;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        if investment_currency_input == "CLP":
            serie_a = pd.DataFrame([
                {"Métrica": f"Valorización base ({selected_basis_label})", "Valor": format_clp(valorizacion_fluxial_hoy * fx_input)},
                {"Métrica": "Inversión piloto (CLP)", "Valor": format_clp(inversion_clp_input)},
                {"Métrica": "Post-money", "Valor": format_clp(post_money_serie_a_clp)},
                {"Métrica": "% IMELSA", "Valor": f"{imelsa_pct:.1%}"},
                {"Métrica": "% Fluxial", "Valor": f"{fluxial_pct:.1%}"},
            ])
        else:
            serie_a = pd.DataFrame([
                {"Métrica": f"Valorización base ({selected_basis_label})", "Valor": format_usd(valorizacion_fluxial_hoy)},
                {"Métrica": "Inversión piloto (USD)", "Valor": format_usd(inversion_usd)},
                {"Métrica": "Post-money", "Valor": format_usd(post_money_serie_a)},
                {"Métrica": "% IMELSA", "Valor": f"{imelsa_pct:.1%}"},
                {"Métrica": "% Fluxial", "Valor": f"{fluxial_pct:.1%}"},
            ])
        if alloc_manual_input:
            serie_a = pd.concat(
                [
                    serie_a,
                    pd.DataFrame(
                        [
                            {
                                "Métrica": "Valor complementario (CLP)" if investment_currency_input == "CLP" else "Valor complementario (USD)",
                                "Valor": format_clp(aporte_no_pecuniario_clp) if investment_currency_input == "CLP" else format_usd(aporte_no_pecuniario_usd),
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
        cap_table_a = pd.DataFrame({
            "Socio": ["Fluxial Wind", "IMELSA"],
            "Participación": [fluxial_pct, imelsa_pct],
            "Valor implícito (USD)": [post_money_serie_a * fluxial_pct, post_money_serie_a * imelsa_pct],
        }).sort_values("Participación", ascending=True).reset_index(drop=True)
        cap_table_a["Participación_pct"] = cap_table_a["Participación"] * 100
        cap_table_a["Etiqueta"] = cap_table_a.apply(
            lambda r: f"{r['Participación']:.1%} · {format_usd(r['Valor implícito (USD)'])}",
            axis=1,
        )
        c1, c2 = st.columns([0.95, 1.05])
        with c1:
            st.markdown(
                f"""
                <div class="seriea-shell">
                  <div class="seriea-shell-k">Lectura económica</div>
                  <div class="seriea-shell-t">Resumen de entrada Serie A</div>
                  <div class="seriea-shell-s">Conversión de inversión, valorización base activa, post-money y estructura societaria proyectada tras el ingreso de capital.</div>
                  <div class="seriea-mini-band">
                    <div class="seriea-mini-chip">Base activa: {selected_basis_label}</div>
                    <div class="seriea-mini-chip">Post-money: {format_usd(post_money_serie_a)}</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
            st.dataframe(style_engineering_table(serie_a), hide_index=True, use_container_width=True, height=310)
            st.markdown(
                f"""
                <div class="seriea-foot">
                  La tabla resume la valorización base seleccionada, el capital invertido y la distribución accionaria resultante.
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            fig_cap_a = px.bar(
                cap_table_a,
                x="Participación_pct",
                y="Socio",
                orientation="h",
                text="Etiqueta",
                color="Socio",
                color_discrete_map={"Fluxial Wind": "#1D4ED8", "IMELSA": "#0F766E"},
                title=None,
            )
            fig_cap_a.update_traces(
                textposition="inside",
                insidetextanchor="middle",
                marker=dict(line=dict(color="rgba(255,255,255,0.85)", width=1.2)),
                hovertemplate="<b>%{y}</b><br>Participación: %{customdata[0]:.1%}<br>Valor implícito: US$%{customdata[1]:,.0f}<extra></extra>",
                customdata=np.stack([cap_table_a["Participación"], cap_table_a["Valor implícito (USD)"]], axis=-1),
            )
            fig_cap_a.update_layout(
                showlegend=False,
                xaxis_tickformat=".0f",
                margin=dict(l=8, r=8, t=56, b=8),
                height=330,
                plot_bgcolor="white",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#334155", size=13),
                title=None,
            )
            fig_cap_a.update_xaxes(
                title="Participación accionaria (%)",
                ticksuffix="%",
                range=[0, max(100, cap_table_a["Participación_pct"].max() * 1.08)],
                showgrid=True,
                gridcolor="rgba(148,163,184,0.22)",
                zeroline=False,
            )
            fig_cap_a.update_yaxes(title="", showgrid=False)
            st.markdown(
                f"""
                <div class="seriea-shell">
                  <div class="seriea-shell-k">Cap table proyectado</div>
                  <div class="seriea-shell-t">Distribución posterior al piloto</div>
                  <div class="seriea-shell-s">Lectura visual de participación y valor implícito por socio después del cierre de la Serie A.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.plotly_chart(fig_cap_a, use_container_width=True, config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})
            st.markdown(
                f"""
                <div class="seriea-foot">
                  Fluxial retiene {fluxial_pct:.1%} e IMELSA consolida {imelsa_pct:.1%} del cap table post-piloto.
                </div>
                """,
                unsafe_allow_html=True,
            )

    elif bloque_sel == "3. Valorización Post-Validación":
        mk1, mk2, mk3, mk4, mk5 = st.columns(5)
        if investment_currency_input == "CLP":
            pre_money_display = format_clp(base_post_money_serie_a_clp)
            ebitda_total_display = format_clp(ebitda_total_clp)
            valor_post_piloto_display = format_clp(valor_post_piloto_clp)
            valor_imelsa_display = format_clp(valor_imelsa_post_clp)
            sens_title = "Sensibilidad de valorización vs volumen comercial (MM CLP)"
            sens_label = "Valorización post piloto (MM CLP)"
            sens_tickprefix = "$"
            sens_values = (base_post_money_serie_a_clp + (sensibilidad["Volumen"] * ebitda_unit_input * multiple_input * fx_input)) if False else None
        else:
            pre_money_display = format_usd(base_post_money_serie_a)
            ebitda_total_display = format_usd(ebitda_total)
            valor_post_piloto_display = format_usd(valor_post_piloto)
            valor_imelsa_display = format_usd(valor_imelsa_post)
            sens_title = "Sensibilidad de valorización vs volumen comercial"
            sens_label = "Valorización post piloto (MM USD)"
            sens_tickprefix = "US$"
        with mk1:
            kpi_card(f"Pre Money actual ({investment_currency_input})", pre_money_display, "Valor base heredado desde Post-money Serie A del bloque 2.")
        with mk2:
            kpi_card(f"EBITDA total ({investment_currency_input})", ebitda_total_display, "EBITDA unitario multiplicado por el volumen.")
        with mk3:
            kpi_card("Valorización post piloto", valor_post_piloto_display, "EBITDA total multiplicado por el múltiplo activo.", variant="sky")
        with mk4:
            kpi_card("Upside vs actual", f"{upside_pct:.1%}", "Crecimiento de valorización post piloto respecto de Pre Money actual.")
        with mk5:
            kpi_card("Valor por 50% post piloto", valor_imelsa_display, "Valor implícito de una participación equivalente al 50% post piloto.")
        st.markdown("#### Motor económico del piloto")
        sensibilidad = pd.DataFrame({"Volumen": [max(1, volume_input * f) for f in [0.6, 0.8, 1.0, 1.2, 1.4]]})
        sensibilidad["Valorización_post_piloto"] = sensibilidad["Volumen"] * ebitda_unit_input * multiple_input
        if investment_currency_input == "CLP":
            sensibilidad["Valorización_millones"] = (sensibilidad["Valorización_post_piloto"] * fx_input) / 1e6
            sensibilidad["Etiqueta"] = sensibilidad["Valorización_millones"].map(lambda v: f"${v:.2f}M")
            sens_customdata = np.stack([sensibilidad["Valorización_post_piloto"] * fx_input, sensibilidad["Volumen"] * ebitda_unit_input * fx_input], axis=-1)
            sens_hover = "<b>Volumen:</b> %{x:.0f} turbinas<br><b>Valorización:</b> $%{customdata[0]:,.0f}<br><b>EBITDA total:</b> $%{customdata[1]:,.0f}<extra></extra>"
            sens_base_y = valor_post_piloto_clp / 1e6
        else:
            sensibilidad["Valorización_millones"] = sensibilidad["Valorización_post_piloto"] / 1e6
            sensibilidad["Etiqueta"] = sensibilidad["Valorización_millones"].map(lambda v: f"US${v:.2f}M")
            sens_customdata = np.stack([sensibilidad["Valorización_post_piloto"], sensibilidad["Volumen"] * ebitda_unit_input], axis=-1)
            sens_hover = "<b>Volumen:</b> %{x:.0f} turbinas<br><b>Valorización:</b> US$%{customdata[0]:,.0f}<br><b>EBITDA total:</b> US$%{customdata[1]:,.0f}<extra></extra>"
            sens_base_y = valor_post_piloto / 1e6
        fig_sens = px.line(sensibilidad, x="Volumen", y="Valorización_millones", markers=True, text="Etiqueta", title=sens_title, labels={"Valorización_millones": sens_label, "Volumen": "Volumen comercial (turbinas)"})
        fig_sens.update_traces(line=dict(color="#0F766E", width=4), marker=dict(size=10, color="#0F766E", line=dict(width=2, color="#ECFDF5")), textposition="top center", hovertemplate=sens_hover, customdata=sens_customdata)
        fig_sens.add_vline(x=volume_input, line_width=1.5, line_dash="dash", line_color="#1D4ED8", opacity=0.8)
        fig_sens.add_hline(y=sens_base_y, line_width=1.5, line_dash="dot", line_color="#1D4ED8", opacity=0.8)
        fig_sens.add_annotation(x=volume_input, y=sens_base_y, text=(f"Base: {volume_input:,.0f} turbinas / ${sens_base_y:.2f}M" if investment_currency_input == "CLP" else f"Base: {volume_input:,.0f} turbinas / US${sens_base_y:.2f}M").replace(",", "."), showarrow=True, arrowhead=2, ax=40, ay=-40, bgcolor="rgba(255,255,255,0.95)", bordercolor="#BFDBFE", font=dict(size=11, color="#0F172A"))
        fig_sens.update_layout(margin=dict(l=10, r=10, t=60, b=10), height=430, plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)", hovermode="x unified")
        fig_sens.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.22)", zeroline=False)
        fig_sens.update_yaxes(tickprefix=sens_tickprefix, ticksuffix="M", showgrid=True, gridcolor="rgba(148,163,184,0.22)", zeroline=False)
        st.plotly_chart(fig_sens, use_container_width=True)

    else:
        mk1, mk2, mk3, mk4 = st.columns(4)
        if investment_currency_input == "CLP":
            val_base_b_display = format_clp(valor_post_piloto_clp)
            capital_b_display = format_clp(capital_raise_clp)
            post_money_b_display = format_clp(post_money_serie_b_clp)
        else:
            val_base_b_display = format_usd(valor_post_piloto)
            capital_b_display = format_usd(capital_raise)
            post_money_b_display = format_usd(post_money_serie_b)
        with mk1:
            kpi_card("Valorización base Serie B", val_base_b_display, "Pre-money sugerido para la segunda ronda.")
        with mk2:
            kpi_card("Capital Serie B", capital_b_display, "Capital implícito a levantar para la nueva cesión objetivo.")
        with mk3:
            kpi_card("Post-money Serie B", post_money_b_display, "Valorización posterior al cierre de la ronda.", variant="sky")
        with mk4:
            kpi_card("% remanente socios actuales", f"{pct_remanente:.1%}", "Participación conjunta remanente de Fluxial + IMELSA.")
        st.markdown(
            """
            <style>
            .serieb-shell{
                border-radius:24px;
                padding:18px 18px 14px 18px;
                border:1px solid rgba(148,163,184,.18);
                background:linear-gradient(180deg,#ffffff 0%,#f8fbff 100%);
                box-shadow:0 12px 26px rgba(15,23,42,.05);
                min-height:100%;
            }
            .serieb-shell-k{
                font-size:11px;
                font-weight:800;
                letter-spacing:.12em;
                text-transform:uppercase;
                color:#64748B;
                margin-bottom:8px;
            }
            .serieb-shell-t{
                font-size:22px;
                font-weight:800;
                color:#0f172a;
                line-height:1.15;
                margin-bottom:8px;
            }
            .serieb-shell-s{
                font-size:13px;
                line-height:1.55;
                color:#475569;
                margin-bottom:12px;
            }
            .serieb-mini-band{
                display:flex;
                gap:10px;
                flex-wrap:wrap;
                margin:10px 0 14px 0;
            }
            .serieb-mini-chip{
                border-radius:999px;
                padding:8px 12px;
                border:1px solid rgba(15,118,110,.12);
                background:#F8FBFF;
                font-size:12px;
                color:#334155;
            }
            .serieb-foot{
                border-top:1px solid rgba(148,163,184,.16);
                padding-top:10px;
                margin-top:10px;
                font-size:12px;
                color:#64748B;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        cap_table_b = pd.DataFrame({"Socio": ["Nuevos inversionistas", "Fluxial Wind", "IMELSA"], "Participación": [ronda_pct_input, fluxial_post_b, imelsa_post_b], "Valor económico": [post_money_serie_b * ronda_pct_input, valor_fluxial_post_b, valor_imelsa_post_b]}).sort_values("Participación", ascending=True).reset_index(drop=True)
        if investment_currency_input == "CLP":
            cap_table_b["Valor económico"] = cap_table_b["Valor económico"] * fx_input
        cap_table_b["Participación_pct"] = cap_table_b["Participación"] * 100
        cap_table_b["Etiqueta"] = cap_table_b.apply(lambda r: f"{r['Participación']:.1%}  ·  {(format_clp(r['Valor económico']) if investment_currency_input == 'CLP' else format_usd(r['Valor económico']))}", axis=1)
        fig_cap_b = px.bar(cap_table_b, x="Participación_pct", y="Socio", orientation="h", text="Etiqueta", color="Socio", color_discrete_map={"Nuevos inversionistas": "#C58940", "Fluxial Wind": "#1D4ED8", "IMELSA": "#0F766E"}, title=None)
        fig_cap_b.update_traces(textposition="inside", insidetextanchor="middle", marker=dict(line=dict(color="rgba(255,255,255,0.85)", width=1.2)), hovertemplate=("<b>%{y}</b><br>Participación: %{customdata[0]:.1%}<br>Valor económico: $%{customdata[1]:,.0f}<extra></extra>" if investment_currency_input == "CLP" else "<b>%{y}</b><br>Participación: %{customdata[0]:.1%}<br>Valor económico: US$%{customdata[1]:,.0f}<extra></extra>"), customdata=np.stack([cap_table_b["Participación"], cap_table_b["Valor económico"]], axis=-1))
        fig_cap_b.update_layout(showlegend=False, margin=dict(l=8, r=8, t=18, b=16), height=360, bargap=0.22, plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#334155", size=13), title=None)
        fig_cap_b.update_xaxes(title="Participación accionaria (%)", ticksuffix="%", range=[0, max(100, cap_table_b["Participación_pct"].max() * 1.08)], showgrid=True, gridcolor="rgba(148,163,184,0.22)", zeroline=False)
        fig_cap_b.update_yaxes(title="", showgrid=False)
        if investment_currency_input == "CLP":
            serie_b_rows = pd.DataFrame([
                {"Métrica": "Pre-money Serie B (CLP)", "Valor": format_clp(valor_post_piloto_clp)},
                {"Métrica": "Capital a levantar (CLP)", "Valor": format_clp(capital_raise_clp)},
                {"Métrica": "Post-money Serie B (CLP)", "Valor": format_clp(post_money_serie_b_clp)},
                {"Métrica": "% Fluxial post ronda", "Valor": f"{fluxial_post_b:.1%}"},
                {"Métrica": "% IMELSA post ronda", "Valor": f"{imelsa_post_b:.1%}"},
            ])
        else:
            serie_b_rows = pd.DataFrame([
                {"Métrica": "Pre-money Serie B (USD)", "Valor": format_usd(valor_post_piloto)},
                {"Métrica": "Capital a levantar (USD)", "Valor": format_usd(capital_raise)},
                {"Métrica": "Post-money Serie B (USD)", "Valor": format_usd(post_money_serie_b)},
                {"Métrica": "% Fluxial post ronda", "Valor": f"{fluxial_post_b:.1%}"},
                {"Métrica": "% IMELSA post ronda", "Valor": f"{imelsa_post_b:.1%}"},
            ])
        c1, c2 = st.columns([0.95, 1.05])
        with c1:
            st.markdown(
                f"""
                <div class="serieb-shell">
                  <div class="serieb-shell-k">Lectura de ronda</div>
                  <div class="serieb-shell-t">Resumen económico Serie B</div>
                  <div class="serieb-shell-s">La tabla consolida pre-money, capital nuevo, post-money y la estructura accionaria remanente tras la expansión comercial.</div>
                  <div class="serieb-mini-band">
                    <div class="serieb-mini-chip">Moneda visible: {investment_currency_input}</div>
                    <div class="serieb-mini-chip">Nueva cesión: {ronda_pct_input:.1%}</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
            st.dataframe(style_engineering_table(serie_b_rows), hide_index=True, use_container_width=True, height=310)
            st.markdown(
                """
                <div class="serieb-foot">
                  La tabla resume la valorización base de la ronda, el capital nuevo comprometido y la distribución accionaria posterior al cierre.
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                """
                <div class="serieb-shell">
                  <div class="serieb-shell-k">Cap table proyectado</div>
                  <div class="serieb-shell-t">Distribución posterior a Serie B</div>
                  <div class="serieb-shell-s">Lectura visual de participación y valor económico por socio después de la nueva cesión para escalamiento.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.plotly_chart(fig_cap_b, use_container_width=True, config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})
            st.markdown(
                f"""
                <div class="serieb-foot">
                  Nuevos inversionistas concentran {ronda_pct_input:.1%}, mientras Fluxial e IMELSA retienen {fluxial_post_b:.1%} y {imelsa_post_b:.1%} respectivamente.
                </div>
                """,
                unsafe_allow_html=True,
            )

    csv_valorizacion = df_valorizacion.to_csv(index=False).encode("utf-8-sig")
    if REPORTLAB_AVAILABLE:
        pdf_export_slot.download_button(
            label="Descargar PDF de supuestos",
            data=build_valorizacion_supuestos_pdf(),
            file_name="Supuestos_Modelo_Valorizacion.pdf",
            mime="application/pdf",
            key=widget_key("download_supuestos_pdf"),
            use_container_width=True,
        )
    else:
        pdf_export_slot.info("PDF deshabilitado: falta `reportlab`.", icon="ℹ️")
    st.download_button(label="📥 Descargar CSV de valorización", data=csv_valorizacion, file_name="valorizacion.csv", mime="text/csv", key=widget_key("download_csv"))


def render_explorador_module_content(key_prefix: str = "explorer_"):
    st.subheader("Explorador interactivo de la tabla CAPEX")

    col_f1, col_f2, col_f3, col_f4 = st.columns(4)

    with col_f1:
        categoria_filter = st.selectbox(
            "Filtrar por categoría:",
            options=["(Todas)"] + sorted(df_capex["Categoria"].unique().tolist()),
            index=0,
            key=f"{key_prefix}categoria_filter",
        )

    with col_f2:
        item_filter = st.selectbox(
            "Filtrar por ítem:",
            options=["(Todos)"] + sorted(df_capex["Item"].unique().tolist()),
            index=0,
            key=f"{key_prefix}item_filter",
        )

    with col_f3:
        min_pct = st.slider(
            "Participación mínima del ítem (%)",
            min_value=0.0,
            max_value=5.0,
            value=0.0,
            step=0.1,
            key=f"{key_prefix}min_pct",
        )

    with col_f4:
        ordenar_por = st.selectbox(
            "Ordenar por:",
            options=["Monto_CLP", "Monto_USD", "Participacion_pct"],
            index=0,
            key=f"{key_prefix}ordenar_por",
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
        key=f"{key_prefix}download_csv",
    )


def render_input_thousands_hint(value: float | int, prefix: str = ""):
    try:
        number = int(round(float(value)))
        formatted = f"{number:,}".replace(",", ".")
        label = f"{prefix}{formatted}" if prefix else formatted
        st.caption(f"Valor formateado: {label}")
    except Exception:
        pass


# =========================
# NAVEGACIÓN PRINCIPAL
# =========================
input_cards = [
    ("estado_actual", "1- Activo Tecnológico y Validación"),
    ("escalamiento", "2- Capital Requerido y Ejecución CAPEX"),
    ("valorizacion", "3-Modelo de Valorización y Estructura de Inversión"),
]

if "inputs_bloque_sel" not in st.session_state:
    st.session_state["inputs_bloque_sel"] = None

def _set_inputs_bloque(value: str):
    st.session_state["inputs_bloque_sel"] = value
    if value == "estado_actual":
        st.session_state["inputs_estado_actual_subbloque_sel"] = None
    elif value == "escalamiento":
        st.session_state["inputs_escalamiento_capex_sel"] = None
    elif value == "valorizacion":
        st.session_state["inputs_val_bloque_sel"] = None

st.markdown(
        """
        <style>
        .inputs-nav-shell{
            border-radius:24px;
            padding:16px 18px 12px 18px;
            border:1px solid rgba(148,163,184,.18);
            background:linear-gradient(180deg,#ffffff 0%,#f8fbff 100%);
            box-shadow:0 12px 28px rgba(15,23,42,.05);
            margin-bottom:18px;
        }
        .inputs-nav-head-k{
            font-size:11px;
            font-weight:800;
            letter-spacing:.12em;
            text-transform:uppercase;
            color:#64748B;
            margin-bottom:6px;
        }
        .inputs-nav-head-t{
            font-size:18px;
            font-weight:900;
            line-height:1.15;
            color:#0f172a;
            margin-bottom:6px;
        }
        .inputs-nav-head-s{
            font-size:13px;
            line-height:1.5;
            color:#475569;
            max-width:760px;
        }
        .inputs-nav-card{
            min-height:118px;
            display:flex;
            flex-direction:column;
            justify-content:flex-start;
            position:relative;
            overflow:hidden;
            border-radius:20px;
            padding:16px 18px 14px 18px;
            border:1px solid rgba(203,213,225,.75);
            background:
                radial-gradient(circle at top right, rgba(191,219,254,.28), transparent 28%),
                linear-gradient(180deg,#f8fafc 0%,#ffffff 74%);
            box-shadow:0 10px 22px rgba(15,23,42,.04);
            margin-bottom:10px;
        }
        .inputs-nav-card:before{
            content:"";
            position:absolute;
            left:0;
            top:0;
            bottom:0;
            width:5px;
            background:linear-gradient(180deg,#cbd5e1 0%,#e2e8f0 100%);
        }
        .inputs-nav-card.active{
            border:1px solid rgba(16,185,129,.30);
            box-shadow:0 16px 34px rgba(16,185,129,.14);
            background:
                radial-gradient(circle at top right, rgba(52,211,153,.20), transparent 26%),
                linear-gradient(90deg,#ecfdf5 0%,#d1fae5 48%,#a7f3d0 100%);
        }
        .inputs-nav-card.active:before{
            background:linear-gradient(180deg,#059669 0%,#34d399 100%);
        }
        .inputs-nav-k{
            font-size:11px;
            font-weight:800;
            letter-spacing:.10em;
            text-transform:uppercase;
            color:#64748B;
            margin-bottom:10px;
        }
        .inputs-nav-t{
            font-size:17px;
            font-weight:900;
            line-height:1.2;
            color:#0f172a;
            margin-bottom:8px;
            max-width:28ch;
        }
        .inputs-nav-title-row{
            display:flex;
            align-items:flex-start;
            gap:10px;
            margin-bottom:8px;
        }
        .inputs-nav-ico{
            font-size:22px;
            line-height:1;
            flex:0 0 auto;
            margin-top:1px;
        }
        .inputs-nav-title-wrap{
            min-width:0;
        }
        .inputs-nav-s{
            font-size:12px;
            line-height:1.45;
            color:#475569;
        }
        .inputs-nav-card.active .inputs-nav-k{color:#166534;}
        .inputs-nav-card.active .inputs-nav-t{color:#064e3b;}
        .inputs-nav-card.active .inputs-nav-s{color:#065f46;}
        .inputs-active-banner{
            border-radius:18px;
            padding:12px 14px;
            border:1px solid rgba(59,130,246,.16);
            background:linear-gradient(90deg,#eff6ff 0%,#eefbf5 100%);
            margin:8px 0 4px 0;
        }
        .inputs-active-banner-k{
            font-size:10px;
            font-weight:800;
            letter-spacing:.12em;
            text-transform:uppercase;
            color:#64748b;
            margin-bottom:4px;
        }
        .inputs-active-banner-t{
            font-size:14px;
            font-weight:800;
            color:#0f172a;
        }
        .inputs-info-box{
            border-radius:20px;
            padding:22px 24px;
            border:1px solid rgba(148,163,184,.24);
            background:linear-gradient(180deg,#ffffff 0%,#f8fafc 100%);
            box-shadow:0 10px 24px rgba(15,23,42,.05);
        }
        .inputs-info-k{
            font-size:11px;
            font-weight:800;
            letter-spacing:.12em;
            text-transform:uppercase;
            color:#64748B;
            margin-bottom:8px;
        }
        .inputs-info-t{
            font-size:28px;
            font-weight:800;
            line-height:1.1;
            color:#0f172a;
            margin-bottom:10px;
        }
        .inputs-info-p{
            font-size:15px;
            line-height:1.6;
            color:#475569;
            margin:0;
        }
        </style>
        """,
        unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="inputs-nav-shell">
      <div class="inputs-nav-head-k">Mapa de lectura</div>
      <div class="inputs-nav-head-t">Selecciona el bloque estratégico que quieres revisar</div>
      <div class="inputs-nav-head-s">La pantalla está organizada en tres vistas: activo tecnológico, capital requerido y estructura de valorización.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

nav_cols = st.columns(3)
for idx, (block_value, block_title) in enumerate(input_cards):
    is_active = st.session_state.get("inputs_bloque_sel") == block_value
    with nav_cols[idx]:
        st.markdown(
            f"""
            <div class="inputs-nav-card {'active' if is_active else ''}">
                <div class="inputs-nav-k">BLOQUE {idx + 1}</div>
                <div class="inputs-nav-t">{block_title}</div>
                <div class="inputs-nav-s">{'Vista activa para análisis' if is_active else 'Abrir vista estratégica'}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.button(
            "Seleccionado" if is_active else "Abrir bloque",
            key=f"inputs_nav_{idx}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
            on_click=_set_inputs_bloque,
            args=(block_value,),
        )

st.markdown("---")
selected_input_block = st.session_state.get("inputs_bloque_sel")

if selected_input_block:
    st.markdown(
        """
        <div id="inputs-subblocks-anchor"></div>
        <script>
        const subblockAnchor = window.parent.document.getElementById("inputs-subblocks-anchor");
        if (subblockAnchor) {
          subblockAnchor.scrollIntoView({ behavior: "smooth", block: "start" });
        }
        </script>
        """,
        unsafe_allow_html=True,
    )

input_block_copy = {
    "estado_actual": (
        "1- Activo Tecnológico y Validación",
        "Aquí consolidaremos el diagnóstico base del proyecto: situación actual, hitos alcanzados, brechas técnicas y supuestos iniciales del modelo.",
    ),
    "escalamiento": (
        "2- Capital Requerido y Ejecución CAPEX",
        "Este bloque quedará preparado para mostrar la ruta de escalamiento industrial, prioridades de inversión y asignación de fondos por etapa.",
    ),
    "valorizacion": (
        "3-Modelo de Valorización y Estructura de Inversión",
        "En esta sección vamos a integrar la lectura financiera de valorización, estructura de capital y criterios de inversión asociados al modelo.",
    ),
}
if selected_input_block == "estado_actual":
    render_inputs_estado_actual_dashboard()
elif selected_input_block == "escalamiento":
    capex_selector_state_key = "inputs_escalamiento_capex_sel"

    def _set_capex_focus(value: str):
        st.session_state[capex_selector_state_key] = value

    if capex_selector_state_key not in st.session_state:
        st.session_state[capex_selector_state_key] = None

    capex_10kw_val = 0.0
    try:
        df_restante_10kw = build_restante_piloto_10kw_view(RESTANTE_PILOTO_10KW_CSV_URL_DEFAULT, refresh_nonce=data_refresh_nonce)
        if not df_restante_10kw.empty:
            capex_10kw_val = float(df_restante_10kw["Valor C"].sum() or 0.0)
    except Exception:
        capex_10kw_val = 0.0

    capex_80kw_usd_total = float(df_capex_base["Monto_USD"].sum() or 0.0) if "Monto_USD" in df_capex_base.columns else 0.0
    capex_80kw_val = (capex_80kw_usd_total * fx_used) + float(direccion_total_clp or 0.0)
    capital_recaudar_val = capex_10kw_val + capex_80kw_val

    capex_10kw_active = st.session_state.get(capex_selector_state_key) == "10kw"
    capex_80kw_active = st.session_state.get(capex_selector_state_key) == "80kw"
    st.markdown(
        """
        <style>
        .capex-summary-hero{
            border-radius:24px;
            padding:22px 24px;
            background:
                radial-gradient(circle at top right, rgba(14,165,164,.16), transparent 24%),
                linear-gradient(90deg,#f8fbff 0%,#e7f5ff 48%,#d4efff 100%);
            border:1px solid rgba(125,211,252,.42);
            box-shadow:0 16px 36px rgba(15,23,42,.08);
            margin-bottom:18px;
        }
        .capex-summary-grid{
            display:grid;
            grid-template-columns:1.25fr .95fr;
            gap:18px;
            align-items:stretch;
        }
        @media (max-width:1100px){.capex-summary-grid{grid-template-columns:1fr;}}
        .capex-summary-k{
            font-size:11px;
            font-weight:800;
            letter-spacing:.14em;
            text-transform:uppercase;
            color:#0f766e;
            margin-bottom:8px;
        }
        .capex-summary-t{
            font-size:18px;
            font-weight:800;
            line-height:1.2;
            color:#0f172a;
            margin-bottom:10px;
        }
        .capex-summary-v{
            font-size:52px;
            font-weight:900;
            line-height:1;
            color:#0f172a;
            margin-bottom:12px;
            letter-spacing:-.03em;
        }
        .capex-summary-p{
            font-size:15px;
            line-height:1.6;
            color:#475569;
            max-width:720px;
        }
        .capex-summary-panel{
            border-radius:18px;
            padding:16px 18px;
            background:rgba(255,255,255,.76);
            border:1px solid rgba(148,163,184,.24);
            backdrop-filter:blur(6px);
        }
        .capex-summary-panel-h{
            font-size:12px;
            font-weight:800;
            letter-spacing:.10em;
            text-transform:uppercase;
            color:#64748b;
            margin-bottom:10px;
        }
        .capex-summary-row{
            display:flex;
            justify-content:space-between;
            gap:12px;
            align-items:flex-start;
            padding:10px 0;
            border-bottom:1px solid rgba(226,232,240,.9);
        }
        .capex-summary-row:last-child{border-bottom:none;padding-bottom:0}
        .capex-summary-label{
            font-size:14px;
            font-weight:700;
            color:#0f172a;
            line-height:1.35;
        }
        .capex-summary-value{
            font-size:16px;
            font-weight:800;
            color:#0f172a;
            white-space:nowrap;
        }
        .capex-detail-grid{
            display:grid;
            grid-template-columns:1fr 1fr;
            gap:16px;
            margin-bottom:10px;
        }
        @media (max-width:900px){.capex-detail-grid{grid-template-columns:1fr;}}
        .capex-detail-card{
            border-radius:20px;
            padding:18px 18px 16px 18px;
            background:linear-gradient(180deg,#ffffff 0%,#f8fafc 100%);
            border:1px solid rgba(148,163,184,.26);
            box-shadow:0 10px 24px rgba(15,23,42,.05);
        }
        .capex-detail-card.active{
            background:linear-gradient(90deg,#ecfdf5 0%,#d1fae5 42%,#a7f3d0 100%);
            border:1px solid rgba(22,163,74,.30);
            box-shadow:0 14px 28px rgba(21,128,61,.10);
        }
        .capex-detail-k{
            font-size:11px;
            font-weight:800;
            letter-spacing:.12em;
            text-transform:uppercase;
            color:#64748b;
            margin-bottom:8px;
        }
        .capex-detail-card.active .capex-detail-k{color:#166534;}
        .capex-detail-t{
            font-size:26px;
            font-weight:900;
            line-height:1;
            color:#0f172a;
            margin-bottom:10px;
        }
        .capex-detail-s{
            font-size:14px;
            line-height:1.5;
            color:#475569;
        }
        .capex-detail-card.active .capex-detail-s{color:#065f46;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="capex-summary-hero">
          <div class="capex-summary-grid">
            <div>
              <div class="capex-summary-k">Resumen de capital</div>
              <div class="capex-summary-t">Capital a Recaudar</div>
              <div class="capex-summary-v">{format_clp(capital_recaudar_val)}</div>
              <div class="capex-summary-p">
                Monto consolidado requerido para cubrir la referencia de CAPEX 10kW y el CAPEX integrado del piloto 80kW.
              </div>
            </div>
            <div class="capex-summary-panel">
              <div class="capex-summary-panel-h">Composición del capital</div>
              <div class="capex-summary-row">
                <div class="capex-summary-label">BRECHA PILOTO 10 KW</div>
                <div class="capex-summary-value">{format_clp(capex_10kw_val)}</div>
              </div>
              <div class="capex-summary-row">
                <div class="capex-summary-label">ESCALAMIENTO 80 KW</div>
                <div class="capex-summary-value">{format_clp(capex_80kw_val)}</div>
              </div>
              <div class="capex-summary-row">
                <div class="capex-summary-label">Capital consolidado</div>
                <div class="capex-summary-value">{format_clp(capital_recaudar_val)}</div>
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="capex-detail-grid">
          <div class="capex-detail-card {'active' if capex_10kw_active else ''}">
            <div class="capex-detail-k">BRECHA PILOTO 10 KW</div>
            <div class="capex-detail-t">{format_clp(capex_10kw_val)}</div>
            <div class="capex-detail-s">CAPEX 10kW asociado al piloto de validación tecnológica inicial.</div>
          </div>
          <div class="capex-detail-card {'active' if capex_80kw_active else ''}">
            <div class="capex-detail-k">ESCALAMIENTO 80 KW</div>
            <div class="capex-detail-t">{format_clp(capex_80kw_val)}</div>
            <div class="capex-detail-s">CAPEX 80kW total, incorporando estructura técnica y capital humano asociado.</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    capex10_col, capex80_col = st.columns(2)
    with capex10_col:
        st.button(
            "Seleccionado" if capex_10kw_active else "Seleccionar CAPEX 10kW",
            key="inputs_capex_focus_kpi_10kw",
            use_container_width=True,
            type="primary" if capex_10kw_active else "secondary",
            on_click=_set_capex_focus,
            args=("10kw",),
        )
    with capex80_col:
        st.button(
            "Seleccionado" if capex_80kw_active else "Seleccionar CAPEX 80kW",
            key="inputs_capex_focus_kpi_80kw",
            use_container_width=True,
            type="primary" if capex_80kw_active else "secondary",
            on_click=_set_capex_focus,
            args=("80kw",),
        )

    if capex_10kw_active:
        render_inputs_capex_10kw_detail()
    elif capex_80kw_active:
        input_dashboard_tab, input_capex_tab, input_direccion_tab, input_explorador_tab = st.tabs(
            ["📊 Dashboard general", "🏗️ Detalle capex 80kW", "🧑‍💼 Capital Humano", "🔍 Explorador interactivo"]
        )
        with input_dashboard_tab:
            render_resumen_content(
                key_prefix="inputs_dashboard_general_",
                include_export=False,
                include_direction_item=True,
            )
        with input_capex_tab:
            render_resumen_content(
                key_prefix="inputs_capex_overview_",
                include_export=False,
                include_direction_item=False,
            )
            render_capex_module_content(selector_key="inputs_capex_internal_selector")
        with input_direccion_tab:
            render_direccion_module_content()
        with input_explorador_tab:
            render_explorador_module_content(key_prefix="inputs_explorer_")
    else:
        st.info("Selecciona `CAPEX 10kW` o `CAPEX 80kW` para abrir el contenido asociado.")
elif selected_input_block == "valorizacion":
    render_valorizacion_module_content(key_prefix="inputs_val_")
else:
    st.info("Selecciona uno de los KPIs principales para abrir sus sub-bloques y contenido.")

# -------------------------
# TAB RESUMEN
# -------------------------
if False:
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
if False:
    render_pagos_hitos(capex_url, fx_used, pagos_scale, include_direction_salaries=False)
    
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
    
def render_capex_categoria_content():
    st.subheader("Análisis técnico por categoría")

    df_cat_filtrado = df_cat.copy()
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

                    total_item = df_item_cat["Monto_CLP"].sum()
                    total_capex_visible = df_capex_filtrado["Monto_CLP"].sum()
                    pct_item_total = (total_item / total_capex_visible) if total_capex_visible > 0 else 0

                    fig_donut_item = px.pie(df_item_cat, values="Monto_CLP", names="Categoria", hole=0.70)
                    fig_donut_item.update_traces(
                        textinfo="percent",
                        textposition="inside",
                        hovertemplate="<b>%{label}</b><br>Participación dentro del ítem: %{percent:.1%}<br>Monto CLP: %{value:,.0f}<br><extra></extra>",
                        insidetextorientation="horizontal"
                    )
                    fig_donut_item.add_annotation(
                        x=0.5, y=0.5, text=f"{pct_item_total*100:.1f}%", showarrow=False,
                        font=dict(size=22, color="black"), xanchor="center", yanchor="middle"
                    )
                    fig_donut_item.update_layout(
                        showlegend=True,
                        legend=dict(orientation="v", x=1.25, y=0.5, xanchor="left", font=dict(size=11)),
                        margin=dict(l=0, r=120, t=10, b=10),
                        height=280,
                    )
                    st.plotly_chart(fig_donut_item, use_container_width=True)
    else:
        st.info("No hay ítems para mostrar en los dónuts según las categorías seleccionadas.")

    st.markdown("### Participación porcentual por categoría")
    total_clp_cat = df_cat_filtrado["Monto_CLP"].sum()
    df_cat_plot = df_cat_filtrado.copy().sort_values("Monto_CLP", ascending=False)
    df_cat_plot["Pct_cat"] = df_cat_plot["Monto_CLP"] / total_clp_cat if total_clp_cat > 0 else 0.0

    df_cat_item = (
        df_capex.groupby(["Categoria", "Item"], as_index=False)
        .agg(Monto_CLP=("Monto_CLP", "sum"))
        .sort_values(["Categoria", "Monto_CLP"], ascending=[True, False])
    )
    top_item_by_cat = df_cat_item.drop_duplicates(subset=["Categoria"], keep="first")
    cat_item_color_map = {
        row["Categoria"]: item_color_map.get(row["Item"], CAT_COLOR_MAP.get(row["Categoria"], "#2563EB"))
        for _, row in top_item_by_cat.iterrows()
    }

    fig_cat = px.bar(
        df_cat_plot,
        x="Categoria",
        y="Pct_cat",
        color="Categoria",
        color_discrete_map=cat_item_color_map,
        text="Pct_cat",
        labels={"Categoria": "Categoría", "Pct_cat": "Participación"},
        title="Distribución porcentual del CAPEX por categoría",
    )
    fig_cat.update_traces(texttemplate="%{text:.1%}", textposition="outside")
    max_part = float(df_cat_plot["Pct_cat"].max() or 0)
    fig_cat.update_yaxes(tickformat=".0%", range=[0, max_part * 1.15 if max_part > 0 else 1])
    fig_cat.update_layout(
        xaxis_title="",
        yaxis_title="Participación (%)",
        margin=dict(l=10, r=10, t=80, b=120),
        height=460,
        bargap=0.25,
        showlegend=False,
    )
    st.plotly_chart(fig_cat, use_container_width=True)
    st.session_state["fig_cat_categoria"] = fig_cat

    legend_css = """
    <style>
    .item-legend { display:flex; flex-wrap:wrap; gap:0.45rem 0.75rem; margin-top:0.4rem; margin-bottom:0.8rem; }
    .item-legend-title { font-size:0.82rem; font-weight:700; color:#6B7280; text-transform:uppercase; letter-spacing:.08em; margin-top:0.4rem; margin-bottom:0.25rem; }
    .item-legend-chip { display:inline-flex; align-items:center; gap:0.4rem; font-size:0.78rem; font-weight:600; color:#111827; }
    .item-legend-swatch { width:12px; height:12px; border-radius:2px; border:1px solid rgba(17, 24, 39, 0.2); }
    </style>
    """
    legend_items = []
    legend_order = [
        "Desarrollo Tecnológico", "Componentes Mecánicos", "Sistema Eléctrico y Control",
        "Obras Civiles", "Montaje y Logística", "Ensayos y Certificación", "Contingencias y Administración",
    ]
    for item in legend_order:
        color = CAT_COLOR_MAP.get(item, "#2563EB")
        legend_items.append(
            f'<span class="item-legend-chip"><span class="item-legend-swatch" style="background:{color}"></span>{item}</span>'
        )
    st.markdown(
        legend_css + '<div class="item-legend-title">Ítem</div>' + f'<div class="item-legend">{"".join(legend_items)}</div>',
        unsafe_allow_html=True,
    )

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
        df_show[["Categoria", "Participación (%)", "Monto_CLP_fmt", "Monto_USD_fmt", "Bullet_cat"]],
        hide_index=True,
        use_container_width=True,
    )


def render_capex_items_content():
    st.subheader("Top ítems por monto")
    render_category_palette()

    top_n = st.slider("Número de ítems a mostrar (Top N):", 5, 30, 15, step=1, key="capex_items_top_n")
    df_top = df_capex.sort_values("Monto_CLP", ascending=False).head(top_n).copy()
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
        labels={"Monto_CLP_MM": "Monto (MM CLP)", "Item": "Ítem", "Categoria": "Categoría"},
        title=f"Top {top_n} ítems por monto (millones de CLP)",
    )
    fig_top.update_traces(text=df_top["Monto_CLP_MM"].apply(lambda v: f"{v:.1f} MM"), textposition="outside")
    fig_top.update_layout(xaxis_title="Monto (millones de CLP)", yaxis_title="", margin=dict(l=10, r=10, t=60, b=10))
    st.plotly_chart(fig_top, use_container_width=True)
    st.session_state["fig_top_items"] = fig_top

    st.markdown("#### Tabla detallada")
    st.dataframe(
        df_top[["Item", "Categoria", "Participación (%)", "Monto_CLP_fmt", "Monto_USD_fmt", "Bullet"]],
        hide_index=True,
        use_container_width=True,
    )


def render_capex_module_content(selector_key: str = "capex_internal_selector"):
    st.markdown(
        """
        <div style="
            display:flex;
            align-items:center;
            gap:12px;
            margin:6px 0 14px 0;
            padding:10px 14px;
            border:1px solid rgba(229,231,235,1);
            border-radius:16px;
            background:linear-gradient(180deg,#ffffff 0%,#f8fafc 100%);
            width:fit-content;
            box-shadow:0 6px 14px rgba(15,23,42,.06);
        ">
            <div style="
                width:42px;
                height:42px;
                border-radius:12px;
                display:flex;
                align-items:center;
                justify-content:center;
                background:#0f172a;
                color:white;
                font-size:22px;
                font-weight:700;
            ">C</div>
            <div>
                <div style="font-size:11px;font-weight:700;letter-spacing:.12em;color:#64748b;text-transform:uppercase;">
                    Modulo
                </div>
                <div style="font-size:22px;font-weight:800;color:#111827;line-height:1.1;">
                    Capex
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    capex_subview = st.radio(
        "Visualización dentro de Capex",
        ["📂 Por categoría", "📄 Detalle de ítems"],
        index=0,
        horizontal=True,
        key=selector_key,
    )
    if capex_subview == "📂 Por categoría":
        render_capex_categoria_content()
    else:
        render_capex_items_content()


def render_direccion_module_content():
    st.subheader("Fondos de Dirección / Director General Técnico")
    st.info(
        "Esta pestaña muestra fondos de estructura técnica y dirección que se leen desde la hoja "
        "`Director General Técnico`. Estos montos no se suman al CAPEX base de 480 MM CLP."
    )

    if direccion_error:
        st.error(direccion_error)
    elif df_direccion.empty:
        st.warning("La hoja `Director General Técnico` no tiene registros válidos para mostrar.")
    else:
        total_direccion = float(df_direccion["Total"].sum() or 0.0)
        total_meses = float(df_direccion["Meses"].sum() or 0.0)
        costo_mensual_prom = (
            float(df_direccion["Costo empresa mensual"].mean() or 0.0)
            if not df_direccion["Costo empresa mensual"].empty else 0.0
        )
        capex_mas_direccion = capex_total_clp + total_direccion

        dk1, dk2, dk3, dk4 = st.columns(4)
        with dk1:
            kpi_card(
                "Fondos dirección (CLP)",
                format_clp(total_direccion),
                "Monto total separado del CAPEX técnico base."
            )
        with dk2:
            kpi_card(
                "Cargos cubiertos",
                f"{len(df_direccion):,}".replace(",", "."),
                "Roles leídos desde la hoja Dirección General Técnico."
            )
        with dk3:
            kpi_card(
                "Meses acumulados",
                f"{total_meses:,.0f}".replace(",", "."),
                "Suma de meses reportados por cargo."
            )
        with dk4:
            kpi_card(
                "CAPEX + Dirección",
                format_clp(capex_mas_direccion),
                "Vista referencial si quisieras observar ambos bloques juntos."
            )

        df_direccion["Total_MM"] = df_direccion["Total"] / 1e6
        df_direccion["Costo_mensual_fmt"] = df_direccion["Costo empresa mensual"].apply(format_clp)
        df_direccion["Total_fmt"] = df_direccion["Total"].apply(format_clp)

        df_dir_plot = df_direccion.sort_values("Total", ascending=True).copy()
        direccion_color_map = {
            "Ingeniero Eléctrico": "#7C3AED",
            "Ingeniero Mecánico": "#0F766E",
            "Ingeniero Proyecto (PMO)": "#C2410C",
            "Director General Técnico": "#1D4ED8",
        }
        fig_direccion = px.bar(
            df_dir_plot,
            x="Total_MM",
            y="Cargo",
            orientation="h",
            text=df_dir_plot["Total_MM"].map(lambda v: f"{v:.1f} MM"),
            color="Cargo",
            color_discrete_map=direccion_color_map,
            title="Fondos por cargo de dirección técnica",
            labels={"Total_MM": "Monto total (MM CLP)", "Cargo": ""},
        )
        fig_direccion.update_traces(
            textposition="outside",
            marker=dict(line=dict(color="rgba(255,255,255,0.85)", width=1.2)),
            hovertemplate="<b>%{y}</b><br>Total: %{x:.1f} MM CLP<extra></extra>",
        )
        fig_direccion.update_layout(
            showlegend=False,
            margin=dict(l=10, r=32, t=70, b=24),
            height=430,
            plot_bgcolor="white",
            paper_bgcolor="rgba(0,0,0,0)",
            bargap=0.20,
            title=dict(
                text="Fondos por cargo de direccion tecnica",
                font=dict(size=22, color="#0f172a"),
                x=0.02,
            ),
            font=dict(color="#334155", size=13),
        )
        fig_direccion.update_xaxes(
            showgrid=True,
            gridcolor="rgba(148,163,184,0.25)",
            zeroline=False,
            ticksuffix=" MM",
        )
        fig_direccion.update_yaxes(showgrid=False)
        st.plotly_chart(fig_direccion, use_container_width=True)

        col_dir_1, col_dir_2 = st.columns([1.2, 1])
        with col_dir_1:
            st.markdown("#### Tabla base")
            st.dataframe(
                df_direccion[["Cargo", "Meses", "Costo_mensual_fmt", "Total_fmt"]].rename(columns={
                    "Costo_mensual_fmt": "Costo empresa mensual",
                    "Total_fmt": "Total",
                }),
                hide_index=True,
                use_container_width=True,
            )
        with col_dir_2:
            st.markdown("#### Lectura ejecutiva")
            st.markdown(
                f"- Fondos de dirección identificados: **{format_clp(total_direccion)}**.\n"
                f"- Costo empresa mensual promedio: **{format_clp(costo_mensual_prom)}**.\n"
                f"- Si se observa junto al CAPEX técnico, la referencia total sería **{format_clp(capex_mas_direccion)}**.\n"
                f"- Este bloque se mantiene deliberadamente separado para no contaminar el desglose del CAPEX de ingeniería."
            )


if False:
    render_top_summary_kpis()
    render_capex_module_content(selector_key="capex_internal_selector")

# -------------------------
# TAB EXPLORADOR
# -------------------------
if False:
    render_top_summary_kpis()
    render_explorador_module_content(key_prefix="explorer_")

# -------------------------
# TAB DIRECCIÓN TÉCNICA
# -------------------------
if False:
    render_top_summary_kpis()
    render_direccion_module_content()

# -------------------------
# TAB VALORIZACIÓN
# -------------------------
if False:
    render_top_summary_kpis()
    render_valorizacion_module_content(key_prefix="val_")

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


if False:
    st.markdown("---")
    st.subheader("📄 Exportar informe técnico")
    
    pdf_bytes = build_pdf_report()
    st.download_button(
        label="📥 Descargar reporte PDF técnico (CAPEX Piloto 80 kW)",
        data=pdf_bytes,
        file_name="Reporte_CAPEX_Piloto_80kW.pdf",
        mime="application/pdf",
    )

# PILOTOS_1050.py
# =============================================================================
# Dashboard Financiero — Pestaña «Pruebas» (desde URL CSV o CSV local)
# =============================================================================
# Requisitos:
#   pip install streamlit pandas plotly openpyxl
# Ejecución:
#   streamlit run PILOTOS_1050.py
# -----------------------------------------------------------------------------

import io
import re
import datetime as dt
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Dashboard Piloto 10kW", layout="wide")
st.title("📊 Dashboard Piloto 10kW")

PRIMARY = "#0E9F6E"
GRID    = "rgba(148,163,184,.25)"
PALETTE_SM = {
    "Suministro": "#0EA5A4",   # teal
    "I+D":        "#6366F1",   # indigo
    "Montaje":    "#F59E0B",   # amber
}


DEFAULT_CSV_URL = (
    "https://docs.google.com/spreadsheets/d/e/2PACX-1vQmVzOg9X7VfxAmOImXHuMvyH4dQmxbFL3DIBqOubi32jKLncgqBEBwnl6j0dXWsm5FkRAcrY4y8BD2/pub?gid=1596174884&single=true&output=csv"
)

# -----------------------------------------------------------------------------
# Utilidades
# -----------------------------------------------------------------------------
def _money_fmt(x):
    try:
        return f"${x:,.0f}".replace(",", ".")
    except Exception:
        return x

def _to_num_strict(val):
    """
    Convierte a float distintos formatos:
      - '65.306.085' (miles con puntos, sin decimales)
      - '$ 1.234.567,89' (EU)
      - '1,234,567.89'  (US)
      - '(12.345,00)'   (negativo con paréntesis)
    """
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if s == "" or s.lower() in {"nan", "none", "null", "-", "s/n"}:
        return np.nan

    neg = s.startswith("(") and s.endswith(")")
    if neg:
        s = s[1:-1]

    # quitar símbolos
    s = (s.replace("$", "").replace("CLP", "").replace("USD", "")
           .replace(" ", "").replace("\u00a0", ""))

    if "," in s and "." in s:
        # mezcla coma y punto -> decidir por el último símbolo
        if s.rfind(",") > s.rfind("."):
            # '1.234.567,89' => coma decimal
            s = s.replace(".", "").replace(",", ".")
        else:
            # '1,234,567.89' => punto decimal
            s = s.replace(",", "")
    elif "," in s:
        # solo comas: asume coma decimal y puntos miles
        s = s.replace(".", "").replace(",", ".")
    else:
        # solo puntos o nada
        if s.count(".") > 1:
            # '65.306.085' => todos son miles
            s = s.replace(".", "")
        elif "." in s:
            # un solo punto: decidir si miles (###) o decimal
            left, right = s.split(".")
            if len(right) == 3 and left.isdigit():
                s = left + right

    s = re.sub(r"[^0-9.\-]", "", s)

    try:
        num = float(s)
        return -num if neg else num
    except Exception:
        return np.nan

def _safe_rerun():
    """Rerun compatible con versiones nuevas/antiguas de Streamlit."""
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun()
        except Exception:
            pass

@st.cache_data(show_spinner=False)
def load_csv(source) -> pd.DataFrame:
    """Carga CSV desde URL o archivo subido y normaliza columnas clave."""
    if isinstance(source, str):
        df = pd.read_csv(source)
    else:
        df = pd.read_csv(source)

    df.columns = [c.strip() for c in df.columns]

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
    for a, b in rename_map.items():
        if a in df.columns and b not in df.columns:
            df.rename(columns={a: b}, inplace=True)

    expected_text = [
        "Etapa","Estado de pago","Provedor","item","Sub-item","Descripciónn",
        "Suministro / montaje","Material","Uni","Centro de costo","Observación",
        "Factor de costo","Estado de costo","Justificación % e E.E",
        "Tributa la HIBRIDA","Tributa la DARRIEUS","N° OC","Boleta / fac","Situación factura","Forma de pago"
    ]
    expected_dates = ["Fecha inicio","Fecha fin","Fecha entrega","Fin proyecto"]
    expected_nums  = ["Monto","Dif-1","Dif-2","diF-T","% DI-T","Dias de proyecto",
                      "Descuento ec escala","Precio final ec esc","ID-elemento"]

    for c in expected_text + expected_dates + expected_nums + ["ID-elemento"]:
        if c not in df.columns:
            df[c] = np.nan

    for c in expected_dates:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", infer_datetime_format=True)

    for c in expected_nums:
        if c in df.columns:
            df[c] = df[c].apply(_to_num_strict)

    return df

def build_filters():
    st.sidebar.header("⚙️ Fuente y Filtros")

    # Botón para limpiar caché y recargar datos desde Drive/URL
    if st.sidebar.button("🔁 Actualizar datos (Drive/URL)"):
        st.cache_data.clear()
        _safe_rerun()

    modo = st.sidebar.radio("Fuente de datos", ["URL CSV", "Subir CSV"], index=0)
    if modo == "URL CSV":
        url = st.sidebar.text_input("URL CSV (Google Sheets publicado)", value=DEFAULT_CSV_URL)
        source = url
    else:
        up = st.sidebar.file_uploader("Sube CSV", type=["csv"])
        source = up

    if not source:
        st.info("Indica una URL o sube un CSV para comenzar.")
        st.stop()

    df = load_csv(source)

    etapas  = sorted([e for e in df["Etapa"].dropna().unique()])
    estados = sorted([e for e in df["Estado de pago"].dropna().unique()])
    provs   = sorted([e for e in df["Provedor"].dropna().unique()])

    etapa_sel  = st.sidebar.multiselect("Etapa", etapas, default=etapas)
    estado_sel = st.sidebar.multiselect("Estado de pago", estados, default=estados if estados else [])
    prov_sel   = st.sidebar.multiselect("Proveedor", provs, default=provs if provs else [])

    fechas_candidatas = pd.concat([
        df["Fecha inicio"].dropna(),
        df["Fecha fin"].dropna(),
        df["Fecha entrega"].dropna()
    ], ignore_index=True) if any(col in df.columns for col in ["Fecha inicio","Fecha fin","Fecha entrega"]) else pd.Series([], dtype="datetime64[ns]")

    if len(fechas_candidatas):
        fmin = fechas_candidatas.min().date()
        fmax = fechas_candidatas.max().date()
    else:
        fmin = dt.date(dt.date.today().year, 1, 1)
        fmax = dt.date.today()

    rango = st.sidebar.date_input("Rango de fechas (inicio/fin/entrega)", (fmin, fmax))
    if isinstance(rango, tuple) and len(rango) == 2:
        d1, d2 = rango
    else:
        d1, d2 = fmin, fmax

    query_txt = st.sidebar.text_input("Buscar texto (Descripción, item, Sub-item)", value="").strip()

    return df, (etapa_sel, estado_sel, prov_sel, (d1, d2), query_txt)

def apply_filters(df, etapa_sel, estado_sel, prov_sel, date_range, query_txt):
    d1, d2 = date_range
    mask = pd.Series(True, index=df.index)

    if etapa_sel:
        mask &= df["Etapa"].isin(etapa_sel)
    if estado_sel and "Estado de pago" in df.columns:
        mask &= df["Estado de pago"].isin(estado_sel)
    if prov_sel and "Provedor" in df.columns:
        mask &= df["Provedor"].isin(prov_sel)

    def in_range(series):
        return series.between(pd.to_datetime(d1), pd.to_datetime(d2), inclusive="both")

    if any(col in df.columns for col in ["Fecha inicio","Fecha fin","Fecha entrega"]):
        fmask = pd.Series(False, index=df.index)
        for c in ["Fecha inicio","Fecha fin","Fecha entrega"]:
            if c in df.columns:
                fmask |= in_range(df[c].fillna(pd.NaT))
        mask &= fmask

    if query_txt:
        q = query_txt.lower()
        txt_mask = (
            df.get("Descripciónn", pd.Series("", index=df.index)).astype(str).str.lower().str.contains(q, na=False) |
            df.get("item", pd.Series("", index=df.index)).astype(str).str.lower().str.contains(q, na=False) |
            df.get("Sub-item", pd.Series("", index=df.index)).astype(str).str.lower().str.contains(q, na=False)
        )
        mask &= txt_mask

    return df[mask].copy()

# ============================
# KPIs principales (tarjetas PRO v2)
# ============================
def render_main_kpi_cards(df_in: pd.DataFrame, presupuesto_total: float | None = None):
    import html
    from textwrap import dedent

    # ---- Cálculos base
    m_series = df_in.get("Monto", pd.Series(dtype=float)).apply(_to_num_strict)
    monto_total = float(m_series.sum(skipna=True) or 0)
    monto_prom  = float(m_series.mean(skipna=True) or 0)
    n_items     = int(m_series.notna().sum())
    n_prov      = int(df_in.get("Provedor", pd.Series(dtype=object)).nunique(dropna=True))

    # ---- Delta (últimos 30 días vs 30 previos)
    date_col = next((c for c in ["Fecha entrega","Fecha fin","Fecha inicio"] if c in df_in.columns), None)
    delta_txt, delta_cls = "—", ""
    if date_col is not None:
        tmp = df_in[[date_col]].copy()
        tmp["Fecha"] = pd.to_datetime(df_in[date_col], errors="coerce")
        tmp["Monto"] = m_series
        tmp = tmp.dropna(subset=["Fecha","Monto"])
        if not tmp.empty:
            end  = tmp["Fecha"].max().normalize()
            p2_s, p2_e = end - pd.Timedelta(days=29), end
            p1_s, p1_e = p2_s - pd.Timedelta(days=30), p2_s - pd.Timedelta(days=1)
            v2 = float(tmp.loc[tmp["Fecha"].between(p2_s, p2_e), "Monto"].sum())
            v1 = float(tmp.loc[tmp["Fecha"].between(p1_s, p1_e), "Monto"].sum())
            if v1 == 0 and v2 == 0:
                delta_txt = "0,0% vs. período anterior"
            elif v1 == 0 and v2 > 0:
                delta_txt, delta_cls = "▲ +∞% vs. período anterior", "kpi-delta--up"
            else:
                d = (v2 - v1) / v1 * 100.0
                delta_txt = f"{'▲' if d>=0 else '▼'} {d:+.1f}% vs. período anterior"
                delta_cls = "kpi-delta--up" if d >= 0 else "kpi-delta--down"

    # ---- Progreso vs presupuesto (opcional)
    prog_pct = None
    if isinstance(presupuesto_total, (int, float)) and presupuesto_total > 0:
        prog_pct = max(0.0, min(100.0, (monto_total / presupuesto_total) * 100.0))

    # ---- Estilos y tarjetas
    st.markdown(dedent("""
    <style>
      .kpi-wrap{display:grid;grid-template-columns:repeat(4,minmax(240px,1fr));gap:16px;margin:6px 0 18px}
      @media (max-width:1200px){.kpi-wrap{grid-template-columns:repeat(2,minmax(240px,1fr))}}
      @media (max-width:700px){.kpi-wrap{grid-template-columns:1fr}}

      .kpi-card{
        border-radius:16px;padding:16px 18px;
        background:linear-gradient(180deg,#f8fafc 0%,#ffffff 62%);
        border:1px solid rgba(148,163,184,.35); box-shadow:0 6px 14px rgba(15,23,42,.06)
      }
      .kpi-h{font-size:13px;font-weight:700;color:#0f172a;letter-spacing:.2px;margin:0 0 6px}
      .kpi-v{font-size:28px;font-weight:800;color:#0f172a;margin:2px 0 8px}
      .kpi-sub{display:flex;gap:8px;flex-wrap:wrap}
      .chip{display:inline-block;font-size:12px;padding:2px 8px;border-radius:999px;border:1px solid rgba(148,163,184,.35);background:#f1f5f9;color:#0f172a}
      .chip--accent{background:#e6fff6;color:#047857;border-color:#34d399}
      .chip--indigo{background:#eef2ff;color:#3730a3;border-color:#a5b4fc}
      .kpi-delta{font-size:12px;margin-top:8px;color:#475569}
      .kpi-delta--up{color:#047857}
      .kpi-delta--down{color:#b91c1c}
      .kpi-row{display:flex;align-items:center;gap:8px}
      .kpi-ico{width:28px;height:28px;border-radius:999px;display:inline-flex;align-items:center;justify-content:center;background:#0ea5a41a;border:1px solid #0ea5a455}
      .kpi-bar{position:relative;width:100%;height:8px;border-radius:999px;background:#eef2f7;margin-top:8px}
      .kpi-bar>span{position:absolute;left:0;top:0;height:100%;border-radius:999px;background:#0EA5A4}
    </style>
    """), unsafe_allow_html=True)

    def money(v): return _money_fmt(v)
    prog_html = f"""<div class="kpi-bar"><span style="width:{prog_pct:.6f}%"></span></div>
                    <div class="kpi-delta" style="margin-top:4px">Avance: {prog_pct:.1f}% del presupuesto</div>""" if prog_pct is not None else ""

    cards = f"""
    <div class="kpi-wrap">

      <div class="kpi-card">
        <div class="kpi-row"><div class="kpi-ico">💰</div><div class="kpi-h">Monto Total (CLP)</div></div>
        <div class="kpi-v">{money(monto_total)}</div>
        <div class="kpi-sub">
          <span class="chip chip--accent">Base: {n_items:,} ítems</span>
          <span class="chip">Proveedores: {n_prov:,}</span>
        </div>
        <div class="kpi-delta {delta_cls}">{html.escape(delta_txt)}</div>
        {prog_html}
      </div>

      <div class="kpi-card">
        <div class="kpi-row"><div class="kpi-ico">🧮</div><div class="kpi-h">Monto Promedio / ítem</div></div>
        <div class="kpi-v">{money(monto_prom)}</div>
        <div class="kpi-sub"><span class="chip">Distribución por ítem</span></div>
      </div>

      <div class="kpi-card">
        <div class="kpi-row"><div class="kpi-ico">📦</div><div class="kpi-h">Ítems con Monto</div></div>
        <div class="kpi-v">{str(n_items).replace(',', '.')}</div>
        <div class="kpi-sub"><span class="chip chip--indigo">Con valor asignado</span></div>
      </div>

      <div class="kpi-card">
        <div class="kpi-row"><div class="kpi-ico">🏭</div><div class="kpi-h">Proveedores únicos</div></div>
        <div class="kpi-v">{str(n_prov).replace(',', '.')}</div>
        <div class="kpi-sub"><span class="chip">Diversidad de oferta</span></div>
      </div>

    </div>
    """
    st.markdown(cards, unsafe_allow_html=True)


def charts_block(df):
    st.markdown("### 📈 Gráficos")
    base = df.copy()
    base = base[np.isfinite(base["Monto"].fillna(0))]  # evita NaN/inf

    # --- Monto por Etapa (barras)
        # --- Monto por Etapa (barras)  🔥 versión mejorada
    if "Etapa" in base.columns and base["Etapa"].notna().any():
        orden_etapas = ["1-Investigación", "2-Desarrollo", "3-Piloto"]

        g_etapa = (base.groupby("Etapa", as_index=False)["Monto"].sum())
        g_etapa["Etapa_orden"] = g_etapa["Etapa"].apply(lambda x: orden_etapas.index(x) if x in orden_etapas else 999)
        g_etapa = g_etapa.sort_values(["Etapa_orden","Monto"], ascending=[True, False]).drop(columns="Etapa_orden")

        total_etapas = float(g_etapa["Monto"].sum() or 0)
        g_etapa["%"] = (g_etapa["Monto"] / total_etapas * 100.0).round(2) if total_etapas>0 else 0.0
        g_etapa["label"] = g_etapa.apply(lambda r: f"{_money_fmt(r['Monto'])}  ·  {r['%']:.2f}%", axis=1)

        st.markdown("#### 📊 Distribución por Etapa")

        # Toggle: CLP vs %
        ver_pct = st.toggle("Ver % del total (100%)", value=False, key="etapa_pct_toggle")

        if not ver_pct:
            # Monto CLP
            fig1 = px.bar(
                g_etapa, x="Monto", y="Etapa", orientation="h",
                color="Etapa",
                color_discrete_map={
                    "1-Investigación": "#60A5FA",  # azul claro
                    "2-Desarrollo":    "#34D399",  # green
                    "3-Piloto":        "#0EA5A4",  # teal
                },
                text="label",
                title="Distribución de Monto por Etapa",
            )
            fig1.update_traces(
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>Monto: $%{x:,.0f}<br>Participación: %{customdata:.2f}%<extra></extra>",
                customdata=g_etapa["%"],
            )
            fig1.update_xaxes(
                tickprefix="$", separatethousands=True,
                title="Monto (CLP)"
            )
        else:
            # 100% stacked style (pero por etapa individual)
            g_pct = g_etapa.copy()
            g_pct["%"] = g_pct["%"].round(2)
            fig1 = px.bar(
                g_pct, x="%", y="Etapa", orientation="h",
                color="Etapa",
                color_discrete_map={
                    "1-Investigación": "#60A5FA",
                    "2-Desarrollo":    "#34D399",
                    "3-Piloto":        "#0EA5A4",
                },
                text=g_pct["%"].map(lambda v: f"{v:.2f}%"),
                title="Distribución por Etapa — % del total",
            )
            fig1.update_traces(
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>Participación: %{x:.2f}%<extra></extra>",
            )
            fig1.update_xaxes(range=[0,100], title="% del total")

        # Estética general
        fig1.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="rgba(0,0,0,0)",
            legend_title="Etapa",
            margin=dict(l=90, r=30, t=60, b=40),
        )
        fig1.update_yaxes(title="Categoría", showgrid=False)
        fig1.add_annotation(
            xref="paper", yref="paper", x=0, y=1.12, showarrow=False,
            text=f"<span style='font-size:13px;color:#475569'>Total: {_money_fmt(total_etapas)}</span>"
        )

        st.plotly_chart(fig1, use_container_width=True)


    # --- (ELIMINADO) Estado de Pago por Monto (dona)

    # --- Top 10 Proveedores por Monto (barras)
    if "Provedor" in base.columns and base["Provedor"].notna().any():
        g_prov = (base.groupby("Provedor", as_index=False)["Monto"]
                        .sum().sort_values("Monto", ascending=False).head(10))
        if len(g_prov):
            fig3 = px.bar(g_prov, x="Monto", y="Provedor", orientation="h",
                          color="Monto", color_continuous_scale="Blues",
                          title="Proveedores por Monto")
            fig3.update_layout(yaxis={"categoryorder": "total ascending"})
            fig3.update_xaxes(separatethousands=True)
            st.plotly_chart(fig3, use_container_width=True)

    # --- Series temporales (si hay Fecha entrega)
    if "Fecha entrega" in base.columns and base["Fecha entrega"].notna().any():
        tmp = base.copy()
        tmp["Mes"] = tmp["Fecha entrega"].dt.to_period("M").dt.to_timestamp()
        g_time = (tmp.groupby("Mes", as_index=False)
                     .agg(Monto=("Monto","sum"), Items=("Mes","count"))
                     .sort_values("Mes"))
        col1, col2 = st.columns(2)
        if len(g_time):
            with col1:
                fig4 = px.line(g_time, x="Mes", y="Monto", markers=True,
                               title="Monto Entregado por Mes")
                fig4.update_yaxes(showgrid=True, gridcolor=GRID, separatethousands=True)
                st.plotly_chart(fig4, use_container_width=True)
            with col2:
                fig5 = px.bar(g_time, x="Mes", y="Items", color="Items",
                              color_continuous_scale="Sunset",
                              title="Ítems Entregados por Mes")
                st.plotly_chart(fig5, use_container_width=True)

def table_block(df):
    st.markdown("### 🧾 Tabla de datos filtrados")
    preferred_cols = [
        "ID-elemento","Etapa","Suministro / montaje","item","Sub-item",
        "Descripciónn","Provedor","Material","Uni","Monto",
        "Estado de pago","Forma de pago","N° OC","Boleta / fac","Situación factura",
        "Centro de costo","Fecha inicio","Fecha fin","Fecha entrega",
        "Dif-1","Dif-2","diF-T","% DI-T","Dias de proyecto","Fin proyecto",
        "Observación","Factor de costo","Estado de costo","Justificación % e E.E",
        "Descuento ec escala","Precio final ec esc","Tributa la HIBRIDA","Tributa la DARRIEUS"
    ]
    cols = [c for c in preferred_cols if c in df.columns] + [c for c in df.columns if c not in preferred_cols]
    view = df[cols].copy()

    if "Monto" in view.columns:
        view["Monto (formato)"] = view["Monto"].apply(_money_fmt)
        cols_new = view.columns.tolist()
        if "Monto (formato)" in cols_new and "Monto" in cols_new:
            cols_new.insert(cols_new.index("Monto")+1, cols_new.pop(cols_new.index("Monto (formato)")))
            view = view[cols_new]

    st.dataframe(view, use_container_width=True, height=420)

    st.markdown("#### ⬇️ Exportar")
    csv_bytes = view.to_csv(index=False).encode("utf-8-sig")
    st.download_button("Descargar CSV", data=csv_bytes, file_name="pruebas_filtrado.csv", mime="text/csv")

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        view.to_excel(writer, index=False, sheet_name="Pruebas_filtrado")
    st.download_button(
        "Descargar Excel",
        data=buffer.getvalue(),
        file_name="pruebas_filtrado.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# -----------------------------------------------------------------------------
# Helpers S/M (definiciones ANTES del uso)
# -----------------------------------------------------------------------------
def _to_num_strict_inline(val):
    import re, numpy as np, pandas as pd
    if pd.isna(val): return np.nan
    s = str(val).strip()
    if s == "" or s.lower() in {"nan","none","null","-","s/n"}: return np.nan

    neg = s.startswith("(") and s.endswith(")")
    if neg: s = s[1:-1]

    s = (s.replace("$","").replace("CLP","").replace("USD","")
           .replace(" ","").replace("\u00a0",""))

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

    s = re.sub(r"[^0-9.\-]", "", s)
    try:
        num = float(s)
        return -num if neg else num
    except Exception:
        return np.nan

def make_suministro_chart(df_in):
    import numpy as np
    import pandas as pd
    import plotly.express as px

    df = df_in.copy()
    if "Suministro / montaje" not in df.columns: df["Suministro / montaje"] = np.nan
    if "Monto" not in df.columns: df["Monto"] = np.nan

    df["Categoria"] = df["Suministro / montaje"].astype(str).str.strip()
    try:
        df["Monto_num"] = df["Monto"].apply(_to_num_strict)   # parser global
        if df["Monto_num"].isna().all(): raise NameError
    except Exception:
        df["Monto_num"] = df["Monto"].apply(_to_num_strict_inline)

    base = df[~df["Categoria"].isin(["nan","None",""]) & df["Monto_num"].notna()].copy()
    agg = (base.groupby("Categoria", as_index=False)
                .agg(Monto=("Monto_num","sum"), Items=("Monto_num","count")))
    if agg.empty: 
        return None, None

    total = agg["Monto"].sum()
    agg["% del total"] = (agg["Monto"]/total*100).round(2)

    # Orden sugerido y formato
    orden = ["Suministro","I+D","Montaje"]
    agg["__ord"] = agg["Categoria"].apply(lambda x: orden.index(x) if x in orden else 999)
    agg = agg.sort_values(["__ord","Monto"], ascending=[True,False]).drop(columns="__ord")
    agg["label_pct"] = agg["% del total"].map(lambda v: f"{v:.2f}%")

    # 🔹 Colores DISCRETOS fijos (mismos que las tarjetas KPI)
    fig = px.bar(
        agg, x="Monto", y="Categoria", orientation="h",
        color="Categoria", color_discrete_map=PALETTE_SM,
        text="label_pct",
        title="Suministro / Montaje — Monto y % del total"
    )
    fig.update_traces(
        textposition="outside",
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Monto: $%{x:,.0f}<br>"
            "% del total: %{customdata[0]:.2f}%<br>"
            "Nº ítems: %{customdata[1]}<extra></extra>"
        ),
        customdata=np.stack([agg["% del total"], agg["Items"]], axis=-1),
    )
    fig.update_layout(
        xaxis_title="Monto (CLP)", yaxis_title="Categoría",
        legend_title="S/M",
        margin=dict(l=80, r=40, t=60, b=60),
    )
    fig.update_xaxes(separatethousands=True)

    tabla = agg[["Categoria","Monto","% del total","Items"]].copy()
    return fig, tabla

def render_sm_kpi_cards(tabla_sm):
    """Tarjetas KPI — Suministro/Montaje a partir de tabla_sm."""
    import streamlit as st
    from textwrap import dedent
    import html

    if tabla_sm is None or tabla_sm.empty:
        st.info("No hay datos para mostrar KPIs.")
        return

    df = tabla_sm.copy().sort_values("Monto", ascending=False)

    def color_for(cat):
        palette = {"Suministro": "#0EA5A4", "I+D": "#6366F1", "Montaje": "#F59E0B"}
        return palette.get(cat, "#0E9F6E")

    st.markdown(dedent("""
    <style>
      .smkpi-grid{
        display: grid;
        grid-template-columns: repeat(3, minmax(260px,1fr));
        gap: 16px; align-items: stretch;
      }
      @media (max-width: 1200px){ .smkpi-grid{ grid-template-columns: repeat(2, minmax(260px,1fr)); } }
      @media (max-width: 700px){  .smkpi-grid{ grid-template-columns: 1fr; } }

      .smkpi-card{
        box-sizing: border-box; border-radius: 16px; padding: 14px 16px;
        background: linear-gradient(180deg,#f8fafc 0%, #ffffff 62%);
        border: 1px solid rgba(148,163,184,.35);
        box-shadow: 0 4px 10px rgba(15,23,42,.05);
      }
      .smkpi-title{font-size:14px; font-weight:700; color:#0f172a; letter-spacing:.2px; margin-bottom:6px;}
      .smkpi-value{font-size:28px; font-weight:800; color:#0f172a; margin:2px 0 6px 0;}
      .smkpi-row{font-size:13px; color:#475569; display:flex; gap:8px; flex-wrap:wrap;}
      .smkpi-chip{
        display:inline-block; padding:2px 8px; border-radius:999px;
        font-size:12px; border:1px solid rgba(148,163,184,.35);
        background:#f1f5f9; color:#0f172a;
      }
      .smkpi-bar{ position: relative; width:100%; height:8px; border-radius:999px; background:#eef2f7; margin-top:10px; }
      .smkpi-bar > span{ position:absolute; left:0; top:0; height:100%; border-radius:999px; }
    </style>
    """), unsafe_allow_html=True)

    def fmt_money(v):
        try:
            return _money_fmt(v)
        except Exception:
            return f"${v:,.0f}"

    cards = ['<div class="smkpi-grid">']
    for _, r in df.iterrows():
        cat   = str(r["Categoria"])
        monto = float(r["Monto"])
        pct   = float(r["% del total"])
        items = int(r["Items"])
        accent = color_for(cat)

        cards.append(dedent(f"""
        <div class="smkpi-card">
          <div class="smkpi-title">{html.escape(cat)}</div>
          <div class="smkpi-value">{fmt_money(monto)}</div>
          <div class="smkpi-row">
            <span class="smkpi-chip" style="background:{accent}1a; color:{accent}; border-color:{accent}55;">
              % del total: {pct:.2f}%
            </span>
            <span class="smkpi-chip">Ítems: {items:,}</span>
          </div>
          <div class="smkpi-bar"><span style="width:{pct:.6f}%; background:{accent};"></span></div>
        </div>
        """).strip())

    cards.append("</div>")
    st.markdown("".join(cards), unsafe_allow_html=True)

# ============================
# Flujo principal
# ============================
df_raw, filters = build_filters()
df = df_raw.copy()

# Reasegurar numérico por si cambió la fuente
if "Monto" in df.columns:
    df["Monto"] = df["Monto"].apply(_to_num_strict)

etapa_sel, estado_sel, prov_sel, rango_fechas, query_txt = filters
df_f = apply_filters(df, etapa_sel, estado_sel, prov_sel, rango_fechas, query_txt)
base = df_f if len(df_f) else df

render_main_kpi_cards(base, presupuesto_total=None)
charts_block(base)

# 👉 Bloque Suministro/Montaje (antes de la tabla)
st.markdown("### 📊 I+D - Suministros - Montaje  ")
fig_sm, tabla_sm = make_suministro_chart(base)
if fig_sm is None or tabla_sm is None or tabla_sm.empty:
    st.info("No hay datos válidos para graficar Suministro / Montaje.")
else:
    render_sm_kpi_cards(tabla_sm)                 # KPI cards estilizadas
    st.plotly_chart(fig_sm, use_container_width=True)

# Tabla general (OCULTA)
# table_block(base)

# Notas (OCULTAS)
# with st.expander("ℹ️ Notas"):
#     st.markdown(""" ... """)

# ===============================
# Bloque: Analítica por ITEM (con interacción Suministro/Montaje)
# ===============================
def _to_num_strict_fallback(val):
    import re, numpy as np, pandas as pd
    if pd.isna(val): return np.nan
    s = str(val).strip()
    if s == "" or s.lower() in {"nan","none","null","-","s/n"}: return np.nan
    neg = s.startswith("(") and s.endswith(")")
    if neg: s = s[1:-1]
    s = (s.replace("$","").replace("CLP","").replace("USD","")
           .replace(" ","").replace("\u00a0",""))
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".") if s.rfind(",") > s.rfind(".") else s.replace(",", "")
    elif "," in s:
        s = s.replace(".", "").replace(",", ".")
    else:
        if s.count(".") > 1:
            s = s.replace(".", "")
        elif "." in s:
            left, right = s.split(".")
            if len(right) == 3 and left.isdigit():
                s = left + right
    s = re.sub(r"[^0-9.\-]", "", s)
    try:
        num = float(s)
        return -num if neg else num
    except Exception:
        return np.nan

def render_item_kpi_cards(tabla_show, item_col):
    import streamlit as st
    from textwrap import dedent
    import html

    dfk = (tabla_show[[item_col, "Monto", "% del total", "Items", "Promedio"]]
           .sort_values("Monto", ascending=False)
           .copy())

    st.markdown("""
    <style>
      .kpi-grid{
        display: grid;
        grid-template-columns: repeat(3, minmax(260px, 1fr));
        gap: 16px;
        align-items: stretch;
      }
      @media (max-width: 1200px){ .kpi-grid{ grid-template-columns: repeat(2, minmax(260px,1fr)); } }
      @media (max-width: 700px){  .kpi-grid{ grid-template-columns: 1fr; } }

      .kpi-card{
        box-sizing: border-box;
        border-radius: 16px;
        padding: 14px 16px;
        background: linear-gradient(180deg,#f8fafc 0%, #ffffff 62%);
        border: 1px solid rgba(148,163,184,.35);
        box-shadow: 0 4px 10px rgba(15,23,42,.05);
      }
      .kpi-card * { text-decoration: none !important; }
      .kpi-title{font-size:14px; font-weight:700; color:#0f172a; letter-spacing:.2px; margin-bottom:2px;}
      .kpi-value{font-size:26px; font-weight:800; color:#0f172a; margin-top:4px;}
      .kpi-sub{font-size:13px; color:#475569; margin-top:8px;}
      .kpi-chip{
        display:inline-block; padding:2px 8px; border-radius:999px;
        font-size:12px; margin-right:6px; border:1px solid rgba(148,163,184,.35);
        background:#f1f5f9; color:#0f172a;
      }
      .kpi-chip--accent{ background:#e6fff6; color:#047857; border-color:#34d399; }
      .kpi-chip--info{   background:#eef2ff; color:#3730a3; border-color:#a5b4fc; }
    </style>
    """, unsafe_allow_html=True)

    def fmt_money(v):
        try:
            return _money_fmt(v)
        except Exception:
            return f"${v:,.0f}"

    cards_html = ['<div class="kpi-grid">']
    for _, rec in dfk.iterrows():
        name  = html.escape(str(rec[item_col]))
        monto = fmt_money(rec["Monto"])
        pct   = f"{rec['% del total']:.2f}%"
        items = f"{int(rec['Items']):,}".replace(",", ".")
        prom  = fmt_money(rec["Promedio"])

        card = dedent(f"""
        <div class="kpi-card">
          <div class="kpi-title">{name}</div>
          <div class="kpi-value">{monto}</div>
          <div class="kpi-sub">
            <span class="kpi-chip kpi-chip--accent">% del total: {pct}</span>
            <span class="kpi-chip">Ítems: {items}</span>
            <span class="kpi-chip kpi-chip--info">Prom: {prom}</span>
          </div>
        </div>
        """).strip()
        cards_html.append(card)

    cards_html.append("</div>")
    st.markdown("".join(cards_html), unsafe_allow_html=True)
def _render_cat_summary_pills(df2, cat_col: str):
    import html
    from textwrap import dedent

    pal = {"Suministro":"#0EA5A4", "I+D":"#6366F1", "Montaje":"#F59E0B"}

    agg = (df2.groupby(cat_col, as_index=False)
              .agg(Monto=("Monto_num","sum"), Items=("Monto_num","count")))
    if agg.empty:
        return

    total = float(agg["Monto"].sum() or 0)
    agg["pct"] = (agg["Monto"] / total * 100.0).round(2) if total>0 else 0.0

    # orden sugerido
    orden = ["Suministro", "I+D", "Montaje"]
    agg["__o"] = agg[cat_col].apply(lambda x: orden.index(x) if x in orden else 999)
    agg = agg.sort_values(["__o","Monto"], ascending=[True, False]).drop(columns="__o")

    css = """
    <style>
      .pill-wrap{display:flex;gap:8px;flex-wrap:wrap;margin:6px 0 8px}
      .pill{
        display:inline-flex;align-items:center;gap:8px;padding:6px 10px;border-radius:999px;
        border:1px solid rgba(148,163,184,.35); background:#fff;
        box-shadow:0 1px 2px rgba(15,23,42,.05); font-size:12px; color:#0f172a
      }
      .pill-dot{width:10px;height:10px;border-radius:999px}
      .pill-sub{opacity:.75}
    </style>
    """
    html_items = []
    for _, r in agg.iterrows():
        c = str(r[cat_col]); monto = float(r["Monto"]); pct = float(r["pct"]); n=int(r["Items"])
        color = pal.get(c, "#334155")
        html_items.append(
            f'''<div class="pill">
                   <span class="pill-dot" style="background:{color}"></span>
                   <strong>{html.escape(c)}</strong>
                   <span class="pill-sub">— {_money_fmt(monto)} ({pct:.2f}%) · {n} ítems</span>
                </div>'''
        )

    st.markdown(css + f'<div class="pill-wrap">{"".join(html_items)}</div>', unsafe_allow_html=True)


def render_item_analytics(df_in):
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import streamlit as st

    # --- Resolver nombres de columnas
    cols_norm = {c.strip().lower(): c for c in df_in.columns}
    item_col  = cols_norm.get("item", "item")
    cat_col   = (cols_norm.get("suministro / montaje")
                 or cols_norm.get("suministro/montaje")
                 or "Suministro / montaje")
    monto_col = cols_norm.get("monto", "Monto")

    # --- Base y conversión de montos
    df = df_in.copy()
    try:
        df["Monto_num"] = df[monto_col].apply(_to_num_strict)
        if df["Monto_num"].isna().all():
            raise NameError
    except Exception:
        df["Monto_num"] = df[monto_col].apply(_to_num_strict_fallback)

    df = df[df["Monto_num"].notna()].copy()
    df[item_col] = (df[item_col].astype(str).str.strip()
                    .replace({"nan": np.nan, "None": np.nan, "": np.nan})
                    .fillna("(Vacío)"))
    if cat_col not in df.columns:
        df[cat_col] = "Total"
    else:
        df[cat_col] = (df[cat_col].astype(str).str.strip()
                       .replace({"nan": np.nan, "None": np.nan, "": np.nan})
                       .fillna("(Sin categoría)"))

       # --- Controles UI (mejorados)
    st.markdown("### 🧩 Análisis por **Item**")
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        top_n = st.selectbox("Top N por monto", [5,10,15,20,30], index=2)
    with c2:
        modo = st.radio("Visualizar", ["Monto CLP", "% por Item (100%)"], index=0, horizontal=True)
    with c3:
        st.write("")  # espaciador

    # Presets de categorías
    cats_all = sorted(df[cat_col].dropna().unique())
    p1, p2, p3, p4 = st.columns([1,1,1,1])
    with p1:
        preset_all = st.button("🟢 Todas")
    with p2:
        preset_sum = st.button("⚙️ Solo Suministro")
    with p3:
        preset_mon = st.button("🛠️ Solo Montaje")
    with p4:
        preset_id  = st.button("🧪 Solo I+D")

    # Estado persistente para selección
    if "cats_sel_state" not in st.session_state:
        st.session_state.cats_sel_state = cats_all

    if preset_all:
        st.session_state.cats_sel_state = cats_all
    if preset_sum:
        st.session_state.cats_sel_state = [c for c in cats_all if str(c).lower().startswith("suministro")]
    if preset_mon:
        st.session_state.cats_sel_state = [c for c in cats_all if "montaje" in str(c).lower()]
    if preset_id:
        st.session_state.cats_sel_state = [c for c in cats_all if "i+d" in str(c).lower() or "i + d" in str(c).lower()]

    cats_sel = st.multiselect("Categorías S/M", cats_all, default=st.session_state.cats_sel_state, key="cats_sm_sel")
    st.session_state.cats_sel_state = cats_sel if cats_sel else st.session_state.cats_sel_state

    # Resumen de categorías (chips)
    _render_cat_summary_pills(df, cat_col)


    # --- Trabajar SIEMPRE con df2 (filtrado por categorías)
    df2 = df[df[cat_col].isin(cats_sel)].copy()
    if df2.empty:
        st.info("No hay datos para las categorías seleccionadas.")
        return

    # --- Resumen por ITEM (sobre df2)
    resumen_item = (df2.groupby(item_col, as_index=False)
                      .agg(Monto=("Monto_num","sum"),
                           Items=("Monto_num","count"),
                           Promedio=("Monto_num","mean")))
    resumen_item["% del total"] = (resumen_item["Monto"] /
                                   resumen_item["Monto"].sum() * 100).round(2)
    resumen_item["Promedio"] = resumen_item["Promedio"].round(0)
    resumen_item = resumen_item.sort_values("Monto", ascending=False)

    # --- Top-N y búsqueda
    top_items = resumen_item.head(top_n)[item_col].tolist()
    tabla_show = resumen_item[resumen_item[item_col].isin(top_items)].sort_values("Monto", ascending=False)
    

    # --- Tarjetas KPI (ya filtradas)
    render_item_kpi_cards(tabla_show, item_col)

    # --- Gráfico apilado Item x S/M usando SOLO df2 y los items mostrados
    items_keep = tabla_show[item_col].unique().tolist()
    pivot = (df2[df2[item_col].isin(items_keep)]
             .pivot_table(index=item_col, columns=cat_col, values="Monto_num",
                          aggfunc="sum", fill_value=0.0))

    if not pivot.empty:
        if modo == "Monto CLP":
            plot_df = pivot.reset_index().melt(id_vars=item_col, var_name="Categoría", value_name="Monto")
            fig = px.bar(
                plot_df,
                x="Monto", y=item_col, color="Categoría", orientation="h",
                color_discrete_map=PALETTE_SM,   # ← misma paleta que los KPI
                title="Top Items — Monto por categoría S/M"
            )
            fig.update_traces(hovertemplate="<b>%{y}</b><br>%{trace.name}: $%{x:,.0f}<extra></extra>")
            fig.update_xaxes(separatethousands=True)
        else:
            row_sums = pivot.sum(axis=1).replace(0, np.nan)
            pct = pivot.div(row_sums, axis=0) * 100
            plot_df = pct.reset_index().melt(id_vars=item_col, var_name="Categoría", value_name="%")
            fig = px.bar(
                plot_df,
                x="%", y=item_col, color="Categoría", orientation="h",
                color_discrete_map=PALETTE_SM,   # ← misma paleta que los KPI
                title="Top Items — % por categoría S/M (100%)"
            )
            fig.update_traces(hovertemplate="<b>%{y}</b><br>%{trace.name}: %{x:.2f}%<extra></extra>")
            fig.update_xaxes(range=[0,100])

        fig.update_layout(margin=dict(l=80, r=40, t=60, b=40), legend_title="S/M")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No hay datos para los filtros seleccionados.")
# ======= Ejecuta el bloque de Análisis por Item =======
render_item_analytics(base)

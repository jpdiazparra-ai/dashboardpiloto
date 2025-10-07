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
    st.sidebar.header("⚙️ Fuente de datos")

    # Campo para ingresar o pegar la URL de tu Google Sheet (publicado como CSV)
    src_url = st.sidebar.text_input(
        "URL CSV (Google Sheets publicado)",
        DEFAULT_CSV_URL,
        help="Pega aquí la URL publicada como CSV desde Google Sheets."
    )

    # Botón para refrescar caché y recargar datos
    if st.sidebar.button("🔁 Actualizar datos (Drive/URL)"):
        st.cache_data.clear()
        _safe_rerun()

    # Carga los datos directamente desde la URL
    df = load_csv(src_url)

    # No se aplican filtros (solo retorna el DataFrame)
    dmin, dmax = dt.date(2020, 1, 1), dt.date(2026, 12, 31)
    etapa_sel, estado_sel, prov_sel, query_txt = [], [], [], ""

    return df, (etapa_sel, estado_sel, prov_sel, (dmin, dmax), query_txt)




def apply_filters(df, etapa_sel, estado_sel, prov_sel, date_range, query_txt):
    d1, d2 = date_range
    d1 = pd.to_datetime(d1)
    d2 = pd.to_datetime(d2) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)  # incluir fin del día

    mask = pd.Series(True, index=df.index)

    # Etapa
    if etapa_sel:
        mask &= df.get("Etapa", pd.Series(index=df.index)).isin(etapa_sel)

    # Estado de pago
    if estado_sel and "Estado de pago" in df.columns:
        mask &= df["Estado de pago"].isin(estado_sel)

    # Proveedor (soporta Provedor/Proveedor)
    prov_col = "Provedor" if "Provedor" in df.columns else ("Proveedor" if "Proveedor" in df.columns else None)
    if prov_col and prov_sel:
        mask &= df[prov_col].isin(prov_sel)

    # ✅ FECHAS — NO excluir filas sin fecha:
    # Si tiene alguna fecha dentro del rango → se queda.
    # Si no tiene fecha → también se queda.
    date_cols = [c for c in ["Fecha inicio","Fecha fin","Fecha entrega"] if c in df.columns]
    if date_cols:
        fmask = pd.Series(True, index=df.index)
        for c in date_cols:
            col = pd.to_datetime(df[c], errors="coerce")
            inrng = col.between(d1, d2, inclusive="both")
            # Mantener filas sin fecha (col.isna()) o en rango
            fmask &= (~col.notna()) | inrng
        mask &= fmask

    # Texto
    if query_txt:
        q = str(query_txt).lower()
        txt_mask = (
            df.get("Descripciónn", pd.Series("", index=df.index)).astype(str).str.lower().str.contains(q, na=False) |
            df.get("item",         pd.Series("", index=df.index)).astype(str).str.lower().str.contains(q, na=False) |
            df.get("Sub-item",     pd.Series("", index=df.index)).astype(str).str.lower().str.contains(q, na=False)
        )
        mask &= txt_mask

    return df[mask].copy()


    # Soporte “Provedor” o “Proveedor”
    prov_col = "Provedor" if "Provedor" in df.columns else ("Proveedor" if "Proveedor" in df.columns else None)
    if prov_col and prov_sel:
        mask &= df[prov_col].isin(prov_sel)

    # Cualquier fecha cae dentro del rango
    if any(col in df.columns for col in ["Fecha inicio","Fecha fin","Fecha entrega"]):
        fmask = pd.Series(False, index=df.index)
        for c in ["Fecha inicio","Fecha fin","Fecha entrega"]:
            if c in df.columns:
                fmask |= pd.to_datetime(df[c], errors="coerce").between(d1, d2, inclusive="both")
        mask &= fmask

    if query_txt:
        q = str(query_txt).lower()
        txt_mask = (
            df.get("Descripciónn", pd.Series("", index=df.index)).astype(str).str.lower().str.contains(q, na=False) |
            df.get("item",         pd.Series("", index=df.index)).astype(str).str.lower().str.contains(q, na=False) |
            df.get("Sub-item",     pd.Series("", index=df.index)).astype(str).str.lower().str.contains(q, na=False)
        )
        mask &= txt_mask

    return df[mask].copy()


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
etapa_sel, estado_sel, prov_sel, rango_fechas, query_txt = filters
df_f = apply_filters(df, etapa_sel, estado_sel, prov_sel, rango_fechas, query_txt)


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
    st.markdown("### 🧩 Análisis por **Categorías**")
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        top_n = st.selectbox("Top N por monto", [5,10,15,20,30], index=2)
    with c2:
        modo = st.radio("Visualizar", ["Monto CLP", "% por Item (100%)"], index=0, horizontal=True)
    with c3:
        st.write("")  # espaciador

        # Presets de categorías
    cats_all = sorted(df[cat_col].dropna().unique())

    # Inicializa el valor controlado del multiselect
    if "cats_sm_sel" not in st.session_state:
        st.session_state["cats_sm_sel"] = cats_all

    p1, p2, p3, p4 = st.columns([1,1,1,1])
    with p1:
        preset_all = st.button("🟢 Todas")
    with p2:
        preset_sum = st.button("⚙️ Solo Suministro")
    with p3:
        preset_mon = st.button("🛠️ Solo Montaje")
    with p4:
        preset_id  = st.button("🧪 Solo I+D")

    # Al presionar un preset, actualiza directamente el valor del widget y relanza
    if preset_all:
        st.session_state["cats_sm_sel"] = cats_all
        _safe_rerun()
    if preset_sum:
        st.session_state["cats_sm_sel"] = [c for c in cats_all if str(c).strip().lower().startswith("suministro")]
        _safe_rerun()
    if preset_mon:
        st.session_state["cats_sm_sel"] = [c for c in cats_all if "montaje" in str(c).strip().lower()]
        _safe_rerun()
    if preset_id:
        st.session_state["cats_sm_sel"] = [c for c in cats_all if "i+d" in str(c).lower() or "i + d" in str(c).lower()]
        _safe_rerun()

    # El multiselect ahora está controlado por session_state (no dependas de 'default')
    cats_sel = st.multiselect("Categorías S/M", cats_all, key="cats_sm_sel")


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
                title="Items — Monto por categoría S/M"
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
# ===============================
# NUEVO — Tabla Factor × Item (elegante, autosuficiente)
# ===============================

st.markdown("""
<style>
  .fxi-card{border-radius:18px;padding:14px 16px;background:linear-gradient(180deg,#f8fafc 0%,#ffffff 62%);
            border:1px solid rgba(148,163,184,.35);box-shadow:0 6px 14px rgba(15,23,42,.06);margin-top:8px}
  .fxi-head{display:flex;align-items:center;gap:10px;margin:0 0 8px}
  .chip{display:inline-flex;align-items:center;gap:6px;font-size:12px;padding:2px 10px;border-radius:999px;
        border:1px solid rgba(148,163,184,.35);background:#eef2ff;color:#3730a3}
  .chip--ok{background:#e6fff6;color:#047857;border-color:#34d399}
  .fxi-card .stDataFrame{border-radius:14px;border:1px solid rgba(148,163,184,.25);}
  .fxi-card .stDataFrame [data-testid="stTable"] thead tr th{
      background:linear-gradient(180deg,#f1f5f9 0%,#eef2f7 100%); font-weight:700; color:#0f172a;
      border-bottom:1px solid rgba(148,163,184,.35);
  }
  .fxi-card .stDataFrame tbody tr:nth-child(odd){background:#fcfcff}
  .fxi-card .stDataFrame tbody tr:hover{background:#f0f9ff}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='fxi-card'><div class='fxi-head'>"
            "<h3 style='margin:0'>🧮 ¿Que factores son Necesario y Evitables para un próximo Piloto?</h3>"
            "</div>", unsafe_allow_html=True)

# -------- Calcular agg directamente aquí --------
df_cost = base.copy()
cat_col = "Suministro / montaje"
if cat_col not in df_cost.columns:
    df_cost[cat_col] = "Total"
cats_sel = st.session_state.get("cats_sm_sel", sorted(df_cost[cat_col].dropna().unique()))
if cats_sel:
    df_cost = df_cost[df_cost[cat_col].isin(cats_sel)]

for need_col in ["Monto", "Precio final ec esc", "Descuento ec escala", "Factor de costo", "item"]:
    if need_col not in df_cost.columns:
        df_cost[need_col] = np.nan

df_cost["Monto_num"]        = df_cost["Monto"].apply(_to_num_strict)
df_cost["Precio_final_num"] = df_cost["Precio final ec esc"].apply(_to_num_strict)
df_cost = df_cost[df_cost["Monto_num"].notna()].copy()

df_cost["Factor de costo"] = df_cost["Factor de costo"].fillna("Sin clasificar").astype(str)
df_cost["item"]            = df_cost["item"].fillna("(Vacío)").astype(str)

agg = (df_cost.groupby(["Factor de costo","item"], as_index=False)
              .agg(
                  n_items=("Monto_num","count"),
                  monto_total=("Monto_num","sum"),
                  precio_final_total=("Precio_final_num","sum")
              ))
agg["%_esc"] = np.where(
    agg["monto_total"]>0,
    (agg["precio_final_total"]/agg["monto_total"]*100).round(2),
    np.nan
)

# -------- Construir tabla de display + Justificación --------
# candidatos de nombre según tu fuente (URL/Sheet)
cand_just_cols = [
    "Justificación Nece - Evit",
    "Justificación Nece-Evit",
    "Justificación Necesario/Evitable",
    "Justificación % e E.E",  # fallback
]
src_just_col = next((c for c in cand_just_cols if c in df_cost.columns), None)
if src_just_col is None:
    df_cost["Justificación Nece - Evit"] = np.nan
    src_just_col = "Justificación Nece - Evit"

def _join_justifs(s: pd.Series, max_len: int = 200) -> str:
    vals = [str(x).strip() for x in s.dropna().astype(str) if str(x).strip()]
    if not vals: return ""
    txt = " · ".join(pd.unique(vals))
    return (txt[:max_len] + "…") if len(txt) > max_len else txt

just_tbl = (
    df_cost.groupby(["Factor de costo", "item"], dropna=False)[src_just_col]
           .apply(_join_justifs)
           .reset_index(name="Justificación Nece - Evit")
           .rename(columns={"Factor de costo": "Factor", "item": "Item"})
)

df_display = agg.rename(columns={
    "Factor de costo": "Factor",
    "item": "Item",
    "n_items": "Ítems",
    "monto_total": "Monto total (CLP)",
    "precio_final_total": "Precio final ec esc (CLP)",
    "%_esc": "% del monto al escalado"
})
df_display["Factor"] = df_display["Factor"].fillna("")
df_display = df_display[df_display["Factor"].str.strip().str.lower() != "sin clasificar"]
df_display = df_display.merge(just_tbl, on=["Factor","Item"], how="left")

# === Formato chileno a las columnas de monto (como TEXTO para la vista) ===
def _fmt_chileno(x):
    try:
        return "$ {:,}".format(int(round(float(x)))).replace(",", ".")
    except Exception:
        return x
df_display["Monto total (CLP)"] = df_display["Monto total (CLP)"].apply(_fmt_chileno)

# 👇 Quedarnos SOLO con estas columnas (se agrega la justificación)
df_display = df_display[["Factor", "Item", "Ítems", "Monto total (CLP)", "Justificación Nece - Evit"]]

# --- Tabla (compacta, solo hasta Monto total + justificación) ---
st.dataframe(
    df_display,
    use_container_width=True,
    height=460,
    hide_index=True,
    column_config={
        "Factor": st.column_config.TextColumn("Factor", width="medium"),
        "Item": st.column_config.TextColumn("Item", width="medium"),
        "Ítems": st.column_config.NumberColumn("Ítems", width="small", format="%d"),
        "Monto total (CLP)": st.column_config.TextColumn("Monto total (CLP)", width="small"),
        "Justificación Nece - Evit": st.column_config.TextColumn(
            "Justificación Nece - Evit", width="large",
            help="Motivo asociado al Factor (Necesario/Evitable) por Item"
        ),
    }
)

# --- Chips por Factor (Necesario vs Evitable) usando Monto_num ---
_tmp = df_cost.copy()
_tmp["__fac"] = (
    _tmp["Factor de costo"].fillna("").astype(str).str.strip().str.lower()
)
_tmp["__NE"] = np.where(
    _tmp["__fac"].str.contains("necesar"), "Necesario",
    np.where(_tmp["__fac"].str.contains("evit"), "Evitable", "Otro")
)

tot_nec = float(_tmp.loc[_tmp["__NE"]=="Necesario", "Monto_num"].sum() or 0.0)
tot_evi = float(_tmp.loc[_tmp["__NE"]=="Evitable", "Monto_num"].sum() or 0.0)
tot_all = tot_nec + tot_evi

pct_nec = (tot_nec / tot_all * 100.0) if tot_all > 0 else 0.0
pct_evi = (tot_evi / tot_all * 100.0) if tot_all > 0 else 0.0

st.markdown(
    f"<div style='display:flex;gap:8px;margin-top:10px'>"
    f"<span class='chip chip--ok'>Necesario: {_money_fmt(tot_nec)} · {pct_nec:.2f}%</span>"
    f"<span class='chip'>Evitable: {_money_fmt(tot_evi)} · {pct_evi:.2f}%</span>"
    f"</div>",
    unsafe_allow_html=True
)


# ===============================
# (Reemplazo) — Factor × Categoría (S/M/I+D)
# Barras apiladas por Factor, coloreadas por "Suministro / montaje"
# Muestra CLP + % dentro del Factor y %_esc por segmento
# ===============================
st.markdown("### 📊 Factor Necesario vs Evitable")

_fac_col = "Factor de costo"
_cat_col = "Suministro / montaje"

if (_fac_col not in df_cost.columns) or (_cat_col not in df_cost.columns):
    st.info("Faltan columnas 'Factor de costo' o 'Suministro / montaje' para construir el gráfico.")
else:
    fac_cat = (df_cost.copy()
               .assign(
                   Factor=df_cost[_fac_col].fillna("Sin clasificar").astype(str).str.strip(),
                   Categoria=df_cost[_cat_col].fillna("(Sin categoría)").astype(str).str.strip()
               ))

    # Enfocarnos en Necesario/Evitable y categorías S/M/I+D
    fac_cat = fac_cat[fac_cat["Factor"].str.lower() != "sin clasificar"]
    # normalizar nombres típicos
    fac_cat["Categoria"] = fac_cat["Categoria"].replace({
        "Suministro/montaje": "Montaje",
        "I + D": "I+D",
        "i+d": "I+D",
    })

    if fac_cat.empty:
        st.info("No hay datos clasificados como 'Necesario' o 'Evitable'.")
    else:
        aggFC = (fac_cat.groupby(["Factor","Categoria"], as_index=False)
                        .agg(Monto=("Monto_num","sum"),
                             PrecioEsc=("Precio_final_num","sum"),
                             Items=("Monto_num","count")))

        # %_esc por segmento (PrecioEsc / Monto)
        aggFC["%_esc"] = np.where(aggFC["Monto"] > 0,
                                  (aggFC["PrecioEsc"]/aggFC["Monto"]*100).round(2),
                                  np.nan)

        # % dentro de cada Factor (participación del segmento dentro del total del Factor)
        tot_por_factor = aggFC.groupby("Factor")["Monto"].transform("sum")
        aggFC["% dentro del Factor"] = np.where(tot_por_factor > 0,
                                                (aggFC["Monto"]/tot_por_factor*100).round(2),
                                                0.0)

        # Orden de Factor y de Categoría
        orden_factor = ["Necesario", "Evitable"]
        orden_cat    = ["Suministro", "I+D", "Montaje"]
        aggFC["__of"] = aggFC["Factor"].apply(lambda x: orden_factor.index(x) if x in orden_factor else 999)
        aggFC["__oc"] = aggFC["Categoria"].apply(lambda x: orden_cat.index(x) if x in orden_cat else 999)
        aggFC = aggFC.sort_values(["__of","__oc"]).drop(columns=["__of","__oc"])

        # Etiquetas compactas (CLP + % dentro del Factor)
        def _abbr_money(x: float) -> str:
            if x is None or (isinstance(x,float) and (np.isnan(x) or np.isinf(x))): return ""
            if abs(x) >= 1_000_000_000: return f"${x/1_000_000_000:,.2f}B"
            if abs(x) >= 1_000_000:     return f"${x/1_000_000:,.2f}M"
            if abs(x) >= 1_000:         return f"${x/1_000:,.0f}K"
            return f"${x:,.0f}"

        aggFC["label"] = aggFC.apply(
            lambda r: f"{_abbr_money(r['Monto'])} · {r['% dentro del Factor']:.2f}%",
            axis=1
        )

        # Colores consistentes con todo el dashboard
        color_map = PALETTE_SM  # {"Suministro": teal, "I+D": indigo, "Montaje": amber}

        figFC = px.bar(
            aggFC,
            x="Monto", y="Factor", orientation="h",
            color="Categoria", color_discrete_map=color_map,
            text="label",
            title="Montos Necesarios para un nuevo Piloto con el know How "
        )

        figFC.update_traces(
            textposition="inside",
            insidetextanchor="middle",
            cliponaxis=False,
            hovertemplate=(
                "<b>%{y}</b> · %{trace.name}<br>"
                "Monto seg.: $%{x:,.0f}<br>"
                "Participación en Factor: %{customdata[0]:.2f}%<br>"
                "%_esc seg.: %{customdata[1]:.2f}%<br>"
                "Ítems seg.: %{customdata[2]}<extra></extra>"
            ),
            customdata=np.stack([
                aggFC["% dentro del Factor"],
                aggFC["%_esc"],
                aggFC["Items"]
            ], axis=-1)
        )

        figFC.update_layout(
            barmode="stack",
            margin=dict(l=130, r=40, t=60, b=40),
            plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)",
            legend_title="Categoría (S/M/I+D)"
        )
        figFC.update_xaxes(separatethousands=True, tickprefix="$", gridcolor=GRID)
        figFC.update_yaxes(title="Factor", showgrid=False)

        # Anotar total por Factor al final de cada barra
        tot_factor_num = aggFC.groupby("Factor", as_index=False)["Monto"].sum()
        for _, r in tot_factor_num.iterrows():
            figFC.add_annotation(
                x=r["Monto"], y=r["Factor"], xanchor="left", yanchor="middle",
                text=_abbr_money(r["Monto"]), showarrow=False,
                font=dict(size=11, color="#475569"), xshift=10
            )

        st.plotly_chart(figFC, use_container_width=True)

# ===============================
# REEMPLAZO — Gráfico PRO (Monto inicial vs. Precio escalado)
# Evita sobreposición y muestra montos formateados + %_esc
# ===============================
st.markdown("### 📊 Economias de escalas para una etapa comercial ")

import plotly.graph_objects as go

# --- Base alineada a filtros activos (mismo patrón que el bloque anterior)
_df = base.copy()
_cat = "Suministro / montaje"
if _cat not in _df.columns:
    _df[_cat] = "Total"

_cats_sel = st.session_state.get("cats_sm_sel", sorted(_df[_cat].dropna().unique()))
if _cats_sel:
    _df = _df[_df[_cat].isin(_cats_sel)]

for c in ["Monto", "Precio final ec esc"]:
    if c not in _df.columns:
        _df[c] = np.nan

_df["Monto_num"]        = _df["Monto"].apply(_to_num_strict)
_df["Precio_final_num"] = _df["Precio final ec esc"].apply(_to_num_strict)

base_cat = (
    _df.groupby(_cat, as_index=False)
       .agg(Monto=("Monto_num","sum"),
            PrecioEsc=("Precio_final_num","sum"))
)

# --- Orden sugerido y %_esc
orden = ["Suministro", "I+D", "Montaje"]
base_cat["__ord"] = base_cat[_cat].apply(lambda x: orden.index(x) if x in orden else 999)
base_cat = base_cat.sort_values(["__ord","Monto"], ascending=[True, False]).drop(columns="__ord")
base_cat["%_esc"] = np.where(base_cat["Monto"]>0, (base_cat["PrecioEsc"]/base_cat["Monto"]*100).round(2), np.nan)

cats = base_cat[_cat].tolist()
monto = base_cat["Monto"].tolist()
esc   = base_cat["PrecioEsc"].tolist()
pct   = base_cat["%_esc"].tolist()

# --- Helper para etiquetas compactas (K/M)
def _abbr(x: float) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return ""
    if abs(x) >= 1_000_000_000:
        return f"${x/1_000_000_000:,.2f}B"
    if abs(x) >= 1_000_000:
        return f"${x/1_000_000:,.2f}M"
    if abs(x) >= 1_000:
        return f"${x/1_000:,.0f}K"
    return f"${x:,.0f}"

# --- Colores por CATEGORÍA (Suministro / I+D / Montaje) usando PALETTE_SM
def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16); g = int(hex_color[2:4], 16); b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

# mismo color por categoría: sólido para Monto inicial, más claro para Precio escalado
colors_init = [_hex_to_rgba(PALETTE_SM.get(cat, PRIMARY), 0.95) for cat in cats]
colors_esc  = [_hex_to_rgba(PALETTE_SM.get(cat, PRIMARY), 0.60) for cat in cats]

# Etiquetas más informativas
total_init = float(np.nansum(monto) or 0.0)
labels_init = [
    (f"{_abbr(x)} • {x/total_init*100:.2f}%" if total_init > 0 else _abbr(x))
    for x in monto
]
labels_esc  = [
    (f"{_abbr(x)} • {p:.2f}%_esc" if pd.notna(p) else _abbr(x))
    for x, p in zip(esc, pct)
]

# --- Figura con barras separadas (sin solaparse)
fig = go.Figure()

fig.add_trace(go.Bar(
    y=cats, x=monto, name="Monto inicial (CLP)", orientation="h",
    marker=dict(color=colors_init, line=dict(color="rgba(15,23,42,.20)", width=0.6)),
    text=labels_init, textposition="outside",
    hovertemplate="<b>%{y}</b><br>Monto inicial: $%{x:,.0f}<extra></extra>"
))
fig.add_trace(go.Bar(
    y=cats, x=esc, name="Precio escalado (CLP)", orientation="h",
    marker=dict(color=colors_esc, line=dict(color="rgba(15,23,42,.20)", width=0.6)),
    text=labels_esc, textposition="outside",
    hovertemplate="<b>%{y}</b><br>Precio escalado: $%{x:,.0f}<extra></extra>"
))

# Espacio a la derecha para textos
max_x = max([v for v in (monto + esc) if pd.notna(v)] + [0])
fig.update_xaxes(
    range=[0, max_x * 1.28],
    separatethousands=True, tickprefix="$",
    showgrid=True, gridcolor=GRID, zeroline=False
)

# Estética PRO
fig.update_layout(
    barmode="group", bargap=0.32, bargroupgap=0.24,
    margin=dict(l=120, r=32, t=56, b=48),
    plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)",
    title=dict(
        text="Totales por categoría (según filtros activos)",
        font=dict(size=18, color="#0f172a")
    ),
    legend=dict(
        title="Serie",
        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
        bgcolor="rgba(255,255,255,.75)", bordercolor="rgba(148,163,184,.35)", borderwidth=1
    ),
    uniformtext_minsize=11, uniformtext_mode="hide",
    font=dict(family="Inter, Segoe UI, system-ui, sans-serif", size=13, color="#0f172a"),
    hoverlabel=dict(bgcolor="white", bordercolor="rgba(148,163,184,.6)", font=dict(size=12))
)
fig.update_yaxes(title="Categoría", showgrid=False)



st.plotly_chart(fig, use_container_width=True)
# ===============================
# NUEVO — Tabla SOLO "Factor Necesario"
# (idéntica a la tabla principal, pero filtrada)
# ===============================

st.markdown("""
<style>
  .fxi-card2{border-radius:18px;padding:14px 16px;background:linear-gradient(180deg,#f8fafc 0%,#ffffff 62%);
             border:1px solid rgba(148,163,184,.35);box-shadow:0 6px 14px rgba(15,23,42,.06);margin-top:14px}
  .fxi-card2 .stDataFrame{border-radius:14px;border:1px solid rgba(148,163,184,.25);}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='fxi-card2'>"
            "<h3 style='margin:0 0 10px'>🧮 Factores Necesarios para el Piloto 10 kW ",
            unsafe_allow_html=True)

# --- Base desde df_cost (respeta filtros activos de S/M que aplicaste arriba)
_fx = df_cost.copy()

# Normaliza etiqueta de Factor (por si viniera con mayúsculas/espacios)
_fx["Factor de costo"] = _fx["Factor de costo"].fillna("").astype(str).str.strip()
_fx = _fx[_fx["Factor de costo"].str.lower().str.contains("necesario")]

if _fx.empty:
    st.info("No hay registros clasificados como **Factor Necesario** para los filtros actuales.")
else:
    # Asegurar columnas necesarias
    for c in ["Monto","Precio final ec esc","item","Justificación % e E.E"]:
        if c not in _fx.columns:
            _fx[c] = np.nan

    # Numéricos
    _fx["Monto_num"]        = _fx["Monto"].apply(_to_num_strict)
    _fx["Precio_final_num"] = _fx["Precio final ec esc"].apply(_to_num_strict)
    _fx["item"]             = _fx["item"].fillna("(Vacío)").astype(str)

    # Aggregación
    aggN = (_fx.groupby(["Factor de costo","item"], as_index=False)
              .agg(
                  n_items=("Monto_num","count"),
                  monto_total=("Monto_num","sum"),
                  precio_final_total=("Precio_final_num","sum")
              ))
    aggN["%_esc"] = np.where(
        aggN["monto_total"]>0,
        (aggN["precio_final_total"]/aggN["monto_total"]*100).round(2),
        np.nan
    )

    # Unir justificaciones (compuesto, sin duplicados)
    def _join_justifs2(s: pd.Series, max_len: int = 200) -> str:
        vals = [str(x).strip() for x in s.dropna().astype(str) if str(x).strip()]
        if not vals: return ""
        txt = " · ".join(pd.unique(vals))
        return (txt[:max_len] + "…") if len(txt) > max_len else txt

    justN = (_fx.groupby(["Factor de costo","item"], dropna=False)["Justificación % e E.E"]
               .apply(_join_justifs2)
               .reset_index(name="Justificación % e E.E"))

    # Tabla de display (mismos nombres/orden que la original)
    dfN = (aggN.merge(justN, on=["Factor de costo","item"], how="left")
               .rename(columns={
                   "Factor de costo": "Factor",
                   "item": "Item",
                   "n_items": "Ítems",
                   "monto_total": "Monto total (CLP)",
                   "precio_final_total": "Precio final ec esc (CLP)",
                   "%_esc": "% del monto al escalado"
               }))

    # Formato chileno para montos (como texto)
    def _fmt_chileno_local(x):
        try:
            return "$ {:,}".format(int(round(float(x)))).replace(",", ".")
        except Exception:
            return x
    dfN["Monto total (CLP)"]        = dfN["Monto total (CLP)"].apply(_fmt_chileno_local)
    dfN["Precio final ec esc (CLP)"] = dfN["Precio final ec esc (CLP)"].apply(_fmt_chileno_local)

    # Mostrar
    st.dataframe(
        dfN,
        use_container_width=True,
        height=420,
        hide_index=True,
        column_config={
            "Factor": st.column_config.TextColumn("Factor", width="medium"),
            "Item": st.column_config.TextColumn("Item", width="medium"),
            "Ítems": st.column_config.NumberColumn("Ítems", width="small", format="%d"),
            "Monto total (CLP)": st.column_config.TextColumn("Monto total (CLP)", width="small"),
            "Precio final ec esc (CLP)": st.column_config.TextColumn("Precio final ec esc (CLP)", width="small"),
            "% del monto al escalado": st.column_config.ProgressColumn(
                "% del monto al escalado",
                help="Relación Precio Final / Monto Original",
                min_value=0, max_value=100, format="%.2f%%"
            ),
            "Justificación % e E.E": st.column_config.TextColumn(
                "Justificación % e E.E", width="large",
                help="Motivos/explicaciones por Factor × Item"
            ),
        }
    )

    # Chips de resumen (como en la tabla original)
    total_pf_N = float(aggN["precio_final_total"].sum() or 0)
    prom_pct_N = float(aggN["%_esc"].dropna().mean() or 0)
    st.markdown(
        f"<div style='display:flex;gap:8px;margin-top:10px'>"
        f"<span class='chip chip--ok'>Precio final total: {_money_fmt(total_pf_N)}</span>"
        f"<span class='chip'>Promedio %_esc: {prom_pct_N:.2f}%</span>"
        f"</div>",
        unsafe_allow_html=True
    )


st.markdown("</div>", unsafe_allow_html=True)

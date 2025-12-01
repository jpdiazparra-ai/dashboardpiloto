# app.py
# Requisitos base: streamlit, pandas, plotly, numpy, networkx, python-dateutil
# Sugeridos para exportar: reportlab o fpdf2, y kaleido
# pip install streamlit pandas plotly numpy networkx python-dateutil reportlab fpdf2 kaleido

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from datetime import date
from io import BytesIO
import tempfile, os

# -------------------------------------------------
# Ajustes globales de estilo Plotly
# -------------------------------------------------
PX_COLOR_SEQ = [
    "#2563EB",  # azul principal
    "#16A34A",  # verde éxito
    "#F97316",  # naranjo riesgo
    "#DC2626",  # rojo atraso
    "#0EA5E9",  # celeste
    "#7C3AED",  # violeta
]

# Usar esta paleta por defecto en todos los gráficos Plotly Express
px.defaults.color_discrete_sequence = PX_COLOR_SEQ

# Fondo blanco limpio para todas las figuras
pio.templates.default = "plotly_white"


# Fallback para 'today' si corres en bare mode (sin contexto Streamlit)
if "today" not in st.session_state:
    st.session_state["today"] = date.today()

# -------- Config fuente de datos (Google Sheets CSV) --------
# URL de exportación directa del Google Sheet publicado.
# Se puede sobrescribir estableciendo la variable de entorno ``GSHEET_CSV_URL``.
GSHEET_CSV_URL = os.getenv(
    'GSHEET_CSV_URL',
    'https://docs.google.com/spreadsheets/d/e/2PACX-1vQOu_diukhhZWDV7kIcU9Ewto4lo_xQdSEZ0FMi2oto-Jb4r2e7aRNCBKF3qoVVk_4XsimMFx7eASkt/pub?gid=0&single=true&output=csv',
)
GSHEET_CACHE_TTL = int(os.getenv('GSHEET_CACHE_TTL', '300'))


# -------- Helpers --------
DATE_COL_START = "Inicio (AAAA-MM-DD)"
DATE_COL_END_PLAN = "Fin plan (AAAA-MM-DD)"
DATE_COL_END_REAL = "Fin real"

def parse_dependencies(s):
    if pd.isna(s) or s in ["—", "-", ""]:
        return []
    if isinstance(s, (int, float)) and not pd.isna(s):
        return [int(s)]
    parts = str(s).replace(" ", "").split(",")
    out = []
    for p in parts:
        try:
            out.append(int(p))
        except Exception:
            pass
    return out

def status_color(s):
    s = (s or "").lower()
    if "complet" in s:
        return "#2ca02c"  # verde
    if "curso" in s:
        return "#de1a3b"  # rojo
    if "planific" in s:
        return "#1778e1"  # azul
    if "recurrente" in s:
        return "#C5C213"  # amarillo
    if "pend" in s:
        return "#ff7f0e"  # naranja
    return "#7f7f7f"      # gris

def infer_piloto(row):
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
    try:
        if int(row.get("ID", 0)) in {24, 25}:
            return "Piloto 55 kW"
    except Exception:
        pass
    return "Piloto 10 kW"

def compute_risk(row, today):
    estado = str(row.get("Estado", "")).lower()
    fin_plan = row.get(DATE_COL_END_PLAN)
    today_ts = pd.to_datetime(today)
    days_left = (fin_plan - today_ts).days if pd.notna(fin_plan) else 9999
    if "complet" in estado:
        prob = 1
    else:
        if days_left <= 7:
            prob = 3
        elif days_left <= 14:
            prob = 2
        else:
            prob = 1
    impact = row.get("_impact_tmp", 2)
    return prob, impact

def build_burnup_fig(df):
    df_burn = df.copy()
    df_burn["_fecha_done"] = df_burn[DATE_COL_END_REAL].fillna(df_burn[DATE_COL_END_PLAN])
    df_burn = df_burn.dropna(subset=["_fecha_done"]).sort_values("_fecha_done")
    if df_burn.empty:
        return None

    daily = df_burn.groupby("_fecha_done")["ID"].count().rename("Completadas_día")
    cum = daily.cumsum().rename("Completadas_acum").to_frame()
    cum["Totales"] = len(df)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cum.index,
        y=cum["Completadas_acum"],
        mode="lines+markers",
        name="Tareas completadas",
    ))
    fig.add_trace(go.Scatter(
        x=cum.index,
        y=cum["Totales"],
        mode="lines",
        name="Tareas totales planificadas",
        line=dict(dash="dash"),
    ))

    fig.update_layout(
        title="Evolución de tareas completadas vs total planificado",
        xaxis_title="Fecha",
        yaxis_title="Nº de tareas",
        hovermode="x unified",
        height=320,
        margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right", x=1
        ),
    )
    return fig

# -------- Exportadores --------
def make_pdf_summary(df, fdf, fig_gantt=None, fig_burn=None, project_title="Tablero Proyecto Eólico"):
    """Genera PDF con reportlab; si falla, intenta fpdf2; si falta kaleido, omite imágenes."""
    total = len(fdf)
    done = (fdf["Estado"].str.contains("Complet", case=False, na=False)).sum()
    in_course = (fdf["Estado"].str.contains("curso", case=False, na=False)).sum()
    planned = (fdf["Estado"].str.contains("Planific", case=False, na=False)).sum() + \
              (fdf["Estado"].str.contains("Recurrente", case=False, na=False)).sum()
    today_value = st.session_state.get("today", date.today())
    late = ((fdf[DATE_COL_END_PLAN] < pd.to_datetime(today_value))
            & (~fdf["Estado"].str.contains("Complet", case=False, na=False))).sum()
    progress_avg = np.nanmean(pd.to_numeric(fdf["%"], errors="coerce")) if "%" in fdf.columns else np.nan

    today_ts = pd.to_datetime(today_value)
    soon = fdf[(fdf[DATE_COL_START] >= today_ts) &
               (fdf[DATE_COL_START] <= today_ts + pd.Timedelta(days=60))] \
            .sort_values(DATE_COL_START) \
            .head(6)[["ID","Tarea / Entregable", DATE_COL_START, DATE_COL_END_PLAN]].copy()

    # Top riesgos simple
    finp = fdf[DATE_COL_END_PLAN]
    days_left = (finp - today_ts).dt.days
    prob = np.where(fdf["Estado"].str.contains("complet", case=False, na=False), 1,
                    np.where(days_left<=7,3, np.where(days_left<=14,2,1)))
    sev = prob * 1  # impacto básico=1 si no calculas con grafo
    rtop = fdf.assign(Severidad=sev).sort_values("Severidad", ascending=False) \
              .loc[:, ["ID","Tarea / Entregable","Severidad"]].head(6)

    # ---- Intento 1: reportlab
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

        def _img(fig, wcm, hcm):
            if fig is None: return None
            try:
                png = fig.to_image(format="png", scale=2, width=1000, height=400)  # requiere kaleido
                return Image(BytesIO(png), width=wcm*cm, height=hcm*cm)
            except Exception:
                return None

        buf = BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=1.2*cm, bottomMargin=1.2*cm,
                                leftMargin=1.4*cm, rightMargin=1.4*cm)
        styles = getSampleStyleSheet()
        h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=18, spaceAfter=8)
        h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=14, spaceAfter=6)
        p  = styles["BodyText"]

        elements = [
            Paragraph(project_title, h1),
            Paragraph(f"Fecha de generación: {pd.Timestamp.now():%Y-%m-%d %H:%M}", p),
            Spacer(1, 0.4*cm)
        ]

        kpi_data = [
            ["Tareas totales", total, "Completadas", done, "En curso", in_course],
            ["Planificadas", planned, "Atrasadas", late, "Avance prom.", f"{0 if np.isnan(progress_avg) else round(float(progress_avg),1)}%"],
        ]
        kpi_tbl = Table(kpi_data, colWidths=[3.2*cm, 2.2*cm, 3.2*cm, 2.2*cm, 3.2*cm, 2.2*cm])
        kpi_tbl.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0), colors.whitesmoke),
            ("BOX",(0,0),(-1,-1), 0.5, colors.grey),
            ("GRID",(0,0),(-1,-1), 0.25, colors.grey),
            ("ALIGN",(1,0),(-1,-1),"CENTER"),
        ]))
        elements += [kpi_tbl, Spacer(1, 0.4*cm)]

        g_img = _img(fig_gantt, 17, 6)
        if g_img: elements += [Paragraph("Línea de tiempo (snapshot)", h2), g_img, Spacer(1, 0.3*cm)]
        b_img = _img(fig_burn, 17, 5)
        if b_img: elements += [Paragraph("Burn-up (completadas acumuladas)", h2), b_img]

        elements += [PageBreak(), Paragraph("Próximos hitos (60 días)", h2)]
        if not soon.empty:
            s = soon.copy()
            s[DATE_COL_START]    = s[DATE_COL_START].dt.strftime("%Y-%m-%d")
            s[DATE_COL_END_PLAN] = s[DATE_COL_END_PLAN].dt.strftime("%Y-%m-%d")
            soon_tbl = Table([["ID","Tarea","Inicio","Fin plan"]] + s.values.tolist(),
                             colWidths=[1.5*cm, 9.0*cm, 3.5*cm, 3.5*cm])
            soon_tbl.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0), colors.whitesmoke),
                                          ("GRID",(0,0),(-1,-1), 0.25, colors.grey)]))
            elements.append(soon_tbl)
        else:
            elements.append(Paragraph("No hay hitos en las próximas 8 semanas.", p))

        elements += [Spacer(1, 0.3*cm), Paragraph("Top riesgos por severidad", h2)]
        if not rtop.empty:
            rt_tbl = Table([["ID","Tarea","Severidad"]] + rtop.values.tolist(),
                           colWidths=[1.5*cm, 12.0*cm, 3.0*cm])
            rt_tbl.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0), colors.whitesmoke),
                                        ("GRID",(0,0),(-1,-1), 0.25, colors.grey)]))
            elements.append(rt_tbl)
        else:
            elements.append(Paragraph("No se identifican riesgos destacados.", p))

        doc.build(elements)
        return buf.getvalue()

    except Exception:
        # ---- Intento 2: fpdf2
        try:
            from fpdf import FPDF
        except Exception:
            return None  # sin libs → fallback HTML

        pdf = FPDF(orientation="P", unit="mm", format="A4")
        pdf.set_auto_page_break(auto=True, margin=12)
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.multi_cell(0, 8, project_title)
        pdf.set_font("Helvetica", size=11)
        pdf.cell(0, 6, f"Fecha de generación: {pd.Timestamp.now():%Y-%m-%d %H:%M}", ln=1)
        kpis = f"Tareas: {total} | Completadas: {done} | En curso: {in_course} | Planif.: {planned} | Atrasadas: {late} | Avance prom.: {0 if np.isnan(progress_avg) else round(float(progress_avg),1)}%"
        pdf.multi_cell(0, 6, kpis)
        pdf.ln(2)

        def _add_fig(fig, w=190, h=80):
            if fig is None: return
            try:
                png = fig.to_image(format="png", scale=2, width=1000, height=400)  # kaleido
            except Exception:
                return
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                tmp.write(png); path = tmp.name
            try:
                pdf.image(path, w=w, h=h); pdf.ln(2)
            finally:
                try: os.remove(path)
                except: pass

        _add_fig(fig_gantt, 190, 80)
        _add_fig(fig_burn,  190, 70)

        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 7, "Próximos hitos (60 días)", ln=1)
        pdf.set_font("Helvetica", size=10)
        if not soon.empty:
            for _, r in soon.iterrows():
                s_ini = r[DATE_COL_START].strftime("%Y-%m-%d")
                s_fin = r[DATE_COL_END_PLAN].strftime("%Y-%m-%d")
                pdf.multi_cell(0, 5, f"#{int(r['ID'])} | {r['Tarea / Entregable']} | {s_ini} → {s_fin}")
        else:
            pdf.multi_cell(0, 5, "No hay hitos próximos.")
        pdf.ln(2)

        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 7, "Top riesgos", ln=1)
        pdf.set_font("Helvetica", size=10)
        if not rtop.empty:
            for _, r in rtop.iterrows():
                pdf.multi_cell(0, 5, f"#{int(r['ID'])} | {r['Tarea / Entregable']} | Sev: {int(r['Severidad'])}")
        else:
            pdf.multi_cell(0, 5, "Sin riesgos destacados.")

        out = BytesIO()
        pdf.output(out, dest="S")
        return out.getvalue()

def make_html_summary(df, fdf, fig_gantt=None, fig_burn=None, project_title="Tablero Proyecto Eólico"):
    total = len(fdf)
    done = (fdf["Estado"].str.contains("Complet", case=False, na=False)).sum()
    in_course = (fdf["Estado"].str.contains("curso", case=False, na=False)).sum()
    planned = (fdf["Estado"].str.contains("Planific", case=False, na=False)).sum() + \
              (fdf["Estado"].str.contains("Recurrente", case=False, na=False)).sum()
    today_value = st.session_state.get("today", date.today())
    late = ((fdf[DATE_COL_END_PLAN] < pd.to_datetime(today_value))
            & (~fdf["Estado"].str.contains("Complet", case=False, na=False))).sum()
    progress_avg = np.nanmean(pd.to_numeric(fdf["%"], errors="coerce")) if "%" in fdf.columns else np.nan

    today_ts = pd.to_datetime(today_value)
    soon = fdf[(fdf[DATE_COL_START] >= today_ts) &
               (fdf[DATE_COL_START] <= today_ts + pd.Timedelta(days=60))] \
            .sort_values(DATE_COL_START) \
            .head(6)[["ID","Tarea / Entregable", DATE_COL_START, DATE_COL_END_PLAN]].copy()
    if not soon.empty:
        soon[DATE_COL_START]    = soon[DATE_COL_START].dt.strftime("%Y-%m-%d")
        soon[DATE_COL_END_PLAN] = soon[DATE_COL_END_PLAN].dt.strftime("%Y-%m-%d")

    g_html = pio.to_html(fig_gantt, full_html=False, include_plotlyjs='cdn') if fig_gantt else ""
    b_html = pio.to_html(fig_burn,  full_html=False, include_plotlyjs=False) if fig_burn else ""

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>{project_title}</title>
<style>
body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 24px; }}
.kpis {{ display: grid; grid-template-columns: repeat(3,1fr); gap: 8px; margin: 8px 0 16px; }}
.kpis div {{ background:#f6f6f6; padding:8px 10px; border:1px solid #ddd; border-radius:8px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border:1px solid #ddd; padding:6px 8px; text-align:left; }}
th {{ background:#fafafa; }}
</style></head><body>
<h1>{project_title}</h1>
<p>Fecha de generación: {pd.Timestamp.now():%Y-%m-%d %H:%M}</p>
<div class="kpis">
  <div><b>Tareas</b><br>{total}</div>
  <div><b>Completadas</b><br>{done}</div>
  <div><b>En curso</b><br>{in_course}</div>
  <div><b>Planificadas</b><br>{planned}</div>
  <div><b>Atrasadas</b><br>{late}</div>
  <div><b>Avance prom.</b><br>{0 if np.isnan(progress_avg) else round(float(progress_avg),1)}%</div>
</div>
<h2>Línea de tiempo (snapshot)</h2>{g_html}
<h2>Burn-up</h2>{b_html}
<h2>Próximos hitos (60 días)</h2>
{soon.to_html(index=False) if not soon.empty else "<p>No hay hitos en las próximas 8 semanas.</p>"}
</body></html>"""
    return html.encode("utf-8")
def make_pdf_table(table_df, show_cols, title="Tabla de tareas filtrada – Proyecto Eólico"):
    """
    Genera un PDF simple con la tabla filtrada que se muestra en la pestaña TABLA.
    Usa reportlab si está disponible; si no, devuelve None.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import cm
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

        buf = BytesIO()
        doc = SimpleDocTemplate(
            buf,
            pagesize=A4,
            topMargin=1.2 * cm,
            bottomMargin=1.2 * cm,
            leftMargin=1.2 * cm,
            rightMargin=1.2 * cm,
        )

        styles = getSampleStyleSheet()
        elements = []
        elements.append(Paragraph(title, styles["Heading1"]))
        elements.append(
            Paragraph(
                f"Fecha de generación: {pd.Timestamp.now():%Y-%m-%d %H:%M}",
                styles["Normal"],
            )
        )
        elements.append(Spacer(1, 0.5 * cm))

        # Copia de la tabla con solo las columnas visibles
        df_pdf = table_df[show_cols].copy()

        # Formatear fechas como texto
        for c in df_pdf.columns:
            if np.issubdtype(df_pdf[c].dtype, np.datetime64):
                df_pdf[c] = df_pdf[c].dt.strftime("%Y-%m-%d")

        data = [list(df_pdf.columns)] + df_pdf.astype(str).values.tolist()

        tbl = Table(data, repeatRows=1)
        tbl.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                ]
            )
        )

        elements.append(tbl)
        doc.build(elements)
        return buf.getvalue()
    except Exception:
        return None


def gantt_pro(df, date_mode="Plan", color_by="Estado"):
    """
    Gantt estilo 'Apple-like': minimal, aireado y con interacciones útiles.
    Compatible con Python 3.12 / Plotly 5.x / Streamlit >=1.33
    """
    dfp = df.copy()

    # --- Fechas saneadas ---
    dfp["_start"]     = pd.to_datetime(dfp.get(DATE_COL_START),    errors="coerce")
    dfp["_end_plan"]  = pd.to_datetime(dfp.get(DATE_COL_END_PLAN), errors="coerce")
    dfp["_end_real"]  = pd.to_datetime(dfp.get(DATE_COL_END_REAL), errors="coerce")

    dfp["_start"] = dfp["_start"].fillna(dfp["_end_real"]).fillna(dfp["_end_plan"])
    dfp["_end"]   = dfp["_end_real"] if date_mode == "Real" else dfp["_end_plan"]
    dfp["_end"]   = dfp["_end"].fillna(dfp["_end_plan"]).fillna(dfp["_end_real"]).fillna(dfp["_start"])

    bad = dfp["_end"] <= dfp["_start"]
    dfp.loc[bad, "_end"] = dfp.loc[bad, "_start"] + pd.Timedelta(days=1)

    dfp = dfp[dfp["_start"].notna() & dfp["_end"].notna()].copy()
    if dfp.empty:
        return go.Figure()

    # --- Orden y etiquetas ---
    dfp = dfp.sort_values(["_start", "_end", "ID"], ascending=[False, False, True])
    y_labels = pd.unique(dfp["Tarea / Entregable"].astype(str))
    max_len  = max((len(s) for s in y_labels), default=12)
    rows     = len(y_labels)
    left_margin = min(56 + max_len * 6, 380)
    height_px   = min(max(460, 28 * rows), 1600)

    # --- Paleta inspirada en iOS/macOS ---
    APPLE = {
        "Planificado": "#0A84FF",   # iOS Blue
        "En curso":    "#FF3B30",   # iOS Red
        "Completado":  "#30D158",   # iOS Green
        "Pendiente":   "#FF9F0A",   # iOS Orange
        "Recurrente":  "#C5C213",   # soft yellow
        "default":     "#8E8E93"    # iOS Gray
    }
    def color_estado(s):
        s = (s or "").strip().lower()
        if "plan"   in s: return APPLE["Planificado"]
        if "curso"  in s: return APPLE["En curso"]
        if "complet" in s:return APPLE["Completado"]
        if "pend"   in s: return APPLE["Pendiente"]
        if "recurrente" in s: return APPLE["Recurrente"]
        return APPLE["default"]

    color_arg = color_by if color_by in dfp.columns else "Estado"
    color_map = None

    if color_by == "Estado":
        # Mapea dinámicamente los estados presentes
        color_map = {val: color_estado(val) for val in dfp["Estado"].dropna().unique()}

    elif color_by == "Piloto" and "Piloto" in dfp.columns:
        # Colores consistentes para cada piloto usando la paleta global PX_COLOR_SEQ
        pilotos = dfp["Piloto"].dropna().unique()
        color_map = {p: PX_COLOR_SEQ[i % len(PX_COLOR_SEQ)] for i, p in enumerate(pilotos)}


    # --- Hover data seguro ---
    hover_cols = ["ID","Fase","Línea","Responsable","Ubicación","%","Depende de","Hito (S/N)","Riesgo clave","Piloto"]
    hover_cols = [c for c in hover_cols if c in dfp.columns]

       # --- Gráfico base ---
    fig = px.timeline(
        dfp,
        x_start="_start", x_end="_end",
        y="Tarea / Entregable",
        color=color_arg,
        color_discrete_map=color_map,
        category_orders={"Tarea / Entregable": y_labels},
        hover_data=hover_cols,
        text="%" if "%" in dfp.columns else None,  # ← aquí vinculamos la columna %
        opacity=0.95,
    )

    # Mostrar % dentro de las barras con formato bonito
    if "%" in dfp.columns:
        fig.update_traces(
            texttemplate="%{text:.0f}%",
            textposition="inside",
            insidetextanchor="middle",
            selector=dict(type="bar"),
        )



    # --- Hovertemplate: limpio y consistente ---
    dfp["dur_dias"] = (dfp["_end"] - dfp["_start"]).dt.days.clip(lower=1)
    hovertemplate = (
        "<b>%{y}</b><br>"
        "Fase: %{customdata[0]}<br>" if "Fase" in hover_cols else "<b>%{y}</b><br>"
    )
    # Asignamos un customdata mínimo: (Fase, Estado, ID, Duración, %)
    safe_pct = pd.to_numeric(dfp.get("%", 0), errors="coerce").fillna(0).astype(float)
    fig.update_traces(
        customdata=np.stack([
            dfp.get("Fase", pd.Series([""]*len(dfp))).astype(str),
            dfp.get("Estado", pd.Series([""]*len(dfp))).astype(str),
            dfp.get("ID",    pd.Series([""]*len(dfp))),
            dfp["dur_dias"].astype(int),
            safe_pct.round(0)
        ], axis=1),
        hovertemplate="<b>%{y}</b><br>"
                      "Estado: %{customdata[1]} · ID: %{customdata[2]}<br>"
                      "Inicio: %{x|%Y-%m-%d}<br>"
                      "Fin: %{x|%Y-%m-%d}<extra>Duración: %{customdata[3]} días · Avance: %{customdata[4]}%</extra>"
    )

    # --- Línea de Hoy + etiqueta ---
    hoy_pd  = pd.Timestamp(st.session_state.get("today", pd.Timestamp.today().date()))
    hoy_iso = hoy_pd.strftime("%Y-%m-%d")
    fig.add_shape(
        type="line", x0=hoy_iso, x1=hoy_iso, y0=0, y1=1, xref="x", yref="paper",
        line=dict(dash="dot", width=2, color="#1C1C1E")  # dark gray
    )
    fig.add_annotation(
        x=hoy_iso, y=1.02, xref="x", yref="paper",
        text="Hoy", showarrow=False,
        font=dict(size=12, color="#1C1C1E"),
        bgcolor="rgba(255,255,255,0.7)"
    )

    # --- Milestones (Hito = 'S') como puntos al final ---
    if "Hito (S/N)" in dfp.columns:
        hitos = dfp[dfp["Hito (S/N)"].astype(str).str.upper().eq("S")].copy()
        if not hitos.empty:
            fig.add_trace(go.Scatter(
                x=hitos["_end"], y=hitos["Tarea / Entregable"],
                mode="markers",
                marker=dict(size=10, symbol="diamond", line=dict(width=1, color="#1C1C1E")),
                name="Hito",
                hovertext=hitos["Tarea / Entregable"],
                hoverinfo="text"
            ))

    # --- Resalte sutil de atrasadas (borde) ---
    if "Estado" in dfp.columns:
        late_mask = (dfp["_end_plan"] < hoy_pd) & (~dfp["Estado"].str.lower().str.contains("complet"))
        # No existe “border” por barra en timeline, usamos ligera opacidad +
        # color más saturado si está atrasada (truco visual)
        for i, tr in enumerate(fig.data):
            try:
                # Solo aplica a las barras (no a milestones)
                if getattr(tr, "orientation", "h") == "h":
                    sel = late_mask[tr.y].values if isinstance(tr.y, (list, np.ndarray, pd.Series)) else None
            except Exception:
                sel = None
            # Si no logramos aplicar por barra, mantenemos estilo base.
        # (Plotly no permite borde por barra en timeline; el truco visual anterior suele bastar)

    # --- Rango/Zoom y grilla suave ---
    fig.update_xaxes(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=3, label="3m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(step="year", stepmode="todate", label="YTD"),
                dict(count=1, step="year", stepmode="backward", label="1y"),
                dict(step="all", label="All")
            ])
        ),
        rangeslider=dict(visible=True),
        showgrid=True,
        gridcolor="rgba(60,60,67,0.08)"
    )
    fig.update_yaxes(autorange="reversed", automargin=True, showticklabels=True)

    # --- Layout “Apple-like” ---
    fig.update_layout(
        height=height_px,
        margin=dict(l=left_margin, r=32, t=36, b=18),
        plot_bgcolor="#FBFBFD",
        paper_bgcolor="#FFFFFF",
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right",  x=1,
            bgcolor="rgba(255,255,255,0.6)",
            borderwidth=0
        ),
        xaxis_title=None, yaxis_title=None
    )

    # Guardar métricas en sesión
    st.session_state["gantt_adjusted"] = int(bad.sum())
    st.session_state["gantt_omitted"]  = 0

    return fig


# Back-compat: cualquier llamada a 'gantt' usa la versión pro
gantt = gantt_pro

def tune_gantt_for_export(fig):
    """Copia la figura de Gantt y la ajusta para exportación (más margen/altura)."""
    f2 = go.Figure(fig)  # copia, no modifica la que se ve en pantalla
    labels = []
    for tr in f2.data:
        y = getattr(tr, "y", None)
        if y is not None:
            labels.extend([str(v) for v in y])
    uniq = list(dict.fromkeys(labels))
    n_rows = len(uniq)
    max_len = max((len(s) for s in uniq), default=10)
    left = min(40 + max_len * 6, 360)       # margen izquierdo amplio para etiquetas
    height = min(max(420, 26 * n_rows), 1600)
    f2.update_yaxes(automargin=True)
    f2.update_layout(margin=dict(l=left, r=24, t=40, b=24), height=height)
    return f2

# -------- ETL --------
def process_df(df: pd.DataFrame) -> pd.DataFrame:
    if "ID" in df.columns:
        df["ID"] = pd.to_numeric(df["ID"], errors="coerce").astype("Int64")
    for c in [DATE_COL_START, DATE_COL_END_PLAN, DATE_COL_END_REAL]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
        else:
            df[c] = pd.NaT
    if "Depende de" in df.columns:
        df["Depende de"] = df["Depende de"].apply(lambda v: "" if pd.isna(v) else (",".join(map(str, parse_dependencies(v)))))
        df["_deps"] = df["Depende de"].apply(parse_dependencies)
    else:
        df["Depende de"] = ""
        df["_deps"] = [[] for _ in range(len(df))]
    df["DuraciónPlan(d)"] = (df[DATE_COL_END_PLAN] - df[DATE_COL_START]).dt.days
    df["DuraciónReal(d)"] = (df[DATE_COL_END_REAL] - df[DATE_COL_START]).dt.days
    df["DuraciónPlan(d)"] = pd.to_numeric(df["DuraciónPlan(d)"], errors="coerce")
    df["DuraciónReal(d)"] = pd.to_numeric(df["DuraciónReal(d)"], errors="coerce")
    df["Piloto"] = df.apply(infer_piloto, axis=1)

    # --- Normalizar columna de avance "%" ---
    if "%" in df.columns:
        # 1) Convertir siempre a numérico
        df["%"] = pd.to_numeric(df["%"], errors="coerce")

        # 2) Si viene como 0–1 (1 = 100%), escalar
        try:
            max_val = df["%"].max()
            if pd.notna(max_val) and max_val <= 1:
                df["%"] = df["%"] * 100
        except Exception:
            pass

        # 3) Si está "Completado" y el % está vacío, forzar 100
        if "Estado" in df.columns:
            mask_done = df["Estado"].str.contains("complet", case=False, na=False) & df["%"].isna()
            df.loc[mask_done, "%"] = 100

    return df



@st.cache_data(ttl=GSHEET_CACHE_TTL)
def load_from_gsheet_csv(csv_url: str) -> pd.DataFrame:
    df = pd.read_csv(csv_url, encoding="utf-8-sig")
    return process_df(df)

# -------- Sidebar --------
st.sidebar.title("⚙️ Control")
st.sidebar.text_input("Fuente de datos (Google Sheets CSV):", GSHEET_CSV_URL, disabled=True)
today_widget = st.sidebar.date_input("Fecha de referencia (hoy)", value=st.session_state["today"], key="today")

try:
    df = load_from_gsheet_csv(GSHEET_CSV_URL)
except Exception as e:
    st.error(f"No pude leer el CSV de Google Sheets.\nDetalles: {e}")
    st.stop()

if "refresh_key" not in st.session_state:
    st.session_state.refresh_key = 0
if st.sidebar.button("🔄 Actualizar datos (limpiar caché)"):
    st.session_state.refresh_key += 1
    st.cache_data.clear()
    st.toast("Datos actualizados desde la fuente")
    st.rerun()
st.sidebar.caption(f"Última actualización: {pd.Timestamp.now():%Y-%m-%d %H:%M:%S}")

# -------- Sin filtros (usar dataset completo) --------
# Trabajamos con el DataFrame tal cual viene de la fuente
fdf = df.copy()


# -------- Header / KPIs --------
st.title("🚀 Tablero Proyecto Eólico")
st.caption("Seguimiento integrado del proyecto: avance, ruta crítica y riesgos.")


# ======================
# 🧭 HEADER KPIs – Estilo Apple
# ======================

# --- Métricas base ---
total = len(fdf)
done = (fdf["Estado"].str.contains("Complet", case=False, na=False)).sum()
in_course = (fdf["Estado"].str.contains("curso", case=False, na=False)).sum()
planned = (fdf["Estado"].str.contains("Planific", case=False, na=False)).sum() + \
           (fdf["Estado"].str.contains("Recurrente", case=False, na=False)).sum()
pending = (fdf["Estado"].str.contains("Pend", case=False, na=False)).sum()
late = ((fdf[DATE_COL_END_PLAN] < pd.to_datetime(st.session_state["today"])) &
        (~fdf["Estado"].str.contains("Complet", case=False, na=False))).sum()
progress_avg = np.nanmean(pd.to_numeric(fdf["%"], errors="coerce")) if "%" in fdf.columns else np.nan

# --- Estilo CSS para header de KPIs (3 principales) ---
st.markdown("""
<style>
.kpi-grid {
  display: grid;
  grid-template-columns: 2fr 1.2fr 1.2fr;
  gap: 18px;
  margin-top: 10px;
  margin-bottom: 12px;
}
.kpi-card {
  background: linear-gradient(180deg, #FFFFFF 0%, #F9FAFB 100%);
  border-radius: 18px;
  border: 1px solid #E5E7EB;
  padding: 16px 20px;
  box-shadow: 0 1px 2px rgba(15,23,42,0.04);
}
.kpi-label {
  font-size: 0.85rem;
  color: #6B7280;
}
.kpi-main {
  font-size: 2.4rem;
  font-weight: 600;
  color: #111827;
  margin-top: 2px;
}
.kpi-tag {
  font-size: 0.8rem;
  color: #9CA3AF;
  margin-top: 4px;
}
.kpi-bad {
  color: #DC2626;
}
</style>
""", unsafe_allow_html=True)

# --- HTML del nuevo header de KPIs ---
st.markdown(f"""
<div class="kpi-grid">
  <div class="kpi-card">
    <div class="kpi-label">Avance promedio del proyecto</div>
    <div class="kpi-main">{0 if np.isnan(progress_avg) else progress_avg:.0f}%</div>
    <div class="kpi-tag">Sobre {total} tareas planificadas</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Tareas atrasadas</div>
    <div class="kpi-main kpi-bad">{late}</div>
    <div class="kpi-tag">Plan &lt; hoy y sin completar</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Tareas completadas</div>
    <div class="kpi-main">{done}</div>
    <div class="kpi-tag">De {total} tareas totales</div>
  </div>
</div>
""", unsafe_allow_html=True)

# Detalle secundario de tareas en un expander
with st.expander("Ver detalle de estado de tareas"):
    c1, c2, c3 = st.columns(3)
    c1.metric("En curso", in_course)
    c2.metric("Planificadas", planned)
    c3.metric("Pendientes", pending)

if not np.isnan(progress_avg):
    st.progress(float(progress_avg)/100.0)


# -------- Dependencias / Grafo --------
def build_graph(df):
    try:
        import networkx as nx
    except ImportError:
        class DummyGraph:
            def __len__(self): return 0
            def nodes(self, data=False): return []
            def edges(self, data=False): return []
            def out_degree(self): return []
            def in_degree(self): return []
        return DummyGraph()

    G = nx.DiGraph()
    for _, r in df.iterrows():
        if pd.isna(r.get("ID")): continue
        dur = r.get("DuraciónPlan(d)")
        dur = int(dur) if pd.notna(dur) else 1
        G.add_node(int(r["ID"]), duration=max(1, dur))
    for _, r in df.iterrows():
        if pd.isna(r.get("ID")): continue
        i = int(r["ID"])
        for d in r.get("_deps", []):
            if d in G.nodes:
                G.add_edge(int(d), i)
    return G

def longest_path_by_duration(G):
    try:
        import networkx as nx
    except ImportError:
        return [], 0
    try:
        if G is None or len(G) == 0:
            return [], 0
    except Exception:
        return [], 0
    H = nx.DiGraph()
    for n, data in G.nodes(data=True):
        H.add_node(n, duration=data.get("duration", 1))
    for u, v in G.edges():
        H.add_edge(u, v, w=G.nodes[v].get("duration", 1))
    try:
        order = list(nx.topological_sort(H))
    except nx.NetworkXUnfeasible:
        return [], 0
    dist = {n: H.nodes[n].get("duration", 1) for n in H.nodes()}
    prev = {n: None for n in H.nodes()}
    for n in order:
        for _, v, data in H.out_edges(n, data=True):
            w = data.get("w", 1)
            if dist[n] + w > dist[v]:
                dist[v] = dist[n] + w
                prev[v] = n
    if not dist: return [], 0
    end = max(dist, key=dist.get)
    path, cur = [], end
    while cur is not None:
        path.append(cur); cur = prev[cur]
    path.reverse()
    return path, dist[end]

def critical_path_fallback(df: pd.DataFrame):
    rows = df.dropna(subset=["ID"]).copy()
    if rows.empty: return [], 0, []
    rows["ID"] = rows["ID"].astype(int)
    dur = rows.set_index("ID")["DuraciónPlan(d)"].fillna(1).clip(lower=1).astype(int).to_dict()
    adj = {i: [] for i in dur}
    indeg = {i: 0 for i in dur}
    invalid_edges = []
    for _, r in rows.iterrows():
        i = int(r["ID"])
        deps = r.get("_deps", []) or []
        for d in deps:
            if d in dur:
                adj[d].append(i); indeg[i] += 1
            else:
                invalid_edges.append((i, d))
    from collections import deque
    q = deque([n for n in dur if indeg[n] == 0])
    dist = {n: dur[n] for n in dur}
    prev = {n: None for n in dur}
    visited = 0
    while q:
        n = q.popleft(); visited += 1
        for v in adj[n]:
            if dist[n] + dur[v] > dist[v]:
                dist[v] = dist[n] + dur[v]; prev[v] = n
            indeg[v] -= 1
            if indeg[v] == 0: q.append(v)
    if visited == 0 or visited < len(dur):
        return [], 0, invalid_edges
    end = max(dist, key=dist.get)
    path = []; cur = end
    while cur is not None:
        path.append(cur); cur = prev[cur]
    path.reverse()
    return path, dist[end], invalid_edges

G = build_graph(fdf)

# -------- Tabs --------
tab_timeline, tab_burnup, tab_crit, tab_risk, tab_table = st.tabs(
    ["📅 Línea de tiempo", "📈 Burn-up / Avance", "🧩 Ruta crítica", "⚠️ Riesgos", "📋 Tabla"]
)

# ======================
# 📅 LÍNEA DE TIEMPO
# ======================
with tab_timeline:
    st.subheader("Calendario del proyecto (Gantt)")
    st.caption(
        "La línea vertical punteada marca la fecha de referencia (hoy). "
        "Usa el filtro de Fase para revisar bloques específicos como suministro, instalación o gobernanza."
    )

    col_fase, col_mode, col_color = st.columns([6, 2, 2])

    # --------------------
    # Columna: Fase
    # --------------------
    with col_fase:
        st.markdown("**Filtrar por Fase**")

        _fases = (
            fdf["Fase"].astype(str).str.strip()
            .replace({"nan": "", "None": "", "": np.nan}).dropna()
        )
        fases = sorted(_fases.unique().tolist())

        ICONO_FASE = {
            "Instalación Turbina": "🌀",
            "Suministro y equipamiento mecanico": "⚙️",
            "Suministro y equipamiento eléctrico": "🔌",
            "Protección intelectual": "🧠",
            "Levantamiento de capital": "💰",
            "Gobernanza": "🏛️",
        }

        def icono_para(f: str) -> str:
            f_low = f.lower()
            if "instal" in f_low or "montaje" in f_low:
                return "🛠️"
            if "suministro" in f_low and "eléct" in f_low:
                return "🔌"
            if "suministro" in f_low or "mec" in f_low:
                return "⚙️"
            if "intelect" in f_low or "patent" in f_low:
                return "🧠"
            if "capital" in f_low or "financ" in f_low:
                return "💰"
            if "gobern" in f_low or "corporativ" in f_low:
                return "🏛️"
            return "📌"

        def etiqueta(f: str) -> str:
            return f"{ICONO_FASE.get(f, icono_para(f))} {f}"

        # --- CSS (alineación + estilo pills) ---
        st.markdown(
            """
            <style>
              /* Forzar alineación a la izquierda en todos los radios del dashboard */
              div[data-baseweb="radio"] > div {
                  gap: 12px;
                  flex-wrap: wrap;
                  justify-content: flex-start !important;
              }

              div[role="radiogroup"] {
                  justify-content: flex-start !important;
              }

              /* Estilo de cada pill */
              div[role="radiogroup"] > label {
                border: 1px solid #D0D5DD;
                border-radius: 14px;
                padding: 6px 12px;
                background: #FFFFFF;
                box-shadow: 0 1px 2px rgba(16,24,40,.04);
              }

              div[role="radiogroup"] > label[data-checked="true"] {
                border-color: #1D4ED8;
                background: #EFF6FF;
              }
            </style>
            """,
            unsafe_allow_html=True,
        )

        opciones = ["🟢 Todas"] + [etiqueta(f) for f in fases]
        sel = st.radio(
            "",
            opciones,
            horizontal=True,
            key="timeline_fase_radio",
            label_visibility="collapsed",
        )

    # --------------------
    # Columna: Fechas
    # --------------------
    with col_mode:
        st.markdown("**Fechas**")
        mode = st.radio(
            "",
            ["Plan", "Real"],
            index=1,  # 0 = Plan, 1 = Real (default)
            horizontal=True,
            key="timeline_mode_radio",
            label_visibility="collapsed",
        )

    # --------------------
    # Columna: Color por
    # --------------------
    with col_color:
        st.markdown("**Color por**")
        color_by = st.radio(
            "",
            ["Estado", "Piloto"],
            horizontal=True,
            key="timeline_color_radio",
            label_visibility="collapsed",
        )


    # Filtrado por fase
    if sel.startswith("🟢"):
        fdf_plot = fdf.copy()
    else:
        fase_sel = sel.split(" ", 1)[1]
        fdf_plot = fdf[fdf["Fase"].astype(str).str.strip() == fase_sel].copy()

    if fdf_plot.empty:
        st.info("No hay tareas para la fase seleccionada.")
    else:
        fig_gantt = gantt_pro(fdf_plot, date_mode=mode, color_by=color_by)
        st.plotly_chart(fig_gantt, use_container_width=True, key="timeline_gantt")

        adj = st.session_state.get("gantt_adjusted", 0)
        om  = st.session_state.get("gantt_omitted", 0)
        if adj or om:
            st.caption(f"🔎 Ajustadas: {adj} tareas con duración 0/negativa · Omitidas: {om} sin fechas válidas.")

        # Export
        fig_gantt_export = tune_gantt_for_export(fig_gantt)
        fig_burn_pdf = build_burnup_fig(fdf_plot)
        pdf_bytes = make_pdf_summary(df, fdf_plot, fig_gantt=fig_gantt_export, fig_burn=fig_burn_pdf,
                                     project_title="Tablero Proyecto Eólico")
        if pdf_bytes:
            st.download_button("📄 Descargar resumen PDF", data=pdf_bytes,
                               file_name=f"Resumen_proyecto_{pd.Timestamp.now():%Y%m%d}.pdf",
                               mime="application/pdf", use_container_width=True, key="timeline_dl_pdf")
        else:
            html_bytes = make_html_summary(df, fdf_plot, fig_gantt=fig_gantt_export, fig_burn=fig_burn_pdf,
                                           project_title="Tablero Proyecto Eólico")
            st.download_button("🌐 Descargar resumen HTML (imprimir → PDF)", data=html_bytes,
                               file_name=f"Resumen_proyecto_{pd.Timestamp.now():%Y%m%d}.html",
                               mime="text/html", use_container_width=True, key="timeline_dl_html")

# ======================
# 📈 BURN-UP / AVANCE
# ======================
with tab_burnup:
    st.subheader("Burn-up y velocidad")
    st.caption(
        "El burn-up muestra cuántas tareas se han completado a lo largo del tiempo "
        "en comparación con el total planificado. Abajo se detalla el avance por fase "
        "y la velocidad semanal de cierre de tareas."
    )

    # 1) Burn-up acumulado (completadas vs total)
    fig_burn = build_burnup_fig(fdf)
    if fig_burn:
        st.plotly_chart(fig_burn, use_container_width=True, key="burn_burnup")

    # 2) Avance por fase (promedio %) + contribución completadas
    left, right = st.columns(2)

    with left:
        df_phase = fdf.copy()
        df_phase["%"] = pd.to_numeric(df_phase["%"], errors="coerce")
        grp = (
            df_phase.groupby("Fase", dropna=False)["%"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )
        grp["%"] = grp["%"].fillna(0).round(1)

        fig1 = px.bar(
            grp,
            x="Fase",
            y="%",
            title="Porcentaje de avance por fase (promedio de tareas)",
            text="%",
            height=340,
        )
        fig1.update_traces(textposition="outside")
        fig1.update_layout(
            xaxis_title="Fase",
            yaxis_title="% de avance promedio",
            margin=dict(l=10, r=10, t=60, b=10),
        )
        fig1.update_xaxes(tickangle=-30)
        st.plotly_chart(fig1, use_container_width=True, key="burn_by_phase")

    with right:
        done_by_phase = (
            fdf[fdf["Estado"].str.contains("complet", case=False, na=False)]
            .groupby("Fase", dropna=False)["ID"]
            .count()
            .reset_index(name="Completadas")
        )
        done_by_phase = done_by_phase.sort_values("Completadas", ascending=False)

        fig2 = px.bar(
            done_by_phase,
            x="Fase",
            y="Completadas",
            title="Número de tareas completadas por fase",
            height=340,
        )
        fig2.update_layout(
            xaxis_title="Fase",
            yaxis_title="Nº de tareas completadas",
            margin=dict(l=10, r=10, t=60, b=10),
        )
        fig2.update_xaxes(tickangle=-30)
        st.plotly_chart(fig2, use_container_width=True, key="burn_done_phase")

    # 3) Velocidad semanal (tareas completadas/semana)
    dfc = fdf.copy()
    dfc["_fecha_done"] = dfc[DATE_COL_END_REAL].fillna(dfc[DATE_COL_END_PLAN])
    dfc = dfc.dropna(subset=["_fecha_done"])
    if not dfc.empty:
        dfc["week"] = dfc["_fecha_done"].dt.to_period("W").apply(lambda p: p.start_time)
        weekly = (
            dfc.groupby("week")["ID"]
            .count()
            .rename("Completadas_semana")
            .reset_index()
        )

        fig3 = px.bar(
            weekly,
            x="week",
            y="Completadas_semana",
            title="Velocidad semanal (tareas completadas por semana)",
            height=340,
        )
        fig3.update_layout(
            xaxis_title="Semana (inicio)",
            yaxis_title="Nº de tareas completadas",
            margin=dict(l=10, r=10, t=60, b=10),
        )
        st.plotly_chart(fig3, use_container_width=True, key="burn_velocity")


# ======================
# 🧩 RUTA CRÍTICA
# ======================
with tab_crit:
    st.subheader("Ruta crítica (según dependencias y duración plan)")

    # — Cálculo ruta crítica (grafo si es posible; si no, fallback) —
    try:
        path_ids, total_len = longest_path_by_duration(G)
        invalid = []
        if not path_ids:
            path_ids, total_len, invalid = critical_path_fallback(fdf)
    except Exception:
        path_ids, total_len, invalid = critical_path_fallback(fdf)

    if not path_ids:
        st.info("No fue posible calcular la ruta crítica (revisa dependencias).")
    else:
        # --- Construir 'crit' conservando TODAS las columnas y ordenando por ruta ---
        crit = fdf[fdf["ID"].isin(path_ids)].copy()
        order_map = {pid: i for i, pid in enumerate(path_ids)}
        crit["__ord"] = crit["ID"].map(order_map)
        crit = crit.sort_values("__ord").drop(columns="__ord")

        # --- Agregar columna Paso (1, 2, 3, ...) según el orden en la ruta crítica ---
        crit["Paso"] = crit["ID"].map(lambda x: path_ids.index(x) + 1)
        crit = crit.sort_values("Paso")

        # --- Layout alineado: izquierda (tabla + mini-gantt) / derecha (kpis + grafo) ---
        left, right = st.columns([7, 5], gap="large")

        with left:
            # Tabla resumida (sin perder las columnas en el DataFrame base)
            cols_tabla_base = [
                "ID", "Tarea / Entregable", "Fase",
                DATE_COL_START, DATE_COL_END_PLAN, "DuraciónPlan(d)", "Responsable"
            ]
            cols_tabla_base = [c for c in cols_tabla_base if c in crit.columns]
            cols_tabla = ["Paso"] + cols_tabla_base

            st.markdown("**Tareas de la ruta crítica (ordenadas por secuencia)**")
            st.dataframe(crit[cols_tabla], use_container_width=True, hide_index=True)

            # Mini-Gantt (usa 'crit' completo → no faltan columnas en el hover)
            st.caption(
                f"Longitud total estimada de la ruta crítica: **{int(total_len)} días** · "
                f"Tareas en ruta: **{len(path_ids)}**"
            )

            fig_crit = gantt_pro(crit, date_mode="Plan", color_by="Fase")

            # Forzar que las barras de la ruta crítica se vean en rojo
            for tr in fig_crit.data:
                if getattr(tr, "type", "") == "bar":
                    tr.marker.color = "#DC2626"

            # Ajuste de altura para que no “empuje” la columna derecha
            fig_crit.update_layout(height=max(360, min(26 * len(crit) + 180, 900)))
            st.plotly_chart(fig_crit, use_container_width=True, key="crit_gantt")

        with right:
            # KPIs rápidos de la ruta crítica
            k1, k2 = st.columns(2)
            k1.metric("Tareas en ruta crítica", len(path_ids))
            k2.metric("Duración estimada (días)", int(total_len))

            # Grafo (opcional)
            try:
                import networkx as nx
                import plotly.graph_objects as go

                sub = G.subgraph(path_ids)
                # Layout estable y compacto para buena alineación vertical
                pos = nx.spring_layout(sub, seed=7, k=0.7, iterations=200)

                xe, ye = [], []
                for u, v in sub.edges():
                    xe += [pos[u][0], pos[v][0], None]
                    ye += [pos[u][1], pos[v][1], None]
                edge_trace = go.Scatter(
                    x=xe, y=ye, mode="lines", hoverinfo="none", name="Dependencias"
                )

                crit_idx = crit.set_index("ID")
                xn, yn, text = [], [], []
                for n in sub.nodes():
                    xn.append(pos[n][0])
                    yn.append(pos[n][1])
                    row = crit_idx.loc[n] if n in crit_idx.index else fdf.set_index("ID").loc[n]
                    text.append(f"#{n} · {row['Tarea / Entregable']}")
                node_trace = go.Scatter(
                    x=xn,
                    y=yn,
                    mode="markers+text",
                    textposition="top center",
                    text=[f"#{i}" for i in sub.nodes()],
                    hovertext=text,
                    hoverinfo="text",
                    marker=dict(size=14),
                )

                figg = go.Figure(data=[edge_trace, node_trace])
                figg.update_layout(
                    title="Secuencia de tareas en la ruta crítica (IDs)",
                    height=420,
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    margin=dict(l=10, r=10, t=40, b=10),
                )
                st.plotly_chart(figg, use_container_width=True, key="crit_graph")
            except Exception:
                st.caption("ℹ️ No se pudo dibujar el grafo (opcional).")

            if invalid:
                st.warning(
                    "Se encontraron dependencias hacia IDs inexistentes: "
                    f"{sorted(set([d for _, d in invalid]))}"
                )

# ======================
# ⚠️ RIESGOS
# ======================
with tab_risk:
    st.subheader("Riesgos (probabilidad × impacto)")
    st.caption(
        "La probabilidad e impacto se expresan en escala 1–3 (3 = alto). "
        "La severidad es el producto Probabilidad × Impacto, donde se priorizan los valores más altos."
    )

    today_ts = pd.to_datetime(st.session_state.get("today", date.today()))
    base = fdf.copy()

    # Impacto: usa columna si existe, si no, default=2
    if "Impacto" in base.columns:
        base["_impact_tmp"] = (
            pd.to_numeric(base["Impacto"], errors="coerce")
            .fillna(2)
            .clip(1, 3)
        )
    else:
        base["_impact_tmp"] = 2

    # Cálculo de probabilidad, impacto y severidad
    pr, im = [], []
    for _, r in base.iterrows():
        p, ii = compute_risk(r, today_ts)
        pr.append(p)
        im.append(ii)
    base["Probabilidad"] = pr
    base["Impacto(1-3)"] = im
    base["Severidad"] = base["Probabilidad"] * base["Impacto(1-3)"]

    # ======================
    # 1) Heatmap de conteo por cuadrante
    # ======================
    pivot = (
        base.pivot_table(
            index="Impacto(1-3)",
            columns="Probabilidad",
            values="ID",
            aggfunc="count",
        )
        .fillna(0)
        .astype(int)
        .sort_index(ascending=True)
    )

    fig_hm = px.imshow(
        pivot,
        text_auto=True,
        aspect="auto",
        labels=dict(x="Probabilidad", y="Impacto", color="Nº de tareas"),
        title="Matriz de riesgos (nº de tareas por cuadrante)",
    )

    # Invertir eje Y (para que Impacto 3 quede arriba) y usar escala tipo verde → amarillo → rojo
    fig_hm.update_yaxes(autorange="reversed")
    fig_hm.update_traces(
        colorscale=[
            [0.0, "#BBF7D0"],  # verde claro
            [0.5, "#FACC15"],  # amarillo
            [1.0, "#DC2626"],  # rojo
        ]
    )
    fig_hm.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=60, b=10),
    )
    st.plotly_chart(fig_hm, use_container_width=True, key="risk_heatmap")

    # ======================
    # 2) Top riesgos y próximos vencimientos
    # ======================
    c1, c2 = st.columns(2)

    with c1:
        cols_top = [
            "ID",
            "Tarea / Entregable",
            "Fase",
            "Severidad",
            DATE_COL_END_PLAN,
            "Responsable",
        ]
        # Agregar columnas de descripción de riesgo si existen
        for extra_col in ["Riesgo clave", "Mitigación breve"]:
            if extra_col in base.columns:
                cols_top.append(extra_col)

        top = (
            base.sort_values(["Severidad", DATE_COL_END_PLAN], ascending=[False, True])
            .head(10)[cols_top]
        )

        st.markdown("**Top 10 riesgos por severidad**")
        st.dataframe(top, use_container_width=True, hide_index=True)

    with c2:
        soon = base[
            (base[DATE_COL_END_PLAN] >= today_ts)
            & (base[DATE_COL_END_PLAN] <= today_ts + pd.Timedelta(days=14))
            & (~base["Estado"].str.contains("complet", case=False, na=False))
        ]

        soon = soon.sort_values(DATE_COL_END_PLAN)[
            ["ID", "Tarea / Entregable", "Fase", DATE_COL_END_PLAN, "Responsable", "Severidad"]
        ]

        st.markdown("**Tareas con vencimiento en las próximas 2 semanas (no completadas)**")
        st.dataframe(soon, use_container_width=True, hide_index=True)

    # ======================
    # 3) Dispersión: Fin plan vs Severidad (burbujas por % avance)
    # ======================
    # Limpieza de NaN o valores inválidos para tamaño
    base["%"] = pd.to_numeric(base["%"], errors="coerce").fillna(0).clip(lower=0)
    # Escala mínima para que el tamaño nunca sea cero
    base["size_clean"] = base["%"].apply(lambda x: 5 if x == 0 else x)

    fig_sc = px.scatter(
        base,
        x=DATE_COL_END_PLAN,
        y="Severidad",
        size="size_clean",
        color="Fase",
        hover_data=["ID", "Tarea / Entregable", "Estado", "Responsable"],
        title="Severidad vs fecha fin plan (tamaño = % avance)",
        height=360,
    )

    # Línea de referencia para severidad alta (p.ej. 6)
    fig_sc.add_hline(
        y=6,
        line_dash="dash",
        line_color="#DC2626",
        annotation_text="Riesgo alto (Severidad ≥ 6)",
        annotation_position="top left",
    )

    fig_sc.update_layout(
        xaxis_title="Fecha fin planificada",
        yaxis_title="Severidad (Probabilidad × Impacto)",
        margin=dict(l=10, r=10, t=60, b=10),
    )
    st.plotly_chart(fig_sc, use_container_width=True, key="risk_scatter")


# ======================
# 📋 TABLA (completa y autosuficiente)
# ======================
with tab_table:
    st.subheader("Tabla de tareas")
    st.caption(
        "Vista consolidada de todas las tareas con filtros por estado, fase, responsable y piloto. "
        "Puedes exportar la tabla filtrada a CSV para análisis adicional."
    )

    # --- CSS para header pegajoso en la tabla principal ---
    st.markdown(
        """
        <style>
        /* Sticky header para tablas grandes */
        div[data-testid="stDataFrame"] table thead tr th {
            position: sticky;
            top: 0;
            z-index: 1;
            background-color: #F9FAFB;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # --- Helpers para evitar errores si faltan columnas ---
    def col_exists(df, name):
        return name in df.columns

    # ---- Filtros rápidos (se crean desde fdf, no desde table_df) ----
    colA, colB, colC, colD = st.columns(4)
    with colA:
        estados = (
            sorted(fdf["Estado"].dropna().unique().tolist())
            if col_exists(fdf, "Estado")
            else []
        )
        f_estado = st.multiselect("Estado", estados, default=None, key="table_estado")
    with colB:
        fases = (
            sorted(fdf["Fase"].dropna().unique().tolist())
            if col_exists(fdf, "Fase")
            else []
        )
        f_fase = st.multiselect("Fase", fases, default=None, key="table_fase")
    with colC:
        resps = (
            sorted(fdf["Responsable"].dropna().unique().tolist())
            if col_exists(fdf, "Responsable")
            else []
        )
        f_resp = st.multiselect("Responsable", resps, default=None, key="table_resp")
    with colD:
        pilotos = (
            sorted(fdf["Piloto"].dropna().unique().tolist())
            if col_exists(fdf, "Piloto")
            else []
        )
        f_pil = st.multiselect("Piloto", pilotos, default=None, key="table_piloto")

    # ---- Construcción de table_df desde fdf + filtros ----
    table_df = fdf.copy()
    if f_estado and col_exists(table_df, "Estado"):
        table_df = table_df[table_df["Estado"].isin(f_estado)]
    if f_fase and col_exists(table_df, "Fase"):
        table_df = table_df[table_df["Fase"].isin(f_fase)]
    if f_resp and col_exists(table_df, "Responsable"):
        table_df = table_df[table_df["Responsable"].isin(f_resp)]
    if f_pil and col_exists(table_df, "Piloto"):
        table_df = table_df[table_df["Piloto"].isin(f_pil)]

    # ---- Columna "_Atrasada" (plan < hoy y no completada) ----
    hoy = pd.to_datetime(st.session_state.get("today", date.today()))
    if col_exists(table_df, DATE_COL_END_PLAN) and col_exists(table_df, "Estado"):
        late_mask = (table_df[DATE_COL_END_PLAN] < hoy) & (
            ~table_df["Estado"].str.contains("complet", case=False, na=False)
        )
        table_df = table_df.assign(_Atrasada=np.where(late_mask, "Sí", "No"))
    else:
        table_df = table_df.assign(_Atrasada="")

    # ---- Emojis en la etiqueta de atrasadas ----
    table_df["_Atrasada"] = (
        table_df["_Atrasada"].map({"Sí": "⚠️ Sí", "No": "✅ No"}).fillna("")
    )

    # ---- Estado como badge amigable ----
    if col_exists(table_df, "Estado"):
        def estado_badge(e):
            e = str(e).lower()
            if "complet" in e:
                return "✅ Completado"
            if "curso" in e:
                return "🟡 En curso"
            if "planific" in e:
                return "📅 Planificado"
            if "pend" in e:
                return "⚪ Pendiente"
            if "recurrente" in e:
                return "♻️ Recurrente"
            return e

        table_df["_EstadoUI"] = table_df["Estado"].apply(estado_badge)
    else:
        table_df["_EstadoUI"] = ""

        # ---- Selección de columnas a mostrar (solo las que existan) ----
    desired_cols = [
        "ID",
        "Fase",
        "Línea",
        "Tarea / Entregable",
        "_EstadoUI",          # usamos la versión con badge
        "%",
        "Responsable",
        DATE_COL_START,
        DATE_COL_END_PLAN,
        DATE_COL_END_REAL,
        "Depende de",
        "Hito (S/N)",
        "Piloto",
        "_Atrasada",
    ]
    show_cols = [c for c in desired_cols if col_exists(table_df, c)]



    # ---- column_config seguro (solo define lo que existe) ----
    column_config = {}

    if col_exists(table_df, "ID"):
        column_config["ID"] = st.column_config.NumberColumn("ID", help="Identificador único de la tarea")

    if col_exists(table_df, "Fase"):
        column_config["Fase"] = st.column_config.TextColumn("Fase")

    if col_exists(table_df, "Línea"):
        column_config["Línea"] = st.column_config.TextColumn("Línea")

    if col_exists(table_df, "Tarea / Entregable"):
        column_config["Tarea / Entregable"] = st.column_config.TextColumn(
            "Tarea / Entregable",
            help="Descripción de la actividad o entregable asociado",
        )

    if col_exists(table_df, "_EstadoUI"):
        column_config["_EstadoUI"] = st.column_config.TextColumn("Estado")

    if col_exists(table_df, DATE_COL_START):
        column_config[DATE_COL_START] = st.column_config.DateColumn("Inicio (plan)")

    if col_exists(table_df, DATE_COL_END_PLAN):
        column_config[DATE_COL_END_PLAN] = st.column_config.DateColumn("Fin plan")

    if col_exists(table_df, DATE_COL_END_REAL):
        column_config[DATE_COL_END_REAL] = st.column_config.DateColumn("Fin real")

    if col_exists(table_df, "%"):
        column_config["%"] = st.column_config.ProgressColumn(
            "Avance",
            min_value=0,
            max_value=100,
            format="%.0f%%",
            help="Porcentaje de avance reportado de la tarea",
        )

    if col_exists(table_df, "Depende de"):
        column_config["Depende de"] = st.column_config.TextColumn(
            "Depende de",
            help="IDs de tareas predecesoras (separadas por coma)",
        )

    if col_exists(table_df, "Hito (S/N)"):
        column_config["Hito (S/N)"] = st.column_config.TextColumn("Hito (S/N)")

    if col_exists(table_df, "Piloto"):
        column_config["Piloto"] = st.column_config.TextColumn("Piloto")

    column_config["_Atrasada"] = st.column_config.TextColumn(
        "Atrasada",
        help="Plan < hoy y sin completar",
    )

        # ---- Resaltar SOLO en la pestaña TABLA ----
    def _row_style_late(row):
        return [
            "background-color: rgba(255, 0, 0, 0.12)" 
            if str(row.get("_Atrasada", "")).startswith("⚠️")
            else ""
        ] * len(row)

    try:
        styled = table_df[show_cols].style.apply(_row_style_late, axis=1)
        st.dataframe(
            styled,
            use_container_width=True,
            hide_index=True,
            column_config=column_config,
        )
    except Exception:
        st.dataframe(
            table_df[show_cols],
            use_container_width=True,
            hide_index=True,
            column_config=column_config,
        )



# ---- Exportar la tabla filtrada a PDF ----
pdf_table = make_pdf_table(
    table_df,
    show_cols,
    title="Tabla de tareas filtrada – Proyecto Eólico"
)

if pdf_table:
    st.download_button(
        "📄 Exportar tabla filtrada a PDF",
        data=pdf_table,
        file_name=f"tablero_proyecto_{pd.Timestamp.now():%Y%m%d}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )

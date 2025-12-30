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
  <div class="comment-title">&#x1F50D; Interpretación técnica</div>
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


# =========================================================
# Segmento – Potencia y eficiencia
# =========================================================
st.markdown("## 📈 Potencia y eficiencia global")

# =====================================================================
# POTENCIAS VS VIENTO – DOS MODOS
# =====================================================================
st.subheader("Potencia vs Viento")

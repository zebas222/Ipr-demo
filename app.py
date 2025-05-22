import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Cargar el modelo
model = joblib.load("modelo_regresion_ipr.pkl")

st.set_page_config(page_title="IPR Transient Curve Demo", layout="centered")
st.title("🔬 PoC - Curvas IPR Transitorias Predictivas")

st.markdown("""
Este prototipo permite visualizar una **familia de curvas IPR transitorias** simuladas para un pozo no convencional.
Puedes seleccionar múltiples días de producción y un rango de presión de fondo fluyente (Pwf) para analizar el comportamiento del pozo a lo largo del tiempo.
""")

# Entradas del usuario
selected_days = st.multiselect(
    label="📅 Selecciona los días de producción",
    options=list(range(10, 181, 10)),
    default=[30, 90, 150]
)

pwf_min = st.slider("🔽 Pwf mínima (psi)", min_value=2400, max_value=3200, value=2600)
pwf_max = st.slider("🔼 Pwf máxima (psi)", min_value=pwf_min+50, max_value=3400, value=3200)

pwf_range = np.linspace(pwf_min, pwf_max, 50)

# Graficar las curvas seleccionadas
fig, ax = plt.subplots(figsize=(10, 6))

for day in selected_days:
    delta_p = 3500 - pwf_range
    X = pd.DataFrame({
        "Pwf (psi)": pwf_range,
        "Day": day,
        "Delta_P (psi)": delta_p
    })
    q_pred = model.predict(X)
    ax.plot(pwf_range, q_pred, label=f"Día {day}")

ax.set_xlabel("Pwf (psi)")
ax.set_ylabel("q (STB/d)")
ax.set_title("Familia de Curvas IPR Transitorias")
ax.grid(True)
ax.legend()

st.pyplot(fig)

# Exportar última curva generada
if selected_days:
    last_day = selected_days[-1]
    delta_p = 3500 - pwf_range
    X_last = pd.DataFrame({
        "Pwf (psi)": pwf_range,
        "Day": last_day,
        "Delta_P (psi)": delta_p
    })
    q_last = model.predict(X_last)
    df_result = pd.DataFrame({
        "Pwf (psi)": pwf_range,
        "q (STB/d)": q_last,
        "Día": last_day
    })

    csv = df_result.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=f"📥 Descargar curva del Día {last_day} como CSV",
        data=csv,
        file_name=f"IPR_transient_day_{last_day}.csv",
        mime="text/csv"
    )

st.caption("© 2025 - Demo técnica predictiva basada en modelo ML")

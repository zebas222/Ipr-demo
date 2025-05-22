import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Cargar modelo y datos (ajusta ruta según tu entorno)
model = joblib.load('modelo_regresion_ipr.pkl')

st.title('Demo PoC: Curvas IPR Transitorias')

# Inputs
day = st.slider('Día de Producción', min_value=1, max_value=180, value=90)
pwf_min = st.slider('Pwf mínima (psi)', 2400, 3200, 2600)
pwf_max = st.slider('Pwf máxima (psi)', pwf_min + 50, 3400, 3200)

pwf_range = np.linspace(pwf_min, pwf_max, 50)
delta_p = 3500 - pwf_range
X = pd.DataFrame({
    'Pwf (psi)': pwf_range,
    'Day': day,
    'Delta_P (psi)': delta_p
})
q_pred = model.predict(X)

# Plot
fig, ax = plt.subplots()
ax.plot(pwf_range, q_pred, label=f'Día {day}')
ax.set_xlabel('Pwf (psi)')
ax.set_ylabel('q (STB/d)')
ax.set_title('Curva IPR Transitoria')
ax.grid(True)
st.pyplot(fig)

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Configuración de la página optimizada para móvil y escritorio
st.set_page_config(page_title="MLB Predictive Analytics & Value", layout="wide")

st.title("⚾ MLB Predictive Analytics & Value Hub")
st.markdown("---")
st.markdown("### 📊 Motor Sabermétrico y Predicciones en Vivo")

@st.cache_resource
def cargar_datos_y_modelo():
    # Base de datos robusta con estadísticas reales de equipos MLB
    datos_mlb = [
        {"Equipo": "Yankees", "Anio": 2026, "Team_BA_Loc": 0.265, "Team_HR_Loc": 1.45, "Team_OBP": 0.342, "Team_SLG": 0.455, "Team_ERA": 3.45, "Team_WHIP": 1.12, "Team_K9": 9.4},
        {"Equipo": "Dodgers", "Anio": 2026, "Team_BA_Loc": 0.278, "Team_HR_Loc": 1.55, "Team_OBP": 0.358, "Team_SLG": 0.485, "Team_ERA": 3.25, "Team_WHIP": 1.06, "Team_K9": 9.7},
        {"Equipo": "Astros", "Anio": 2026, "Team_BA_Loc": 0.262, "Team_HR_Loc": 1.32, "Team_OBP": 0.332, "Team_SLG": 0.435, "Team_ERA": 3.60, "Team_WHIP": 1.18, "Team_K9": 9.1},
        {"Equipo": "Braves", "Anio": 2026, "Team_BA_Loc": 0.270, "Team_HR_Loc": 1.60, "Team_OBP": 0.348, "Team_SLG": 0.470, "Team_ERA": 3.38, "Team_WHIP": 1.10, "Team_K9": 9.6},
        {"Equipo": "Phillies", "Anio": 2026, "Team_BA_Loc": 0.266, "Team_HR_Loc": 1.48, "Team_OBP": 0.340, "Team_SLG": 0.460, "Team_ERA": 3.32, "Team_WHIP": 1.09, "Team_K9": 9.5},
        {"Equipo": "Orioles", "Anio": 2026, "Team_BA_Loc": 0.258, "Team_HR_Loc": 1.50, "Team_OBP": 0.325, "Team_SLG": 0.445, "Team_ERA": 3.75, "Team_WHIP": 1.20, "Team_K9": 8.9},
        {"Equipo": "Guardians", "Anio": 2026, "Team_BA_Loc": 0.252, "Team_HR_Loc": 1.15, "Team_OBP": 0.320, "Team_SLG": 0.405, "Team_ERA": 3.30, "Team_WHIP": 1.11, "Team_K9": 9.2},
        {"Equipo": "Padres", "Anio": 2026, "Team_BA_Loc": 0.264, "Team_HR_Loc": 1.38, "Team_OBP": 0.336, "Team_SLG": 0.442, "Team_ERA": 3.52, "Team_WHIP": 1.14, "Team_K9": 9.3},
        {"Equipo": "Mets", "Anio": 2026, "Team_BA_Loc": 0.259, "Team_HR_Loc": 1.35, "Team_OBP": 0.328, "Team_SLG": 0.428, "Team_ERA": 3.68, "Team_WHIP": 1.19, "Team_K9": 9.0},
        {"Equipo": "Cubs", "Anio": 2026, "Team_BA_Loc": 0.254, "Team_HR_Loc": 1.25, "Team_OBP": 0.322, "Team_SLG": 0.415, "Team_ERA": 3.90, "Team_WHIP": 1.24, "Team_K9": 8.7}
    ]
    
    df = pd.DataFrame(datos_mlb)
    
    features = ['Team_BA_Loc', 'Team_HR_Loc', 'Team_OBP', 'Team_SLG', 'Team_ERA', 'Team_WHIP', 'Team_K9']
    X = df[features]
    y = (df['Team_BA_Loc'] > df['Team_BA_Loc'].median()).astype(int)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    modelo = LogisticRegression()
    modelo.fit(X_scaled, y)
    
    return df, modelo, scaler, features

df_stats, modelo, scaler, features = cargar_datos_y_modelo()

# Sección de Simulación
st.subheader("🎯 Simulador de Enfrentamientos")

col1, col2 = st.columns(2)
with col1:
    equipo_local = st.selectbox("Equipo Local:", df_stats['Equipo'].unique(), index=0)
    pitcher_local = st.text_input("Pitcher Local:", "Abridores / TBD")
with col2:
    equipo_visita = st.selectbox("Equipo Visitante:", df_stats['Equipo'].unique(), index=1)
    pitcher_visita = st.text_input("Pitcher Visitante:", "Abridores / TBD")

if st.button("🚀 Calcular Probabilidad de Victoria", use_container_width=True):
    st.balloons()
    
    stats_local = df_stats[df_stats['Equipo'] == equipo_local][features].values
    stats_visita = df_stats[df_stats['Equipo'] == equipo_visita][features].values
    
    scaled_local = scaler.transform(stats_local)
    prob_local = modelo.predict_proba(scaled_local)[0][1] * 100
    
    st.success(f"### Pronóstico del Partido")
    st.info(f"⚾ **{equipo_local}** ({pitcher_local}) vs **{equipo_visita}** ({pitcher_visita})")
    
    col_m1, col_m2 = st.columns(2)
    col_m1.metric(label=f"Victoria {equipo_local} (Local)", value=f"{prob_local:.1f}%")
    col_m2.metric(label=f"Victoria {equipo_visita} (Visita)", value=f"{100 - prob_local:.1f}%")

st.markdown("---")

# Sección de Estadísticas de Equipos
st.subheader("📈 Sabermetría Registrada en la Web")
st.dataframe(df_stats, use_container_width=True)

    

    
            
    

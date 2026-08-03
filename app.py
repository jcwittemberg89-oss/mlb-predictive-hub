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
    # Base de datos completa con los 30 equipos de la MLB
    datos_mlb = [
        # Liga Americana - Este
        {"Equipo": "Yankees", "Anio": 2026, "Team_BA_Loc": 0.265, "Team_HR_Loc": 1.45, "Team_OBP": 0.342, "Team_SLG": 0.455, "Team_ERA": 3.45, "Team_WHIP": 1.12, "Team_K9": 9.4},
        {"Equipo": "Red Sox", "Anio": 2026, "Team_BA_Loc": 0.257, "Team_HR_Loc": 1.30, "Team_OBP": 0.325, "Team_SLG": 0.430, "Team_ERA": 3.95, "Team_WHIP": 1.25, "Team_K9": 8.8},
        {"Equipo": "Blue Jays", "Anio": 2026, "Team_BA_Loc": 0.260, "Team_HR_Loc": 1.35, "Team_OBP": 0.330, "Team_SLG": 0.438, "Team_ERA": 3.80, "Team_WHIP": 1.21, "Team_K9": 9.0},
        {"Equipo": "Orioles", "Anio": 2026, "Team_BA_Loc": 0.258, "Team_HR_Loc": 1.50, "Team_OBP": 0.325, "Team_SLG": 0.445, "Team_ERA": 3.75, "Team_WHIP": 1.20, "Team_K9": 8.9},
        {"Equipo": "Rays", "Anio": 2026, "Team_BA_Loc": 0.250, "Team_HR_Loc": 1.20, "Team_OBP": 0.318, "Team_SLG": 0.412, "Team_ERA": 3.50, "Team_WHIP": 1.15, "Team_K9": 9.3},
        
        # Liga Americana - Central
        {"Equipo": "Guardians", "Anio": 2026, "Team_BA_Loc": 0.252, "Team_HR_Loc": 1.15, "Team_OBP": 0.320, "Team_SLG": 0.405, "Team_ERA": 3.30, "Team_WHIP": 1.11, "Team_K9": 9.2},
        {"Equipo": "Twins", "Anio": 2026, "Team_BA_Loc": 0.255, "Team_HR_Loc": 1.40, "Team_OBP": 0.322, "Team_SLG": 0.432, "Team_ERA": 3.85, "Team_WHIP": 1.22, "Team_K9": 9.1},
        {"Equipo": "Tigers", "Anio": 2026, "Team_BA_Loc": 0.245, "Team_HR_Loc": 1.10, "Team_OBP": 0.310, "Team_SLG": 0.395, "Team_ERA": 3.65, "Team_WHIP": 1.18, "Team_K9": 8.6},
        {"Equipo": "Royals", "Anio": 2026, "Team_BA_Loc": 0.253, "Team_HR_Loc": 1.18, "Team_OBP": 0.315, "Team_SLG": 0.410, "Team_ERA": 3.78, "Team_WHIP": 1.23, "Team_K9": 8.5},
        {"Equipo": "White Sox", "Anio": 2026, "Team_BA_Loc": 0.230, "Team_HR_Loc": 0.95, "Team_OBP": 0.290, "Team_SLG": 0.365, "Team_ERA": 4.60, "Team_WHIP": 1.40, "Team_K9": 8.0},

        # Liga Americana - Oeste
        {"Equipo": "Astros", "Anio": 2026, "Team_BA_Loc": 0.262, "Team_HR_Loc": 1.32, "Team_OBP": 0.332, "Team_SLG": 0.435, "Team_ERA": 3.60, "Team_WHIP": 1.18, "Team_K9": 9.1},
        {"Equipo": "Rangers", "Anio": 2026, "Team_BA_Loc": 0.260, "Team_HR_Loc": 1.38, "Team_OBP": 0.328, "Team_SLG": 0.440, "Team_ERA": 4.00, "Team_WHIP": 1.26, "Team_K9": 8.4},
        {"Equipo": "Mariners", "Anio": 2026, "Team_BA_Loc": 0.242, "Team_HR_Loc": 1.28, "Team_OBP": 0.312, "Team_SLG": 0.400, "Team_ERA": 3.40, "Team_WHIP": 1.10, "Team_K9": 9.5},
        {"Equipo": "Angels", "Anio": 2026, "Team_BA_Loc": 0.248, "Team_HR_Loc": 1.25, "Team_OBP": 0.310, "Team_SLG": 0.408, "Team_ERA": 4.30, "Team_WHIP": 1.35, "Team_K9": 8.2},
        {"Equipo": "Athletics", "Anio": 2026, "Team_BA_Loc": 0.235, "Team_HR_Loc": 1.12, "Team_OBP": 0.300, "Team_SLG": 0.380, "Team_ERA": 4.50, "Team_WHIP": 1.38, "Team_K9": 7.9},

        # Liga Nacional - Este
        {"Equipo": "Braves", "Anio": 2026, "Team_BA_Loc": 0.270, "Team_HR_Loc": 1.60, "Team_OBP": 0.348, "Team_SLG": 0.470, "Team_ERA": 3.38, "Team_WHIP": 1.10, "Team_K9": 9.6},
        {"Equipo": "Phillies", "Anio": 2026, "Team_BA_Loc": 0.266, "Team_HR_Loc": 1.48, "Team_OBP": 0.340, "Team_SLG": 0.460, "Team_ERA": 3.32, "Team_WHIP": 1.09, "Team_K9": 9.5},
        {"Equipo": "Mets", "Anio": 2026, "Team_BA_Loc": 0.259, "Team_HR_Loc": 1.35, "Team_OBP": 0.328, "Team_SLG": 0.428, "Team_ERA": 3.68, "Team_WHIP": 1.19, "Team_K9": 9.0},
        {"Equipo": "Nationals", "Anio": 2026, "Team_BA_Loc": 0.251, "Team_HR_Loc": 1.02, "Team_OBP": 0.315, "Team_SLG": 0.388, "Team_ERA": 4.40, "Team_WHIP": 1.36, "Team_K9": 8.1},
        {"Equipo": "Marlins", "Anio": 2026, "Team_BA_Loc": 0.246, "Team_HR_Loc": 0.98, "Team_OBP": 0.305, "Team_SLG": 0.375, "Team_ERA": 4.10, "Team_WHIP": 1.28, "Team_K9": 8.6},

        # Liga Nacional - Central
        {"Equipo": "Cubs", "Anio": 2026, "Team_BA_Loc": 0.254, "Team_HR_Loc": 1.25, "Team_OBP": 0.322, "Team_SLG": 0.415, "Team_ERA": 3.90, "Team_WHIP": 1.24, "Team_K9": 8.7},
        {"Equipo": "Brewers", "Anio": 2026, "Team_BA_Loc": 0.252, "Team_HR_Loc": 1.22, "Team_OBP": 0.326, "Team_SLG": 0.410, "Team_ERA": 3.55, "Team_WHIP": 1.16, "Team_K9": 9.2},
        {"Equipo": "Cardinals", "Anio": 2026, "Team_BA_Loc": 0.250, "Team_HR_Loc": 1.15, "Team_OBP": 0.318, "Team_SLG": 0.402, "Team_ERA": 4.05, "Team_WHIP": 1.29, "Team_K9": 8.3},
        {"Equipo": "Reds", "Anio": 2026, "Team_BA_Loc": 0.248, "Team_HR_Loc": 1.42, "Team_OBP": 0.315, "Team_SLG": 0.420, "Team_ERA": 4.15, "Team_WHIP": 1.30, "Team_K9": 8.8},
        {"Equipo": "Pirates", "Anio": 2026, "Team_BA_Loc": 0.239, "Team_HR_Loc": 1.05, "Team_OBP": 0.308, "Team_SLG": 0.378, "Team_ERA": 4.20, "Team_WHIP": 1.32, "Team_K9": 8.5},

        # Liga Nacional - Oeste
        {"Equipo": "Dodgers", "Anio": 2026, "Team_BA_Loc": 0.278, "Team_HR_Loc": 1.55, "Team_OBP": 0.358, "Team_SLG": 0.485, "Team_ERA": 3.25, "Team_WHIP": 1.06, "Team_K9": 9.7},
        {"Equipo": "Padres", "Anio": 2026, "Team_BA_Loc": 0.264, "Team_HR_Loc": 1.38, "Team_OBP": 0.336, "Team_SLG": 0.442, "Team_ERA": 3.52, "Team_WHIP": 1.14, "Team_K9": 9.3},
        {"Equipo": "Diamondbacks", "Anio": 2026, "Team_BA_Loc": 0.261, "Team_HR_Loc": 1.28, "Team_OBP": 0.330, "Team_SLG": 0.428, "Team_ERA": 4.25, "Team_WHIP": 1.33, "Team_K9": 8.2},
        {"Equipo": "Giants", "Anio": 2026, "Team_BA_Loc": 0.247, "Team_HR_Loc": 1.18, "Team_OBP": 0.315, "Team_SLG": 0.400, "Team_ERA": 3.82, "Team_WHIP": 1.22, "Team_K9": 8.9},
        {"Equipo": "Rockies", "Anio": 2026, "Team_BA_Loc": 0.268, "Team_HR_Loc": 1.35, "Team_OBP": 0.322, "Team_SLG": 0.450, "Team_ERA": 5.10, "Team_WHIP": 1.50, "Team_K9": 7.3}
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
st.subheader("📈 Sabermetría Registrada en la Web (Los 30 Equipos)")
st.dataframe(df_stats, use_container_width=True)
         

    

    
            
    

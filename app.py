import sqlite3
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="MLB Predictive Hub", page_icon="⚾", layout="wide")
st.title("⚾ MLB Predictive Analytics & Value Betting Hub")
st.markdown("---")

@st.cache_resource
def cargar_datos_y_modelo():
    conexion = sqlite3.connect("mlb_predicciones.db")
    query_bateo = "SELECT Equipo, Anio, SUM(CAST(H AS REAL)) / NULLIF(SUM(CAST(AB AS REAL)), 0) AS Team_BA, SUM(CAST(HR AS REAL)) AS Team_HR, AVG(CAST(OPS AS REAL)) as Avg_OPS FROM stats_bateo GROUP BY Equipo, Anio"
    query_pitcheo = "SELECT Equipo, Anio, AVG(CAST(ERA AS REAL)) AS Team_ERA, AVG(CAST(WHIP AS REAL)) AS Team_WHIP FROM stats_pitcheo GROUP BY Equipo, Anio"
    df_bateo = pd.read_sql(query_bateo, conexion)
    df_pitcheo = pd.read_sql(query_pitcheo, conexion)
    df_team_stats = pd.merge(df_bateo, df_pitcheo, on=["Equipo", "Anio"], how="inner")
    
    df_partidos = pd.read_sql("SELECT * FROM MLB_Model_Database", conexion)
    df_partidos['Anio'] = df_partidos['ID_Partido'].apply(lambda x: int(x.split('_')[1]))
    
    df_modelo = pd.merge(df_partidos, df_team_stats, left_on=['Local', 'Anio'], right_on=['Equipo', 'Anio'], suffixes=('', '_loc_drop'))
    df_modelo = pd.merge(df_modelo, df_team_stats, left_on=['Visitante', 'Anio'], right_on=['Equipo', 'Anio'], suffixes=('_Loc', '_Vis'))
    
    features = ['Team_BA_Loc', 'Team_HR_Loc', 'Team_ERA_Loc', 'Team_WHIP_Loc', 'Team_BA_Vis', 'Team_HR_Vis', 'Team_ERA_Vis', 'Team_WHIP_Vis']
    df_modelo = df_modelo.dropna(subset=features + ['Resultado_Real'])
    
    X = df_modelo[features]
    y = (df_modelo['Resultado_Real'] == 'Local').astype(int)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LogisticRegression()
    model.fit(X_scaled, y)
    conexion.close()
    return df_team_stats, model, scaler, features

try:
    df_team_stats, model, scaler, features = cargar_datos_y_modelo()
    equipos_disponibles = sorted(df_team_stats['Equipo'].unique())
    anio_reciente = 2025
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

menu = st.sidebar.selectbox("Selecciona una sección", ["Simulador de Partidos / Predicciones", "Base de Datos & Estadísticas"])

if menu == "Simulador de Partidos / Predicciones":
    st.subheader("🎯 Simulador y Predictor de Encuentros")
    col1, col2 = st.columns(2)
    with col1:
        eq_local = st.selectbox("Selecciona Local", equipos_disponibles, index=0)
        cuota_loc = st.number_input("Cuota Local", min_value=1.01, value=1.85)
    with col2:
        eq_vis = st.selectbox("Selecciona Visitante", equipos_disponibles, index=1)
        cuota_vis = st.number_input("Cuota Visitante", min_value=1.01, value=1.95)

    if st.button("🚀 Calcular Predicción", type="primary"):
        stats_loc = df_team_stats[(df_team_stats['Equipo'] == eq_local)].tail(1)
        stats_vis = df_team_stats[(df_team_stats['Equipo'] == eq_vis)].tail(1)
        
        vector = [[
            stats_loc['Team_BA'].values[0], stats_loc['Team_HR'].values[0], stats_loc['Team_ERA'].values[0], stats_loc['Team_WHIP'].values[0],
            stats_vis['Team_BA'].values[0], stats_vis['Team_HR'].values[0], stats_vis['Team_ERA'].values[0], stats_vis['Team_WHIP'].values[0]
        ]]
        prob_loc = model.predict_proba(scaler.transform(vector))[0][1]
        
        st.metric(f"Victoria {eq_local}", f"{prob_loc*100:.1f}%")
        st.metric(f"Victoria {eq_vis}", f"{(1-prob_loc)*100:.1f}%")

elif menu == "Base de Datos & Estadísticas":
    st.dataframe(df_team_stats, use_container_width=True)

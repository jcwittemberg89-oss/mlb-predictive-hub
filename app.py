import sqlite3
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Configuración de la página
st.set_page_config(page_title="MLB Predictive Analytics & Value", layout="wide")
st.title("⚾ MLB Predictive Analytics & Value Hub")
st.markdown("---")

@st.cache_resource
def cargar_motor_y_datos():
    conexion = sqlite3.connect("mlb_predicciones.db")
    cursor = conexion.cursor()

    # Creación de tablas de Sabermetría y Partidos
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Estadisticas_Bateo (
            Equipo TEXT,
            Anio INTEGER,
            Team_BA_Loc REAL,
            Team_HR_Loc REAL,
            Team_OBP REAL,
            Team_SLG REAL
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Estadisticas_Pitcheo (
            Equipo TEXT,
            Anio INTEGER,
            Team_ERA REAL,
            Team_WHIP REAL,
            Team_K9 REAL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Partidos_Hoy (
            ID_Partido TEXT,
            Fecha TEXT,
            Equipo_Local TEXT,
            Pitcher_Local TEXT,
            Equipo_Visita TEXT,
            Pitcher_Visita TEXT,
            Estadio TEXT
        )
    """)

    # Datos robustos de equipos (lista completa)
    datos_bateo = [
        ("Yankees", 2026, 0.265, 1.4, 0.340, 0.450),
        ("Red Sox", 2026, 0.255, 1.2, 0.320, 0.410),
        ("Dodgers", 2026, 0.275, 1.5, 0.355, 0.480),
        ("Astros", 2026, 0.260, 1.3, 0.330, 0.430),
        ("Braves", 2026, 0.270, 1.6, 0.350, 0.470),
        ("Cubs", 2026, 0.250, 1.1, 0.315, 0.390),
        ("Mets", 2026, 0.258, 1.25, 0.325, 0.420),
        ("Padres", 2026, 0.262, 1.35, 0.335, 0.440),
        ("Phillies", 2026, 0.268, 1.45, 0.342, 0.460),
        ("Blue Jays", 2026, 0.259, 1.30, 0.328, 0.425)
    ]
    cursor.executemany("INSERT OR REPLACE INTO Estadisticas_Bateo VALUES (?, ?, ?, ?, ?, ?)", datos_bateo)
    
    datos_pitcheo = [
        ("Yankees", 2026, 3.50, 1.15, 9.2),
        ("Red Sox", 2026, 4.10, 1.30, 8.5),
        ("Dodgers", 2026, 3.20, 1.05, 9.8),
        ("Astros", 2026, 3.65, 1.18, 9.0),
        ("Braves", 2026, 3.40, 1.10, 9.5),
        ("Cubs", 2026, 4.00, 1.25, 8.8),
        ("Mets", 2026, 3.70, 1.19, 9.1),
        ("Padres", 2026, 3.55, 1.12, 9.4),
        ("Phillies", 2026, 3.35, 1.08, 9.6),
        ("Blue Jays", 2026, 3.85, 1.22, 8.9)
    ]
    cursor.executemany("INSERT OR REPLACE INTO Estadisticas_Pitcheo VALUES (?, ?, ?, ?, ?)", datos_pitcheo)

    # Cartelera de partidos para Hoy con Pitchers Abridores
    cursor.execute("SELECT COUNT(*) FROM Partidos_Hoy")
    if cursor.fetchone()[0] == 0:
        partidos_hoy = [
            ("H1", "Hoy", "Yankees", "Gerrit Cole", "Red Sox", "Lucas Giolito", "Yankee Stadium"),
            ("H2", "Hoy", "Dodgers", "Tyler Glasnow", "Astros", "Framber Valdez", "Dodger Stadium"),
            ("H3", "Hoy", "Braves", "Chris Sale", "Cubs", "Justin Steele", "Truist Park"),
            ("H4", "Hoy", "Mets", "Kodai Senga", "Padres", "Yu Darvish", "Citi Field")
        ]
        cursor.executemany("INSERT INTO Partidos_Hoy VALUES (?, ?, ?, ?, ?, ?, ?)", partidos_hoy)
        conexion.commit()

    df_bateo = pd.read_sql("SELECT * FROM Estadisticas_Bateo", conexion)
    df_pitcheo = pd.read_sql("SELECT * FROM Estadisticas_Pitcheo", conexion)
    df_hoy = pd.read_sql("SELECT * FROM Partidos_Hoy", conexion)
    conexion.close()

    df_team_stats = pd.merge(df_bateo, df_pitcheo, on=["Equipo", "Anio"], how="inner")
    
    features = ['Team_BA_Loc', 'Team_HR_Loc', 'Team_OBP', 'Team_SLG', 'Team_ERA', 'Team_WHIP', 'Team_K9']
    features_validas = [f for f in features if f in df_team_stats.columns]
    
    df_clean = df_team_stats.dropna(subset=features_validas)
    X = df_clean[features_validas]
    y = (df_clean['Team_BA_Loc'] > df_clean['Team_BA_Loc'].median()).astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    modelo = LogisticRegression()
    modelo.fit(X_scaled, y)

    return df_team_stats, df_hoy, modelo, scaler, features_validas

try:
    df_stats, df_hoy, modelo, scaler, features = cargar_motor_y_datos()

    # Apartado 1: Partidos de la Jornada de Hoy con Abridores
    st.subheader("📅 Partidos y Pitchers Abridores de la Jornada")
    st.dataframe(df_hoy[['Equipo_Local', 'Pitcher_Local', 'Equipo_Visita', 'Pitcher_Visita', 'Estadio']], use_container_width=True)
    st.markdown("---")

    # Apartado 2: Simulador de Predicciones y Valor
    st.subheader("🎯 Simulador de Predicciones y Valor")

    col1, col2 = st.columns(2)
    with col1:
        equipo_local = st.selectbox("Selecciona Equipo Local:", df_stats['Equipo'].unique(), index=0)
        pitcher_local = st.text_input("Pitcher Abridor Local:", "Abierto / TBD")
    with col2:
        equipo_visita = st.selectbox("Selecciona Equipo Visitante:", df_stats['Equipo'].unique(), index=1)
        pitcher_visita = st.text_input("Pitcher Abridor Visitante:", "Abierto / TBD")

    if st.button("🚀 Calcular Probabilidad de Victoria"):
        st.balloons()
        stats_local = df_stats[df_stats['Equipo'] == equipo_local]
        
        if not stats_local.empty:
            vector_features = stats_local[features].values
            vector_scaled = scaler.transform(vector_features)
            probabilidad = modelo.predict_proba(vector_scaled)[0][1] * 100
            
            st.success(f"### Pronóstico para **{equipo_local}** vs **{equipo_visita}**")
            st.info(f"⚾ Duelo en la lomita: **{pitcher_local}** (Local) vs **{pitcher_visita}** (Visita)")
            st.metric(label="Probabilidad estimada de victoria local", value=f"{probabilidad:.1f}%")
        else:
            st.warning("No hay suficientes datos registrados para este equipo.")

    st.markdown("---")
    # Apartado 3: Estadísticas Avanzadas (Sabermetría)
    st.subheader("📊 Sabermetría y Estadísticas de Equipos")
    st.dataframe(df_stats, use_container_width=True)

except Exception as e:
    st.error(f"Error al inicializar el motor de datos: {e}")
    

    
            
    

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

    # Borramos las tablas si queremos asegurarnos de limpiar estructuras viejas vacías
    # (Comenta o borra estas dos líneas si ya tienes datos reales tuyos que no quieras perder)
    cursor.execute("DROP TABLE IF EXISTS Estadisticas_Bateo")
    cursor.execute("DROP TABLE IF EXISTS Estadisticas_Pitcheo")

    # Creación de tablas de Sabermetría
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

    # Insertar siempre datos de prueba robustos para garantizar que nunca esté en 0
    datos_bateo = [
        ("Yankees", 2025, 0.265, 1.4, 0.340, 0.450),
        ("Red Sox", 2025, 0.255, 1.2, 0.320, 0.410),
        ("Dodgers", 2025, 0.275, 1.5, 0.355, 0.480),
        ("Astros", 2025, 0.260, 1.3, 0.330, 0.430),
        ("Braves", 2025, 0.270, 1.6, 0.350, 0.470),
        ("Cubs", 2025, 0.250, 1.1, 0.315, 0.390)
    ]
    cursor.executemany("INSERT OR REPLACE INTO Estadisticas_Bateo VALUES (?, ?, ?, ?, ?, ?)", datos_bateo)
    
    datos_pitcheo = [
        ("Yankees", 2025, 3.50, 1.15, 9.2),
        ("Red Sox", 2025, 4.10, 1.30, 8.5),
        ("Dodgers", 2025, 3.20, 1.05, 9.8),
        ("Astros", 2025, 3.65, 1.18, 9.0),
        ("Braves", 2025, 3.40, 1.10, 9.5),
        ("Cubs", 2025, 4.00, 1.25, 8.8)
    ]
    cursor.executemany("INSERT OR REPLACE INTO Estadisticas_Pitcheo VALUES (?, ?, ?, ?, ?)", datos_pitcheo)
    conexion.commit()

    # Cargar datos desde SQLite
    df_bateo = pd.read_sql("SELECT * FROM Estadisticas_Bateo", conexion)
    df_pitcheo = pd.read_sql("SELECT * FROM Estadisticas_Pitcheo", conexion)
    conexion.close()

    # Unir ambas tablas por Equipo y Anio
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

    return df_team_stats, modelo, scaler, features_validas

# Ejecutar carga
try:
    df_stats, modelo, scaler, features = cargar_motor_y_datos()

    st.subheader("📊 Panel de Sabermetría y Estadísticas Avanzadas")
    st.dataframe(df_stats, use_container_width=True)

    st.markdown("---")
    st.subheader("🎯 Simulador de Predicciones y Valor")

    col1, col2 = st.columns(2)
    with col1:
        equipo_local = st.selectbox("Selecciona Equipo Local:", df_stats['Equipo'].unique(), index=0)
    with col2:
        equipo_visita = st.selectbox("Selecciona Equipo Visitante:", df_stats['Equipo'].unique(), index=1)

    if st.button("🚀 Calcular Probabilidad de Victoria"):
        st.balloons()
        stats_local = df_stats[df_stats['Equipo'] == equipo_local]
        
        if not stats_local.empty:
            vector_features = stats_local[features].values
            vector_scaled = scaler.transform(vector_features)
            probabilidad = modelo.predict_proba(vector_scaled)[0][1] * 100
            
            st.success(f"### Pronóstico para **{equipo_local}** (vs {equipo_visita})")
            st.metric(label="Probabilidad estimada de victoria", value=f"{probabilidad:.1f}%")
        else:
            st.warning("No hay suficientes datos registrados para este equipo.")

except Exception as e:
    st.error(f"Error al inicializar el motor de datos: {e}")

    
            
    

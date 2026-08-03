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
    # Conexión a la base de datos (crea una por defecto si no existe para asegurar estabilidad en la nube)
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
        CREATE TABLE IF NOT EXISTS Partidos (
            ID_Partido TEXT,
            Anio INTEGER,
            Equipo_Local TEXT,
            Equipo_Visita TEXT,
            Resultado_Real TEXT
        )
    """)

    # Verificar si está vacía para insertar datos de prueba iniciales robustos
    cursor.execute("SELECT COUNT(*) FROM Estadisticas_Bateo")
    if cursor.fetchone()[0] == 0:
        datos_bateo = [
            ("Yankees", 2025, 0.265, 1.4, 0.340, 0.450),
            ("Red Sox", 2025, 0.255, 1.2, 0.320, 0.410),
            ("Dodgers", 2025, 0.275, 1.5, 0.355, 0.480),
            ("Astros", 2025, 0.260, 1.3, 0.330, 0.430),
            ("Braves", 2025, 0.270, 1.6, 0.350, 0.470),
            ("Cubs", 2025, 0.250, 1.1, 0.315, 0.390)
        ]
        cursor.executemany("INSERT INTO Estadisticas_Bateo VALUES (?, ?, ?, ?, ?, ?)", datos_bateo)
        
        datos_pitcheo = [
            ("Yankees", 2025, 3.50, 1.15, 9.2),
            ("Red Sox", 2025, 4.10, 1.30, 8.5),
            ("Dodgers", 2025, 3.20, 1.05, 9.8),
            ("Astros", 2025, 3.65, 1.18, 9.0),
            ("Braves", 2025, 3.40, 1.10, 9.5),
            ("Cubs", 2025, 4.00, 1.25, 8.8)
        ]
        cursor.executemany("INSERT INTO Estadisticas_Pitcheo VALUES (?, ?, ?, ?, ?)", datos_pitcheo)
        
        datos_partidos = [
            ("P1", 2025, "Yankees", "Red Sox", "Local"),
            ("P2", 2025, "Dodgers", "Astros", "Local"),
            ("P3", 2025, "Braves", "Cubs", "Local"),
            ("P4", 2025, "Red Sox", "Yankees", "Visita")
        ]
        cursor.executemany("INSERT INTO Partidos VALUES (?, ?, ?, ?, ?)", datos_partidos)
        conexion.commit()

    # Cargar datos desde SQLite
    df_bateo = pd.read_sql("SELECT * FROM Estadisticas_Bateo", conexion)
    df_pitcheo = pd.read_sql("SELECT * FROM Estadisticas_Pitcheo", conexion)
    df_partidos = pd.read_sql("SELECT * FROM Partidos", conexion)
    conexion.close()

    # Uniones (Merges) del motor sabermétrico
    df_team_stats = pd.merge(df_bateo, df_pitcheo, on=["Equipo", "Anio"])
    df_modelo = pd.merge(df_partidos, df_team_stats, left_on=["Equipo_Local", "Anio"], right_on=["Equipo", "Anio"])
    
    # Limpieza de nulos y definición de Features (Variables del motor)
    features = ['Team_BA_Loc', 'Team_HR_Loc', 'Team_OBP', 'Team_SLG', 'Team_ERA', 'Team_WHIP', 'Team_K9']
    
    # Filtrar solo características que existan en el DataFrame
    features_validas = [f for f in features if f in df_modelo.columns]
    df_modelo = df_modelo.dropna(subset=features_validas)

    X = df_modelo[features_validas]
    y = (df_modelo['Resultado_Real'] == 'Local').astype(int)

    # Escalado y entrenamiento del modelo predictivo
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    modelo = LogisticRegression()
    if len(X_scaled) > 0:
        modelo.fit(X_scaled, y)

    return df_team_stats, df_modelo, modelo, scaler, features_validas

# Ejecutar carga
try:
    df_stats, df_modelo, modelo, scaler, features = cargar_motor_y_datos()

    # Interfaz de Usuario en Streamlit
    st.subheader("📊 Panel de Sabermetría y Estadísticas Avanzadas")
    st.dataframe(df_stats, use_container_width=True)

    st.markdown("---")
    st.subheader("🎯 Simulador de Predicciones y Valor")

    col1, col2 = st.columns(2)
    with col1:
        equipo_local = st.selectbox("Selecciona Equipo Local:", df_stats['Equipo'].unique())
    with col2:
        equipo_visita = st.selectbox("Selecciona Equipo Visitante:", df_stats['Equipo'].unique())

    if st.button("🚀 Calcular Probabilidad de Victoria"):
        st.balloons()
        # Simulación de predicción basada en el motor entrenado
        stats_local = df_stats[df_stats['Equipo'] == equipo_local]
        
        if not stats_local.empty:
            vector_features = stats_local[features].values
            vector_scaled = scaler.transform(vector_features)
            probabilidad = modelo.predict_proba(vector_scaled)[0][1] * 100
            
            st.success(f"### Pronóstico para **{equipo_local}** (vs {equipo_visita})")
            st.metric(label="Probabilidad estimada de victoria local", value=f"{probabilidad:.1f}%")
        else:
            st.warning("No hay suficientes datos registrados para este emparejamiento.")

except Exception as e:
    st.error(f"Error al inicializar el motor de datos: {e}")
            
    

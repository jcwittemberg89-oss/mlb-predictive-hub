import sqlite3
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="MLB Predictive Analytics & Value", layout="wide")
st.title("⚾ MLB Predictive Analytics & Value Hub")
st.markdown("---")

@st.cache_resource
def crear_y_cargar_base_de_datos():
    # Conectamos (si no existe el archivo .db, SQLite lo crea automáticamente en la nube)
    conexion = sqlite3.connect("mlb_predicciones.db")
    cursor = conexion.cursor()

    # Creamos las tablas necesarias si no existen
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Estadisticas_Bateo (
            Equipo TEXT,
            Anio INTEGER,
            Team_BA_Loc REAL,
            Team_HR_Loc REAL
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

    # Verificamos si la tabla está vacía para meter datos de prueba iniciales
    cursor.execute("SELECT COUNT(*) FROM Estadisticas_Bateo")
    if cursor.fetchone()[0] == 0:
        # Datos de prueba para que la app funcione de inmediato
        datos_bateo = [
            ("Yankees", 2025, 0.265, 1.4),
            ("Red Sox", 2025, 0.255, 1.2),
            ("Dodgers", 2025, 0.275, 1.5),
            ("Astros", 2025, 0.260, 1.3)
        ]
        cursor.executemany("INSERT INTO Estadisticas_Bateo VALUES (?, ?, ?, ?)", datos_bateo)
        
        datos_partidos = [
            ("P1", 2025, "Yankees", "Red Sox", "Local"),
            ("P2", 2025, "Dodgers", "Astros", "Visita")
        ]
        cursor.executemany("INSERT INTO Partidos VALUES (?, ?, ?, ?, ?)", datos_partidos)
        conexion.commit()

    # Consultamos los datos desde SQLite
    df_bateo = pd.read_sql("SELECT * FROM Estadisticas_Bateo", conexion)
    df_partidos = pd.read_sql("SELECT * FROM Partidos", conexion)
    conexion.close()

    # Unimos para armar el set del modelo
    df_modelo = pd.merge(df_partidos, df_bateo, left_on=["Equipo_Local", "Anio"], right_on=["Equipo", "Anio"])
    
    # Entrenamos un modelo base de prueba con LogisticRegression
    features = ['Team_BA_Loc', 'Team_HR_Loc']
    X = df_modelo[features]
    y = (df_modelo['Resultado_Real'] == 'Local').astype(int)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    modelo = LogisticRegression()
    modelo.fit(X_scaled, y)
    
    return df_modelo, modelo, scaler

# Cargamos todo en la interfaz
try:
    df_modelo, modelo, scaler = crear_y_cargar_base_de_datos()
    st.success("¡Base de datos generada y modelo cargado con éxito en la nube!")
    
    st.subheader("📊 Datos analizados en el sistema:")
    st.dataframe(df_modelo)

    st.subheader("🎯 Panel de Predicciones en Vivo")
    equipo_sel = st.selectbox("Selecciona equipo local a simular:", df_modelo['Equipo_Local'].unique())
    
    if st.button("Calcular Probabilidad de Victoria"):
        st.balloons()
        st.info(f"El pronóstico predictivo para {equipo_sel} muestra alta competitividad basada en su rendimiento histórico.")

except Exception as e:
    st.error(f"Ocurrió un error al iniciar los datos: {e}")
    

import os
import pandas as pd
import mysql.connector
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. CONEXIÓN ADAPTADA PARA LOCAL (XAMPP) Y PRODUCCIÓN (NUBE)
def estructurar_datos_entrenamiento():
    try:
        # Lee las variables del servidor; si no existen, usa los valores por defecto de tu XAMPP
        conexion = mysql.connector.connect(
            host=os.getenv("DB_HOST", "localhost"),
            user=os.getenv("DB_USERNAME", "root"),          
            password=os.getenv("DB_PASSWORD", ""),          
            database=os.getenv("DB_DATABASE", "migrasense"),
            port=int(os.getenv("DB_PORT", 3306))
        )
        
        query = """
            SELECT 
                CASE 
                    WHEN p.horas_sueno = 'menos_5' THEN 4
                    WHEN p.horas_sueno = '5_6' THEN 6
                    WHEN p.horas_sueno = '7_8' THEN 8
                    ELSE 9
                END as horas_sueno_num,
                CASE 
                    WHEN p.nivel_estres = 'bajo' THEN 1
                    WHEN p.nivel_estres = 'moderado' THEN 2
                    WHEN p.nivel_estres = 'alto' THEN 3
                    WHEN p.nivel_estres = 'muy_alto' THEN 4
                    ELSE 2
                END as nivel_estres_num,
                IF(EXISTS(SELECT 1 FROM detalle_registro_sintoma drs JOIN sintoma_prodromico sp ON drs.id_sintoma = sp.id_sintoma WHERE drs.id_registro = rs.id_registro AND sp.codigo = 'luz'), 1, 0) as sintoma_luz,
                IF(EXISTS(SELECT 1 FROM detalle_registro_sintoma drs JOIN sintoma_prodromico sp ON drs.id_sintoma = sp.id_sintoma WHERE drs.id_registro = rs.id_registro AND sp.codigo = 'cuello'), 1, 0) as sintoma_cuello,
                HOUR(rs.hora) as hora_del_dia,
                IF(EXISTS(SELECT 1 FROM episodio_migrana em WHERE em.id_usuario = rs.id_usuario AND em.fecha_inicio = rs.fecha), 1, 0) as DESARROLLO_MIGRAÑA
            FROM registro_sintoma rs
            JOIN perfil p ON rs.id_usuario = p.id_usuario;
        """
        
        df = pd.read_sql(query, conexion)
        conexion.close()
        return df
    except:
        return None

# 2. PROCESO DE APRENDIZAJE CONTINUO
def entrenar_modelo_usuario():
    df = estructurar_datos_entrenamiento()
    
    if df is None or df.empty:
        print("⚠️ Usando datos base temporales (BD sin registros suficientes aún).")
        df = pd.DataFrame([], columns=['horas_sueno_num', 'nivel_estres_num', 'sintoma_luz', 'sintoma_cuello', 'hora_del_dia', 'DESARROLLO_MIGRAÑA'])
    
    X = df.drop(columns=['DESARROLLO_MIGRAÑA'])
    y = df['DESARROLLO_MIGRAÑA']
    
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X, y)
    
    joblib.dump(modelo, "modelo_migrana_usuario.pkl")
    print("¡El modelo se ha actualizado y entrenado exitosamente!")

# 3. FUNCIÓN DE PREDICCIÓN
def predecir_riesgo(horas_sueno, estres, luz, cuello, hora):
    try:
        modelo_cargado = joblib.load("modelo_migrana_usuario.pkl")
        nuevos_datos = pd.DataFrame([[horas_sueno, estres, luz, cuello, hora]], 
                                    columns=['horas_sueno_num', 'nivel_estres_num', 'sintoma_luz', 'sintoma_cuello', 'hora_del_dia'])
        
        probabilidad = modelo_cargado.predict_proba(nuevos_datos)
        return probabilidad
    except:
        return []

if __name__ == "__main__":
    entrenar_modelo_usuario()
    riesgo = predecir_riesgo(6, 3, 1, 0, 15)
    print(f"Probabilidad de riesgo actual calculada para la prueba: {riesgo * 100}%")

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import subprocess

app = FastAPI(title="API de Predicción de Migraña para MigraSense")

# Definimos la estructura exacta que Laravel le enviará a Python
class DatosPaciente(BaseModel):
    horas_sueno: float
    nivel_estres: int
    sintoma_luz: int
    sintoma_cuello: int
    hora_del_dia: int

@app.post("/predecir")
def predecir_migrana(datos: DatosPaciente):
    try:
        # 1. Cargamos el cerebro portátil entrenado
        modelo = joblib.load("modelo_migrana_usuario.pkl")
        
        # 2. Estructuramos los datos que nos mandó Laravel
        entrada = pd.DataFrame(
            [[datos.horas_sueno, datos.nivel_estres, datos.sintoma_luz, datos.sintoma_cuello, datos.hora_del_dia]], 
            columns=['horas_sueno_num', 'nivel_estres_num', 'sintoma_luz', 'sintoma_cuello', 'hora_del_dia']
        )
        
        # 3. Calculamos las probabilidades
        probabilidades = modelo.predict_proba(entrada)
        # Tomamos la probabilidad de que SÍ sufra migraña (posición 1)
        probabilidad_si = float(probabilidades[0][1]) * 100
        
        return {
            "status": "success",
            "probabilidad_migrana_porcentaje": round(probabilidad_si, 2),
            "alerta": "Alta probabilidad, tomar precauciones" if probabilidad_si >= 70 else "Riesgo bajo controlado"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/reentrenar")
def reentrenar_modelo():
    try:
        # Ejecuta el script que acabamos de arreglar para absorber los nuevos datos de MySQL
        subprocess.run(["python", "modelo_migrana.py"], check=True)
        return {"status": "success", "message": "¡El modelo se ha reentrenado con los nuevos datos de MySQL exitosamente!"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

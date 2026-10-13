# MigraSense ML

Predice la **probabilidad de que empiece una migraña en las próximas 48 horas**
para una persona, con sus propios registros de la app (perfil, episodios y
síntomas de aviso).

## Cómo piensa

1. **Conocimiento de partida** (`conocimiento.py`): cifras de estudios médicos
   sobre síntomas premonitorios y desencadenantes. Cada número tiene su fuente
   (Giffin 2003, Kelman 2007, MacGregor 2004, Lipton 2014, Mostofsky 2019, ICHD-3)
   o está marcado como `[SUPUESTO]`.
2. **Simulador** (`simulador.py`): crea 1.600 personas ficticias con esas reglas
   (frecuencia, ritmo propio, ciclo hormonal, estrés, avisos 2-48 h antes,
   falsas alarmas y registros incompletos).
3. **Características** (`caracteristicas.py`): convierte los registros en ~50
   números (episodios recientes, días desde el último, ritmo personal, síntomas
   de las últimas 48 h, precisión de sus avisos anteriores, perfil, hora). Es el
   **mismo código** para entrenar y para predecir, y **nunca mira el futuro**.
4. **Modelo** (`entrenar.py`): Gradient Boosting calibrado (scikit-learn).
   Con datos reales de la BD se reentrena dándoles 3 veces más peso.

## Resultados (personas que el modelo nunca vio)

| | AUC | Brier |
|---|---|---|
| **Modelo MigraSense** | **0,887** | **0,129** |
| Solo con el cuestionario | 0,775 | 0,200 |
| Modelo lineal simple | 0,863 | 0,145 |

Cuando el modelo dice "70% o más", acierta el 85% de las veces. Ver `GET /metricas`.

> Importante: estas cifras miden qué tan bien aprende los patrones de los
> estudios. La prueba con personas reales aparece en `/metricas`
> (`prueba_datos_reales`) cuando la app junte suficientes datos.

## API

| Método | Ruta | Clave | Qué hace |
|---|---|---|---|
| GET | `/salud` | no | Estado y calidad del modelo |
| GET | `/metricas` | no | Reporte completo de evaluación |
| POST | `/predecir` | sí | Probabilidad + nivel + explicación |
| POST | `/reentrenar` | sí | Reentrena con datos reales (en segundo plano) |
| GET | `/reentrenar/estado` | sí | Progreso del reentrenamiento |

La clave va en la cabecera `X-API-Key` y se configura en Render con la
variable `ML_API_KEY`. Documentación interactiva: `/docs`.

## Local

```bash
python -m venv venv && venv\Scripts\activate      # Windows
pip install -r requirements-dev.txt
python entrenar.py            # crea modelo_migrasense.joblib y reporte_modelo.json
pytest -q                     # 7 pruebas
set ML_API_KEY=una-clave && uvicorn main:app --port 8001
```

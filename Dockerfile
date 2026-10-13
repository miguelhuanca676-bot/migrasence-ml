# MigraSense ML · imagen para Render
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY conocimiento.py caracteristicas.py simulador.py entrenar.py main.py ./

# Modelo YA ENTRENADO (con `python entrenar.py`, mismas versiones de requirements.txt).
# Entrenarlo aquí tardaba demasiado en el plan gratis de Render.
COPY modelo_migrasense.joblib reporte_modelo.json ./
RUN python -c "import joblib; p = joblib.load('modelo_migrasense.joblib'); print('Modelo OK', p['version'])"

# Usuario sin privilegios (si alguien rompiera la API, no sería root)
RUN useradd --create-home app && chown -R app /app
USER app

# Render indica el puerto en la variable PORT (por defecto 10000)
EXPOSE 10000
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}"]

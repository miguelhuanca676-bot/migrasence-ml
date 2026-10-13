# MigraSense ML · imagen para Render
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY conocimiento.py caracteristicas.py simulador.py entrenar.py main.py ./

# El modelo se ENTRENA al construir la imagen (base de estudios, ~2-4 min).
# Así siempre coincide con la versión de scikit-learn instalada.
RUN python entrenar.py --simulados 1600 > /dev/null

# Usuario sin privilegios (si alguien rompiera la API, no sería root)
RUN useradd --create-home app && chown -R app /app
USER app

# Render indica el puerto en la variable PORT (por defecto 10000)
EXPOSE 10000
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}"]

# 1. Usamos una versión ligera y oficial de Python 3.11
FROM python:3.11-slim

# 2. Establecemos la carpeta de trabajo dentro del servidor
WORKDIR /app

# 3. Copiamos el archivo de requisitos al servidor
COPY requirements.txt .

# 4. Instalamos todas las librerías sin almacenar caché para ahorrar espacio
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copiamos todo el código de nuestra carpeta local al servidor
COPY . .

# 6. Exponemos el puerto 8000 que es donde escucha FastAPI
EXPOSE 8000

# 7. Comando para encender la API automáticamente cuando el servidor inicie
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

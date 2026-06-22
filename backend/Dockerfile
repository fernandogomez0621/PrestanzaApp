FROM python:3.12-slim

WORKDIR /app

# Dependencias del sistema (algunas libs de ML las necesitan)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Carpetas persistentes (se montan como volumen en compose)
RUN mkdir -p datos_actuales modelos_versionados

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

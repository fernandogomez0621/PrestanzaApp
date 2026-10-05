# Prestanza — Modelo Prospectivo de Riesgo (FastAPI + React)

## Opción A — Docker (recomendada, un solo comando)

Requisito: tener Docker y Docker Compose instalados.

    docker compose up --build

Luego abre:

- **http://localhost:8501** — app principal (4 familias de modelos, SHAP, EDA…)
- **http://localhost:8502** — solo los modelos de riesgo por índices de 6 y 12 meses

- El frontend (React + Nginx) de la app principal queda en el puerto **8501**.
- El backend (FastAPI) corre interno y el frontend le habla por `/api` (Nginx hace el proxy).
- Los datos cargados y los modelos entrenados **persisten** en `backend/datos_actuales/`
  y `backend/modelos_versionados/` aunque apagues los contenedores.

Para detener:  `docker compose down`
Para reconstruir tras cambios:  `docker compose up --build`

## Opción B — Manual (sin Docker)

Backend:

    cd backend
    pip install -r requirements.txt
    uvicorn main:app --reload --port 8000

Frontend (otra terminal):

    cd frontend
    npm install
    npm run dev        # abre http://localhost:5173

## Modelos por índices (puerto 8502)
- Servicios `backend-indices` y `frontend-indices` en `docker-compose.yml`. Usan la misma imagen y el mismo código
  (`backend/main_indices.py`, `backend/core/modelo_indices.py`); el frontend se compila con `MODO=indices`.
- Comparten `backend/datos_actuales/` con la app principal: lo que se cargue en una queda disponible en la otra.
- Los modelos se guardan en `backend/modelos_indices/<fecha>/` (JSON legibles + métricas). Si no hay ninguno al
  arrancar y existen datos, se entrena solo en unos segundos.
- Pestañas: carga de datos, modelos & versiones (métricas por validación cruzada), predicción (índices, puntaje,
  banda y decisión por simulación) y cómo se construyen.

## Flujo de uso
1. **Carga de datos**: sube calificaciones 6M y 12M (mapea columnas si el encabezado no coincide) + datapoints. Botón **reentrenar**.
2. **Calificaciones**: distribución AAA/AA/A… y perfil filtrable, 6M vs 12M.
3. **Modelos & versiones**: elige el entrenamiento por fecha; métricas de las 4 familias.
4. **Predicción**: sube un CSV; muestra deudor principal (titular) + codeudores con 9 clases / 3 clases / Buena-NoBuena / Buena-NoBuena(SHAP) en 6M y 12M.
5. **Interpretabilidad (SHAP)**: a qué le da peso el modelo y qué variables van en sentido lógico.

## Estructura
    prestanza_app/
    ├── docker-compose.yml      # orquesta backend + frontend
    ├── backend/
    │   ├── Dockerfile
    │   ├── main.py             # API FastAPI
    │   ├── core/               # pipeline ML (código ya corregido)
    │   ├── datos_ejemplo/      # datos de prueba para arrancar
    │   ├── datos_actuales/     # (volumen) datos cargados por la app
    │   └── modelos_versionados/# (volumen) modelos entrenados por fecha
    └── frontend/
        ├── Dockerfile          # build React -> Nginx
        ├── nginx.conf
        └── src/

## Notas
- Para arrancar con datos de prueba: copia los de `backend/datos_ejemplo/` o súbelos desde la pestaña de carga.
- Las métricas reflejan los datos reales: Buena/No-Buena ~63-67% es lo sólido; 9 clases y la clase Riesgo son bajas por falta de casos, no por la app. Al reentrenar con más datos de Media/Riesgo mejora solo.

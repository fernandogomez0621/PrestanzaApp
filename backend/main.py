"""
backend/main.py — API FastAPI del sistema Prestanza ML
Expone: carga de archivos, estadísticas, mapeo de columnas, entrenamiento con
versionado, SHAP, y predicción multi-modelo con codeudores.
Correr:  uvicorn main:app --reload --port 8000
"""
import os, sys, json, shutil, tempfile, datetime
import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))
import core.pipeline as P
from io_utils import leer_tabla, normalizar_columnas
from clasificador_riesgo import agrupar_calificacion

app = FastAPI(title="Prestanza ML API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

DATA_DIR = os.path.join(os.path.dirname(__file__), 'datos_actuales')
os.makedirs(DATA_DIR, exist_ok=True)

# Mapa de la calificación numérica a letra (para estadísticas)
# El mapeo número→letra ahora vive en core/calif_config.py (configurable)

COLUMNAS_ESPERADAS_CALIF = ['id_simulacion', 'Valor_calificacion']


def _guardar_upload(file: UploadFile, nombre: str) -> str:
    ruta = os.path.join(DATA_DIR, nombre)
    with open(ruta, 'wb') as f:
        shutil.copyfileobj(file.file, f)
    return ruta


@app.get("/api/health")
def health():
    return {"status": "ok", "fecha": datetime.datetime.now().isoformat()}


# ---------- Pesos del deudor principal (configurables) ----------
PESOS_PATH = os.path.join(DATA_DIR, 'pesos.json')
# etiquetas legibles para el front
PESOS_LABELS = {
    'debtor_level_of_education': 'Nivel de educación',
    'equityFactor': 'Patrimonio / equity',
    'quotefactor': 'Capacidad de pago (cuota)',
    'debtor_credit_score': 'Score crediticio',
    'creditos_cerrados_factor': 'Créditos cerrados',
    'debtor_closed_no_similar_loans': 'Historial crédito similar',
    'bank_inflow_factor': 'Ingresos bancarios',
    'bank_average_factor': 'Saldo bancario promedio',
    'snr_estimated_unique_properties': 'Propiedades',
    'public_db_no_lawsuits': 'Demandas (general) — resta',
    'public_db_no_lawsuits_defendant_money': 'Demandas por dinero — resta',
}


def _pesos_default():
    from data_processor import DataProcessor
    return dict(DataProcessor().PESOS_DEFAULT)


def _leer_pesos():
    if os.path.exists(PESOS_PATH):
        try:
            return json.load(open(PESOS_PATH))
        except Exception:
            pass
    return _pesos_default()


@app.get("/api/pesos")
def get_pesos():
    pesos = _leer_pesos()
    return {
        "pesos": pesos,
        "default": _pesos_default(),
        "labels": PESOS_LABELS,
        "negativos": ['public_db_no_lawsuits', 'public_db_no_lawsuits_defendant_money'],
    }


@app.post("/api/pesos")
async def set_pesos(pesos: dict):
    base = _pesos_default()
    for k, v in (pesos or {}).items():
        if k in base:
            base[k] = float(v)
    with open(PESOS_PATH, 'w') as f:
        json.dump(base, f, indent=2)
    return {"ok": True, "pesos": base}


# ---------- Reglas de puntaje (escalones / categorías) configurables ----------
REGLAS_PATH = os.path.join(DATA_DIR, 'reglas.json')


def _reglas_default():
    import reglas_puntaje as R
    import copy
    return copy.deepcopy(R.CONFIG_DEFAULT)


def _leer_reglas():
    if os.path.exists(REGLAS_PATH):
        try:
            return json.load(open(REGLAS_PATH))
        except Exception:
            pass
    return _reglas_default()


@app.get("/api/reglas")
def get_reglas():
    import reglas_puntaje as R
    return {
        "reglas": _leer_reglas(),
        "default": _reglas_default(),
        "labels": R.LABELS_FACTORES,
        "negativos": list(R.FACTORES_NEGATIVOS),
    }


@app.post("/api/reglas")
async def set_reglas(reglas: dict):
    # validación mínima de estructura
    if not isinstance(reglas, dict) or 'escalones' not in reglas or 'categorias' not in reglas:
        raise HTTPException(400, "Estructura de reglas inválida")
    with open(REGLAS_PATH, 'w') as f:
        json.dump(reglas, f, indent=2, ensure_ascii=False)
    return {"ok": True, "reglas": reglas}


# ---------- Config de calificaciones: letras + cortes de clases ----------
CALIF_PATH = os.path.join(DATA_DIR, 'calif_config.json')


def _calif_default():
    import calif_config as CC
    return {"letras": [dict(r) for r in CC.LETRAS_DEFAULT], "cortes": dict(CC.CORTES_DEFAULT)}


def _leer_calif():
    if os.path.exists(CALIF_PATH):
        try:
            return json.load(open(CALIF_PATH))
        except Exception:
            pass
    return _calif_default()


@app.get("/api/calificaciones-config")
def get_calif_config():
    import calif_config as CC
    return {"config": _leer_calif(), "default": _calif_default(), "orden_clases": CC.ORDEN_CLASES}


@app.post("/api/calificaciones-config")
async def set_calif_config(config: dict):
    if 'letras' not in config or 'cortes' not in config:
        raise HTTPException(400, "Falta 'letras' o 'cortes'")
    with open(CALIF_PATH, 'w') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    return {"ok": True, "config": config}


# ---------- EDA: variables del modelo vs calificación ----------
@app.get("/api/eda/{periodo}")
def eda(periodo: str, modo_seleccion: str = 'titular'):
    import eda as EDA
    dpath = os.path.join(DATA_DIR, 'datapoints.csv')
    cpath = os.path.join(DATA_DIR, f'calificaciones_{periodo}.csv')
    if not (os.path.exists(dpath) and os.path.exists(cpath)):
        return {"disponible": False, "mensaje": "Faltan archivos para este horizonte"}
    pesos = _leer_pesos(); reglas = _leer_reglas(); cortes = _leer_calif().get('cortes')
    res = EDA.analizar(dpath, cpath, modo_seleccion=modo_seleccion,
                       pesos=pesos, reglas=reglas, cortes=cortes)
    res["disponible"] = True
    res["periodo"] = periodo
    return res


# ---------- 1. Inspección de columnas (para el mapeo manual) ----------
@app.post("/api/inspeccionar")
async def inspeccionar(file: UploadFile = File(...)):
    """Devuelve las columnas detectadas + preview, para que el front permita mapear."""
    tmp = os.path.join(tempfile.gettempdir(), file.filename)
    with open(tmp, 'wb') as f:
        shutil.copyfileobj(file.file, f)
    try:
        df = leer_tabla(tmp)
    except Exception as e:
        raise HTTPException(400, f"No se pudo leer el archivo: {e}")
    return {
        "columnas": list(df.columns),
        "preview": df.head(5).fillna('').astype(str).to_dict(orient='records'),
        "filas": len(df),
        "columnas_esperadas": COLUMNAS_ESPERADAS_CALIF,
    }


# ---------- 2. Subir archivos (con mapeo opcional de columnas) ----------
@app.post("/api/subir/calificaciones")
async def subir_calificaciones(periodo: str = Form(...), file: UploadFile = File(...),
                               col_id_simulacion: str = Form(None), col_calificacion: str = Form(None)):
    if periodo not in ('6M', '12M'):
        raise HTTPException(400, "periodo debe ser 6M o 12M")
    tmp = os.path.join(tempfile.gettempdir(), file.filename)
    with open(tmp, 'wb') as f:
        shutil.copyfileobj(file.file, f)
    df = leer_tabla(tmp)
    # aplicar mapeo manual si el encabezado no es el esperado
    ren = {}
    if col_id_simulacion and col_id_simulacion in df.columns:
        ren[col_id_simulacion] = 'id_simulacion'
    if col_calificacion and col_calificacion in df.columns:
        ren[col_calificacion] = 'Valor_calificacion'
    if ren:
        df = df.rename(columns=ren)
    df = normalizar_columnas(df)
    if 'Valor_calificacion' not in df.columns or 'id_simulacion' not in df.columns:
        raise HTTPException(400, "Faltan columnas id_simulacion / Valor_calificacion. Use el mapeo.")
    destino = os.path.join(DATA_DIR, f'calificaciones_{periodo}.csv')
    df.to_csv(destino, index=False)
    return {"ok": True, "periodo": periodo, "filas": len(df), "ruta": destino}


@app.post("/api/subir/datapoints")
async def subir_datapoints(file: UploadFile = File(...)):
    ruta = _guardar_upload(file, 'datapoints.csv')
    df = leer_tabla(ruta)
    return {"ok": True, "filas": len(df), "simulaciones": int(df['id_simulacion'].nunique())}


# ---------- 3. Estadísticas descriptivas por horizonte ----------
@app.get("/api/estadisticas/{periodo}")
def estadisticas(periodo: str):
    dpath = os.path.join(DATA_DIR, 'datapoints.csv')
    cpath = os.path.join(DATA_DIR, f'calificaciones_{periodo}.csv')
    if not (os.path.exists(dpath) and os.path.exists(cpath)):
        return {"disponible": False, "mensaje": "Faltan archivos para este horizonte"}
    datos = P.procesar_horizonte(dpath, cpath)
    cfg = _leer_calif()
    letras = cfg['letras']; cortes = cfg['cortes']
    import calif_config as CC
    vals = datos['Valor_calificacion'].dropna().astype(float)
    dist_letras = {}
    for v in vals:
        L = CC.numero_a_letra(v, letras)
        dist_letras[L] = dist_letras.get(L, 0) + 1
    clase3 = vals.apply(lambda v: CC.agrupar_con_cortes(v, cortes)).value_counts().to_dict()
    # tabla de los datapoints (perfil) de cada crédito, para filtrar en el front
    cols_perfil = ['id_simulacion', 'debtor_credit_score', 'debtor_level_of_education',
                   'original_amount', 'Valor_calificacion']
    cols_perfil = [c for c in cols_perfil if c in datos.columns]
    tabla = datos[cols_perfil].copy()
    tabla['clase3'] = vals.apply(lambda v: CC.agrupar_con_cortes(v, cortes)).values
    tabla['letra'] = vals.apply(lambda v: CC.numero_a_letra(v, letras)).values
    return {
        "disponible": True, "periodo": periodo,
        "n_creditos": int(len(datos)),
        "distribucion_letras": dist_letras,
        "distribucion_3clases": clase3,
        "estadisticas_valor": {
            "media": round(float(vals.mean()), 2), "mediana": float(vals.median()),
            "min": float(vals.min()), "max": float(vals.max()),
        },
        "tabla": tabla.fillna('').astype(str).to_dict(orient='records'),
    }


# ---------- 4. Entrenar (con versionado) ----------
@app.post("/api/entrenar")
def entrenar(modo_seleccion: str = Form('titular')):
    dpath = os.path.join(DATA_DIR, 'datapoints.csv')
    pesos = _leer_pesos()
    reglas = _leer_reglas()
    cortes = _leer_calif().get('cortes')
    bundles = {}
    metricas = {}
    for periodo in ['6M', '12M']:
        cpath = os.path.join(DATA_DIR, f'calificaciones_{periodo}.csv')
        if os.path.exists(dpath) and os.path.exists(cpath):
            datos = P.procesar_horizonte(dpath, cpath, modo_seleccion=modo_seleccion, pesos=pesos, reglas=reglas)
            b = P.entrenar_horizonte(datos, periodo, cortes=cortes)
            bundles[periodo] = b
            metricas[periodo] = b['metricas']
        else:
            bundles[periodo] = None
    if not any(bundles.values()):
        raise HTTPException(400, "No hay datos suficientes para entrenar")
    version = P.guardar_version(bundles, {'origen': 'api', 'modo_seleccion': modo_seleccion})
    return {"ok": True, "version": version, "modo_seleccion": modo_seleccion, "metricas": metricas}


# ---------- 5. Versiones de modelos ----------
@app.get("/api/versiones")
def versiones():
    return {"versiones": P.listar_versiones()}


@app.get("/api/modelos/{version}")
def detalle_modelos(version: str):
    bundles = P.cargar_version(version)
    salida = {}
    for periodo, b in bundles.items():
        if b is None:
            continue
        salida[periodo] = {
            "n": b['n'], "metricas": b['metricas'],
            "shap": b['shap'].get('buena_no_buena', []),
            "variables_con_sentido": b['shap'].get('variables_con_sentido', []),
            "comparacion": b.get('comparacion', {}),
        }
    return salida


# ---------- 6. Predecir con uno o varios CSV ----------
@app.post("/api/predecir")
async def predecir(version: str = Form(...), modo_seleccion: str = Form('titular'),
                   files: list[UploadFile] = File(...)):
    bundles = P.cargar_version(version)
    if not any(bundles.values()):
        raise HTTPException(404, "Versión de modelos no encontrada")
    pesos = _leer_pesos()
    reglas = _leer_reglas()
    archivos = []
    for file in files:
        tmp = os.path.join(tempfile.gettempdir(), file.filename)
        with open(tmp, 'wb') as f:
            shutil.copyfileobj(file.file, f)
        try:
            res = P.predecir_csv(tmp, bundles, modo_seleccion=modo_seleccion, pesos=pesos, reglas=reglas)
            archivos.append({"archivo": file.filename, "ok": True, "resultados": res})
        except Exception as e:
            archivos.append({"archivo": file.filename, "ok": False, "error": str(e)})
    return {"ok": True, "version": version, "modo_seleccion": modo_seleccion, "archivos": archivos}

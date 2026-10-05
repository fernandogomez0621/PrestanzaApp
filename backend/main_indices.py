"""
backend/main_indices.py — API de los modelos de riesgo por índices (6M y 12M).
Se despliega aparte (puerto 8502) y comparte datos_actuales/ con la app principal.
Correr:  uvicorn main_indices:app --port 8000
"""
import os, sys, json, shutil, tempfile, datetime, threading
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))
from io_utils import leer_tabla, normalizar_columnas
import modelo_indices as MI

app = FastAPI(title="Prestanza — Modelos por índices", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

BASE = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE, 'datos_actuales')
MOD_DIR = os.path.join(BASE, 'modelos_indices')
os.makedirs(DATA_DIR, exist_ok=True); os.makedirs(MOD_DIR, exist_ok=True)
U = MI.U
_lock = threading.Lock()


def _tmp(file: UploadFile) -> str:
    ruta = os.path.join(tempfile.gettempdir(), os.path.basename(file.filename or 'archivo.csv'))
    with open(ruta, 'wb') as f:
        shutil.copyfileobj(file.file, f)
    return ruta


def _calif(periodo):
    ruta = os.path.join(DATA_DIR, f'calificaciones_{periodo}.csv')
    if not os.path.exists(ruta): return None
    c = leer_tabla(ruta, dtype={'id_simulacion': 'str'})
    c = normalizar_columnas(c).dropna(subset=['Valor_calificacion'])
    c['id_simulacion'] = c['id_simulacion'].astype(str)
    return c.drop_duplicates('id_simulacion', keep='last').set_index('id_simulacion')['Valor_calificacion'].astype(float)


def _corte_buena():
    """Corte de Buena compartido con la app principal (pestaña 'Letras & clases' -> calif_config.json)."""
    ruta = os.path.join(DATA_DIR, 'calif_config.json')
    try:
        return float(json.load(open(ruta, encoding='utf-8'))['cortes']['Buena'])
    except Exception:
        return float(MI.CORTE_BUENA)


@app.get("/api/corte")
def corte():
    return {"corte_buena": _corte_buena()}


def _leer_datapoints(ruta):
    return leer_tabla(ruta, dtype={'id_simulacion': 'str', U: 'str'})


def _limpio(x):
    """Convierte NaN/inf a None para JSON."""
    if isinstance(x, dict): return {k: _limpio(v) for k, v in x.items()}
    if isinstance(x, list): return [_limpio(v) for v in x]
    if isinstance(x, (float, np.floating)): return None if not np.isfinite(x) else float(x)
    if isinstance(x, np.integer): return int(x)
    return x


# ---------------------------------------------------------------- salud y carga de datos (mismo formato que la app)
@app.get("/api/health")
def health():
    return {"status": "ok", "servicio": "indices", "fecha": datetime.datetime.now().isoformat()}


@app.post("/api/inspeccionar")
async def inspeccionar(file: UploadFile = File(...)):
    try:
        df = leer_tabla(_tmp(file))
    except Exception as e:
        raise HTTPException(400, f"No se pudo leer el archivo: {e}")
    return {"columnas": list(df.columns), "preview": df.head(5).fillna('').astype(str).to_dict(orient='records'),
            "filas": len(df), "columnas_esperadas": ['id_simulacion', 'Valor_calificacion']}


@app.post("/api/subir/calificaciones")
async def subir_calificaciones(periodo: str = Form(...), file: UploadFile = File(...),
                               col_id_simulacion: str = Form(None), col_calificacion: str = Form(None)):
    if periodo not in ('6M', '12M'):
        raise HTTPException(400, "periodo debe ser 6M o 12M")
    df = leer_tabla(_tmp(file))
    ren = {}
    if col_id_simulacion and col_id_simulacion in df.columns: ren[col_id_simulacion] = 'id_simulacion'
    if col_calificacion and col_calificacion in df.columns: ren[col_calificacion] = 'Valor_calificacion'
    df = normalizar_columnas(df.rename(columns=ren))
    if 'Valor_calificacion' not in df.columns or 'id_simulacion' not in df.columns:
        raise HTTPException(400, "Faltan columnas id_simulacion / Valor_calificacion. Use el mapeo.")
    destino = os.path.join(DATA_DIR, f'calificaciones_{periodo}.csv')
    df.to_csv(destino, index=False)
    return {"ok": True, "periodo": periodo, "filas": len(df)}


@app.post("/api/subir/datapoints")
async def subir_datapoints(file: UploadFile = File(...)):
    ruta = os.path.join(DATA_DIR, 'datapoints.csv')
    with open(ruta, 'wb') as f:
        shutil.copyfileobj(file.file, f)
    df = leer_tabla(ruta)
    return {"ok": True, "filas": len(df), "simulaciones": int(df['id_simulacion'].nunique())}


# ---------------------------------------------------------------- entrenamiento con versionado
def _entrenar():
    dpath = os.path.join(DATA_DIR, 'datapoints.csv')
    c6, c12 = _calif('6M'), _calif('12M')
    if not os.path.exists(dpath) or (c6 is None and c12 is None):
        raise HTTPException(400, "Faltan datapoints o calificaciones para entrenar")
    corte = _corte_buena()
    V12, V6 = MI.construir_variables(_leer_datapoints(dpath))
    version = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    carpeta = os.path.join(MOD_DIR, version); os.makedirs(carpeta, exist_ok=True)
    metricas = {}
    for h, V, y, y12 in [('12M', V12, c12, None), ('6M', V6, c6, c12)]:
        if y is None: continue
        yy = y.reindex(V.index); y12r = None if y12 is None else y12.reindex(V.index)
        if yy.notna().sum() < 20 or (yy.dropna() < corte).sum() < 5 or (yy.dropna() >= corte).sum() < 5:
            continue
        m = MI.ModeloIndices(h, corte).fit(V, yy, y12r)
        m.guardar(os.path.join(carpeta, f'modelo_{h}.json'))
        ev = MI.evaluar(h, V, yy, y12r, corte=corte)
        ev.update(umbral=round(m.umbral, 4), terciles=[round(t, 4) for t in m.terciles], frac_marcado=round(m.frac_marcado, 3),
                  n_entrenamiento=m.n_entrenamiento, malos_entrenamiento=m.malos_entrenamiento)
        metricas[h] = ev
    if not metricas:
        shutil.rmtree(carpeta, ignore_errors=True)
        raise HTTPException(400, f"No hay suficientes operaciones calificadas para entrenar con Buena >= {corte:g}")
    json.dump(_limpio({'version': version, 'corte_buena': corte, 'fecha': datetime.datetime.now().isoformat(timespec='seconds'), 'metricas': metricas}),
              open(os.path.join(carpeta, 'meta.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    return version, metricas


@app.post("/api/entrenar")
def entrenar(modo_seleccion: str = Form('titular')):
    with _lock:
        version, metricas = _entrenar()
    return _limpio({"ok": True, "version": version, "corte_buena": _corte_buena(), "metricas": metricas})


@app.on_event("startup")
def _entrenar_si_no_hay():
    """Primera vez: si hay datos y aún no existe ninguna versión, entrena en segundo plano."""
    if not _versiones():
        def tarea():
            try:
                with _lock: _entrenar()
            except Exception as e:
                print('Entrenamiento inicial omitido:', e)
        threading.Thread(target=tarea, daemon=True).start()


# ---------------------------------------------------------------- versiones y detalle
def _versiones():
    out = []
    for v in sorted(os.listdir(MOD_DIR), reverse=True):
        meta = os.path.join(MOD_DIR, v, 'meta.json')
        if os.path.exists(meta):
            try: c = json.load(open(meta, encoding='utf-8')).get('corte_buena', MI.CORTE_BUENA)
            except Exception: c = MI.CORTE_BUENA
            out.append({"version": v, "modo_seleccion": "índices", "corte_buena": c})
    return out


@app.get("/api/versiones")
def versiones():
    return {"versiones": _versiones()}


def _cargar(version):
    carpeta = os.path.join(MOD_DIR, os.path.basename(version))
    if not os.path.exists(os.path.join(carpeta, 'meta.json')):
        raise HTTPException(404, "Versión no encontrada")
    modelos = {h: MI.ModeloIndices.cargar(os.path.join(carpeta, f'modelo_{h}.json'))
               for h in ('6M', '12M') if os.path.exists(os.path.join(carpeta, f'modelo_{h}.json'))}
    return modelos, json.load(open(os.path.join(carpeta, 'meta.json'), encoding='utf-8'))


@app.get("/api/modelos/{version}")
def detalle_modelos(version: str):
    modelos, meta = _cargar(version)
    salida = {}
    for h, m in modelos.items():
        salida[h] = {**meta['metricas'].get(h, {}), 'definiciones': m.defs, 'tasa_entrenamiento': m.tasa_entrenamiento,
                     'corte_buena': m.corte}
    return _limpio(salida)


# ---------------------------------------------------------------- predicción
@app.post("/api/predecir")
async def predecir(version: str = Form(...), modo_seleccion: str = Form('titular'), files: list[UploadFile] = File(...)):
    modelos, _ = _cargar(version)
    archivos = []
    for file in files:
        try:
            V12, V6 = MI.construir_variables(_leer_datapoints(_tmp(file)))
            P = {h: m.predecir(V12 if h == '12M' else V6) for h, m in modelos.items()}
            ids = sorted(set(V12.index) | set(V6.index))
            res = []
            for sid in ids:
                fila = {"id_simulacion": str(sid)}
                for h, m in modelos.items():
                    if sid in P[h].index:
                        r = P[h].loc[sid]
                        fila[h] = {"indices": {k: float(r[k]) for k in m.defs}, "puntaje": float(r['puntaje']), "umbral": round(m.umbral, 4),
                                   "banda": r['banda'], "decision": r['decision']}
                    else:
                        fila[h] = None      # 6M: sin datos del crédito (monto o garantía en 0)
                res.append(fila)
            archivos.append({"archivo": file.filename, "ok": True, "resultados": res})
        except Exception as e:
            archivos.append({"archivo": file.filename, "ok": False, "error": str(e)})
    return _limpio({"ok": True, "version": version, "archivos": archivos})

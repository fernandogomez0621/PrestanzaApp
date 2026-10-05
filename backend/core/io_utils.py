"""
io_utils.py
-----------
Utilidades compartidas por data_processor.py y predictor.py para:

  1. Lectura ROBUSTA de tablas (CSV con cualquier separador/encoding + Excel .xlsx/.xls/.ods).
  2. Normalización de ESQUEMA (distintas versiones de export del sistema de hipotecas
     usan nombres de columna y formatos de fecha diferentes).
  3. Codificación ORDINAL del nivel educativo (consistente entre entrenamiento y predicción).

Centralizar esto evita el problema de "dependiendo del CSV lo lee o no": el separador,
el encoding y el nombre de las columnas dejan de estar hardcodeados.
"""

import io
import os
import csv
import pandas as pd

try:
    import chardet  # opcional, mejora la detección de encoding
except Exception:  # pragma: no cover
    chardet = None


# ---------------------------------------------------------------------------
# 1. EDUCACIÓN COMO VARIABLE ORDINAL
# ---------------------------------------------------------------------------
# El nivel educativo es ORDINAL (sin estudio < primaria < ... < doctorado).
# Codificarlo así (un solo número) en lugar de one-hot evita dos bugs:
#   - El modelo no puede dar "más peso a bachillerato que a profesional"
#     porque la relación es monótona (a mayor nivel, mayor valor).
#   - Si en predicción aparece un nivel que no estaba en entrenamiento
#     (p.ej. "primaria"), NO rompe: simplemente toma su rango. No hay
#     columnas dummy que cuadrar entre train y predict.
EDU_ORDINAL = {
    'desconocido': 0,
    'no_info': 0,
    'sin estudio': 1,
    'primaria': 2,
    'bachillerato': 3,
    'técnico profesional': 4,
    'tecnico profesional': 4,
    'tecnológico': 5,
    'tecnologico': 5,
    'pregrado': 6,
    'profesional': 6,
    'especialización': 7,
    'especializacion': 7,
    'maestría': 8,
    'maestria': 8,
    'doctorado': 9,
}

EDUCATION_ORDINAL_FEATURE = 'debtor_education_ordinal'


def educacion_a_ordinal(valor):
    """Convierte un nivel educativo (texto) a su rango ordinal. Robusto a None / nuevos valores."""
    if valor is None:
        return 0
    clave = str(valor).strip().lower()
    return EDU_ORDINAL.get(clave, 0)


# ---------------------------------------------------------------------------
# 2. NORMALIZACIÓN DE ESQUEMA
# ---------------------------------------------------------------------------
# Distintas versiones del export traen nombres distintos para la misma columna.
# Mapeamos todo a un esquema canónico para que el resto del pipeline no dependa
# de la versión del archivo.
ALIAS_COLUMNAS = {
    'id_usuario': 'id_usuario (opcional)',
    'id_usuario (opcional)': 'id_usuario (opcional)',
    'idsimulacion': 'id_simulacion',
    'id_simulacion': 'id_simulacion',
    # La columna del valor de la calificación cambió de nombre entre versiones
    'valor_calificacion': 'Valor_calificacion',
    'calificacion_de_riesgo_crediticio': 'Valor_calificacion',
    'calificacion_de_riesgo': 'Valor_calificacion',
}


def normalizar_columnas(df):
    """Limpia y estandariza nombres de columna a un esquema canónico."""
    df = df.copy()
    # quitar espacios, comillas y BOM
    df.columns = (
        df.columns.astype(str)
        .str.replace('\ufeff', '', regex=False)
        .str.strip()
        .str.replace('"', '', regex=False)
        .str.replace("'", '', regex=False)
    )
    # aplicar alias (case-insensitive)
    nuevos = {}
    for c in df.columns:
        nuevos[c] = ALIAS_COLUMNAS.get(c.lower(), c)
    df = df.rename(columns=nuevos)
    return df


def normalizar_datapoints(df):
    """
    Normaliza un dataframe de datapoints sin importar la versión del export.
    - Unifica nombres de columna (id_usuario / id_usuario (opcional), etc.).
    - Normaliza data_set: 'general_loan_data' -> 'general-loan-data'
      (algunas versiones usan guion bajo). data_name se deja igual porque
      es consistente entre versiones.
    """
    df = normalizar_columnas(df)
    if 'data_set' in df.columns:
        df['data_set'] = df['data_set'].astype(str).str.replace('_', '-', regex=False)
    return df


def parsear_fechas(serie):
    """
    Parseo flexible de fechas. Soporta dd/mm/yyyy (export viejo) e ISO
    'yyyy-mm-dd hh:mm:ss' (export nuevo) sin reventar.
    """
    # 1) intento ISO / formato estándar
    fechas = pd.to_datetime(serie, errors='coerce', format='mixed', dayfirst=True)
    return fechas


# ---------------------------------------------------------------------------
# 3. LECTURA ROBUSTA (CSV multi-encoding/separador + Excel)
# ---------------------------------------------------------------------------
def _detectar_encoding(raw_bytes):
    """Detecta encoding probando chardet y una lista de candidatos comunes en CO."""
    candidatos = ['utf-8-sig', 'utf-8', 'cp1252', 'latin-1']
    if chardet is not None:
        try:
            guess = chardet.detect(raw_bytes[:100000])
            enc = guess.get('encoding')
            if enc and enc.lower() not in [c.lower() for c in candidatos]:
                candidatos.insert(0, enc)
        except Exception:
            pass
    for enc in candidatos:
        try:
            raw_bytes.decode(enc)
            return enc
        except Exception:
            continue
    return 'latin-1'  # latin-1 nunca falla al decodificar


def _detectar_separador(texto):
    """Detecta el separador usando csv.Sniffer; cae a heurística por conteo."""
    muestra = texto[:8192]
    try:
        dialect = csv.Sniffer().sniff(muestra, delimiters=[',', ';', '\t', '|'])
        return dialect.delimiter
    except Exception:
        # heurística: el separador con más apariciones en la primera línea
        primera = muestra.splitlines()[0] if muestra.splitlines() else muestra
        conteos = {sep: primera.count(sep) for sep in [';', ',', '\t', '|']}
        sep = max(conteos, key=conteos.get)
        return sep if conteos[sep] > 0 else ','


def leer_tabla(origen, dtype=None):
    """
    Lee una tabla desde:
      - ruta a archivo .csv/.tsv/.txt/.xlsx/.xls/.ods
      - objeto tipo archivo (uploads de Streamlit, BytesIO, etc.)
      - un DataFrame ya cargado (se devuelve tal cual, normalizado)

    Detecta automáticamente separador y encoding para CSV, y la engine
    correcta para Excel. Devuelve un DataFrame con columnas normalizadas.
    """
    # Ya es un DataFrame
    if isinstance(origen, pd.DataFrame):
        return normalizar_columnas(origen)

    # Determinar nombre/extensión y obtener bytes
    nombre = getattr(origen, 'name', origen if isinstance(origen, str) else '')
    ext = os.path.splitext(str(nombre))[1].lower()

    # --- Excel ---
    if ext in ('.xlsx', '.xlsm'):
        df = pd.read_excel(origen, engine='openpyxl', dtype=dtype)
        return normalizar_columnas(df)
    if ext == '.xls':
        df = pd.read_excel(origen, engine='xlrd', dtype=dtype)  # requiere xlrd
        return normalizar_columnas(df)
    if ext == '.ods':
        df = pd.read_excel(origen, engine='odf', dtype=dtype)
        return normalizar_columnas(df)

    # --- CSV / texto: leer bytes una sola vez ---
    if isinstance(origen, str):
        with open(origen, 'rb') as f:
            raw = f.read()
    else:
        raw = origen.read()
        if isinstance(raw, str):
            raw = raw.encode('utf-8')
        try:
            origen.seek(0)  # por si alguien lo reusa
        except Exception:
            pass

    encoding = _detectar_encoding(raw)
    texto = raw.decode(encoding, errors='replace')
    sep = _detectar_separador(texto)

    df = pd.read_csv(io.StringIO(texto), sep=sep, dtype=dtype,
                     quotechar='"', engine='python', on_bad_lines='skip')

    # Si quedó en 1 columna, el separador estaba mal: reintentar con candidatos
    if df.shape[1] == 1:
        for alt in [';', ',', '\t', '|']:
            if alt == sep:
                continue
            try:
                tmp = pd.read_csv(io.StringIO(texto), sep=alt, dtype=dtype,
                                  quotechar='"', engine='python', on_bad_lines='skip')
                if tmp.shape[1] > 1:
                    df = tmp
                    break
            except Exception:
                continue

    return normalizar_columnas(df)

const BASE = '/api';

async function j(method, path, body, headers) {
  const opt = { method };
  if (body) opt.body = body;
  if (headers) opt.headers = headers;
  const r = await fetch(BASE + path, opt);
  if (!r.ok) {
    const t = await r.text();
    throw new Error(t || r.statusText);
  }
  return r.json();
}

export const api = {
  health: () => j('GET', '/health'),
  inspeccionar: (file) => {
    const fd = new FormData(); fd.append('file', file);
    return j('POST', '/inspeccionar', fd);
  },
  subirCalificaciones: (periodo, file, colId, colCal) => {
    const fd = new FormData();
    fd.append('periodo', periodo); fd.append('file', file);
    if (colId) fd.append('col_id_simulacion', colId);
    if (colCal) fd.append('col_calificacion', colCal);
    return j('POST', '/subir/calificaciones', fd);
  },
  subirDatapoints: (file) => {
    const fd = new FormData(); fd.append('file', file);
    return j('POST', '/subir/datapoints', fd);
  },
  estadisticas: (periodo) => j('GET', `/estadisticas/${periodo}`),
  versiones: () => j('GET', '/versiones'),
  detalleModelos: (v) => j('GET', `/modelos/${v}`),
  eda: (periodo, modo) => j('GET', `/eda/${periodo}?modo_seleccion=${modo||'titular'}`),
  getCalifConfig: () => j('GET', '/calificaciones-config'),
  setCalifConfig: (cfg) => j('POST', '/calificaciones-config', JSON.stringify(cfg), {'Content-Type':'application/json'}),
  getReglas: () => j('GET', '/reglas'),
  setReglas: (reglas) => j('POST', '/reglas', JSON.stringify(reglas), {'Content-Type':'application/json'}),
  getPesos: () => j('GET', '/pesos'),
  setPesos: (pesos) => j('POST', '/pesos', JSON.stringify(pesos), {'Content-Type':'application/json'}),
  entrenar: (modo) => { const fd = new FormData(); fd.append('modo_seleccion', modo||'titular'); return j('POST', '/entrenar', fd); },
  predecir: (version, files, modo) => {
    const fd = new FormData(); fd.append('version', version); fd.append('modo_seleccion', modo||'titular');
    files.forEach(f => fd.append('files', f));
    return j('POST', '/predecir', fd);
  },
};

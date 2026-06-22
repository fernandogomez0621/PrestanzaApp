import React, { useState } from 'react';
import { api } from '../api';
import { FileDrop } from '../components/ui';

function CalifUploader({ periodo, showToast }) {
  const [file, setFile] = useState(null);
  const [insp, setInsp] = useState(null);
  const [colId, setColId] = useState('');
  const [colCal, setColCal] = useState('');
  const [loading, setLoading] = useState(false);
  const [ok, setOk] = useState(null);

  const onFile = async (f) => {
    setFile(f); setOk(null);
    try {
      const r = await api.inspeccionar(f);
      setInsp(r);
      // autoselección si los nombres esperados ya están
      setColId(r.columnas.find(c => /sim/i.test(c)) || r.columnas[0] || '');
      setColCal(r.columnas.find(c => /calif|valor/i.test(c)) || r.columnas[1] || '');
    } catch (e) { showToast('Error al leer: ' + e.message); }
  };

  const subir = async () => {
    setLoading(true);
    try {
      const r = await api.subirCalificaciones(periodo, file, colId, colCal);
      setOk(r); showToast(`Calificaciones ${periodo}: ${r.filas} filas cargadas`);
    } catch (e) { showToast('Error: ' + e.message); }
    setLoading(false);
  };

  return (
    <div className="card">
      <h3>Calificaciones {periodo} <span className="tag">{periodo === '6M' ? '6 meses' : '12 meses'}</span></h3>
      <FileDrop label={`Subir calificaciones a ${periodo}`} file={file} onFile={onFile} />
      {insp && (
        <div style={{ marginTop: 16 }}>
          <div className="muted" style={{ fontSize: 12, marginBottom: 10 }}>
            {insp.filas} filas · selecciona qué columna es cuál:
          </div>
          <label className="field">
            <span>Columna de ID de simulación</span>
            <select value={colId} onChange={e => setColId(e.target.value)}>
              {insp.columnas.map(c => <option key={c} value={c}>{c}</option>)}
            </select>
          </label>
          <label className="field">
            <span>Columna del valor de calificación</span>
            <select value={colCal} onChange={e => setColCal(e.target.value)}>
              {insp.columnas.map(c => <option key={c} value={c}>{c}</option>)}
            </select>
          </label>
          <button className="btn" onClick={subir} disabled={loading}>
            {loading ? <span className="spinner" /> : `Cargar ${periodo}`}
          </button>
          {ok && <span className="badge ok" style={{ marginLeft: 12 }}>✓ {ok.filas} filas</span>}
        </div>
      )}
    </div>
  );
}

export default function CargaDatos({ showToast, recargarVersiones, setVersionActiva }) {
  const [dpFile, setDpFile] = useState(null);
  const [dpInfo, setDpInfo] = useState(null);
  const [entrenando, setEntrenando] = useState(false);
  const [modo, setModo] = useState('titular');

  const subirDp = async (f) => {
    setDpFile(f);
    try { const r = await api.subirDatapoints(f); setDpInfo(r); showToast(`Datapoints: ${r.simulaciones} simulaciones`); }
    catch (e) { showToast('Error: ' + e.message); }
  };

  const reentrenar = async () => {
    setEntrenando(true);
    showToast('Entrenando las 4 familias para 6M y 12M… puede tardar');
    try {
      const r = await api.entrenar(modo);
      showToast(`✓ Entrenamiento completo — versión ${r.version}`);
      await recargarVersiones();
      setVersionActiva(r.version);
    } catch (e) { showToast('Error al entrenar: ' + e.message); }
    setEntrenando(false);
  };

  return (
    <div>
      <h2 className="section-title">Carga de datos</h2>
      <p className="section-desc">
        Sube las calificaciones de 6 y 12 meses por separado. Si el encabezado del archivo
        no coincide con lo esperado, selecciona manualmente qué columna corresponde a cada campo.
      </p>

      <div className="split">
        <CalifUploader periodo="6M" showToast={showToast} />
        <CalifUploader periodo="12M" showToast={showToast} />
      </div>

      <div className="card" style={{ marginTop: 20 }}>
        <h3>Datapoints <span className="tag">perfil de deudores</span></h3>
        <FileDrop label="Subir archivo de datapoints" file={dpFile} onFile={subirDp} />
        {dpInfo && (
          <div className="kpi-row" style={{ marginTop: 16 }}>
            <div className="kpi"><div className="label">Filas</div><div className="value small">{dpInfo.filas.toLocaleString()}</div></div>
            <div className="kpi"><div className="label">Simulaciones</div><div className="value small">{dpInfo.simulaciones}</div></div>
          </div>
        )}
      </div>

      <div className="card" style={{ marginTop: 20, borderColor: 'var(--amber)' }}>
        <h3>Reentrenar modelos</h3>
        <p className="muted" style={{ marginBottom: 14, fontSize: 13 }}>
          Procesa los datos cargados, selecciona el deudor principal según el modo elegido, entrena las
          4 familias de modelos para ambos horizontes y guarda todo en una nueva versión fechada.
        </p>
        <label className="field" style={{ maxWidth: 460 }}>
          <span>¿Con qué deudor entrenar?</span>
          <select value={modo} onChange={e => setModo(e.target.value)}>
            <option value="titular">Titular del crédito (id_type_debtor = 1)</option>
            <option value="perfil">Mejor perfil (regla de negocio con los pesos)</option>
          </select>
        </label>
        <button className="btn" onClick={reentrenar} disabled={entrenando}>
          {entrenando ? <><span className="spinner" /> Entrenando…</> : 'Actualizar datos y reentrenar'}
        </button>
      </div>
    </div>
  );
}

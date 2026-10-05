import React, { useState } from 'react';
import { api } from '../../api';
import { Clase } from '../../components/ui';
import { nombre } from './nombres';

const num = (x, d = 2) => (x == null ? '—' : Number(x).toFixed(d).replace('.', ','));

function Barras({ indices }) {
  const items = Object.entries(indices).sort((a, b) => b[1] - a[1]);
  const max = Math.max(...items.map(([, v]) => Math.abs(v)), 1);
  return (
    <div style={{ marginTop: 10 }}>
      {items.map(([k, v]) => (
        <div key={k} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 3 }}>
          <div style={{ width: 150, fontSize: 11 }}>{nombre(k)}</div>
          <div style={{ width: 200, display: 'flex', justifyContent: 'center' }}>
            <div style={{ width: 100, display: 'flex', justifyContent: 'flex-end' }}>
              {v < 0 && <div style={{ height: 8, width: `${(-v / max) * 100}px`, background: 'var(--green)', borderRadius: 2 }} />}
            </div>
            <div style={{ width: 1, background: 'var(--linea-2)' }} />
            <div style={{ width: 100 }}>
              {v > 0 && <div style={{ height: 8, width: `${(v / max) * 100}px`, background: 'var(--red)', borderRadius: 2 }} />}
            </div>
          </div>
          <span className="mono" style={{ fontSize: 10 }}>{v > 0 ? '+' : ''}{num(v)}</span>
        </div>
      ))}
    </div>
  );
}

function Horizonte({ per, r }) {
  if (!r) return (
    <div className="deudor-card"><b>{per}</b> <span className="muted" style={{ fontSize: 12 }}>
      sin datos del crédito suficientes (monto o garantía en 0)</span></div>
  );
  const etiqueta = r.decision.startsWith('No') ? 'No-Buena' : 'Buena';
  return (
    <div className={'deudor-card' + (etiqueta === 'No-Buena' ? ' principal' : '')}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
        <b>{per === '6M' ? '6 meses' : '12 meses'}</b>
        <Clase v={etiqueta} />
        {etiqueta === 'No-Buena' && <span className="muted" style={{ fontSize: 11 }}>revisar con más detalle</span>}
        <span className="muted mono" style={{ fontSize: 11 }}>puntaje {num(r.puntaje)} · umbral {num(r.umbral)} · banda {r.banda}</span>
      </div>
      <Barras indices={r.indices} />
    </div>
  );
}

export default function PrediccionIndices({ versionActiva, showToast }) {
  const [files, setFiles] = useState([]);
  const [res, setRes] = useState(null);
  const [loading, setLoading] = useState(false);

  const predecir = async () => {
    if (!versionActiva) { showToast('Primero selecciona una versión de modelos'); return; }
    setLoading(true); setRes(null);
    try {
      const r = await api.predecir(versionActiva, files, 'titular');
      setRes(r.archivos); showToast(`Predicción lista para ${r.archivos.length} archivo(s)`);
    } catch (e) { showToast('Error: ' + e.message); }
    setLoading(false);
  };

  return (
    <div>
      <h2 className="section-title">Predicción</h2>
      <p className="section-desc">
        Sube uno o varios CSV de datapoints. Para cada simulación se calcula el valor de cada índice, el puntaje (suma de índices),
        la banda y la decisión a 6 y 12 meses. En las barras, rojo empuja hacia el riesgo y verde hacia Buena.
      </p>
      <div className="card" style={{ marginBottom: 20 }}>
        <div className="dropzone" onClick={() => document.getElementById('multi-ind').click()}>
          <input id="multi-ind" type="file" multiple accept=".csv,.xlsx,.xls,.ods" style={{ display: 'none' }}
            onChange={e => setFiles(Array.from(e.target.files))} />
          {files.length
            ? <div><b>✓ {files.length} archivo(s) seleccionado(s)</b><div className="muted" style={{ fontSize: 11, marginTop: 4 }}>{files.map(f => f.name).join(', ').slice(0, 90)}</div></div>
            : <div>Subir uno o varios CSV (puedes seleccionar múltiples)<div className="muted" style={{ fontSize: 11, marginTop: 4 }}>CSV · XLSX · ODS</div></div>}
        </div>
        <div style={{ marginTop: 14, display: 'flex', alignItems: 'center', gap: 14 }}>
          <button className="btn" onClick={predecir} disabled={!files.length || loading}>
            {loading ? <><span className="spinner" /> Prediciendo…</> : 'Predecir'}
          </button>
          <span className="muted" style={{ fontSize: 12 }}>modelos: <b className="mono">{versionActiva || 'ninguno'}</b></span>
        </div>
      </div>
      {res && res.map((arch, i) => (
        <div key={i} style={{ marginBottom: 20 }}>
          <div className="muted mono" style={{ fontSize: 12, marginBottom: 8 }}>📄 {arch.archivo}</div>
          {!arch.ok
            ? <div className="card"><span className="badge no">error</span> {arch.error}</div>
            : arch.resultados.map(sim => (
              <div className="card" key={sim.id_simulacion} style={{ marginBottom: 12 }}>
                <h3>Simulación {sim.id_simulacion}</h3>
                <div className="split" style={{ gap: 14 }}>
                  <Horizonte per="6M" r={sim['6M']} />
                  <Horizonte per="12M" r={sim['12M']} />
                </div>
              </div>
            ))}
        </div>
      ))}
    </div>
  );
}

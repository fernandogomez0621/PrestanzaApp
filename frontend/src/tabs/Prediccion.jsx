import React, { useState } from 'react';
import { api } from '../api';
import { Clase, numeroALetra } from '../components/ui';

const MODELOS = [
  { id: 'nueve_clases', label: '9 clases' },
  { id: 'tres_clases', label: '3 clases' },
  { id: 'buena_no_buena', label: 'Buena/No-Buena' },
  { id: 'buena_no_buena_shap', label: 'B/NoB (SHAP)' },
];

function PredCell({ pred, modelo, letras }) {
  const v = pred?.[modelo];
  if (v == null) return <span className="muted">—</span>;
  if (typeof v === 'object')
    return <span><Clase v={v.etiqueta} /> <span className="muted mono" style={{ fontSize: 11 }}>{(v.prob_buena * 100).toFixed(0)}%</span></span>;
  if (modelo === 'tres_clases') return <Clase v={v} />;
  if (modelo === 'nueve_clases') {
    const letra = numeroALetra(v, letras);
    return <span className="mono">{v}{letra ? <span className="muted"> · {letra}</span> : ''}</span>;
  }
  return <span className="mono">{v}</span>;
}

function Desglose({ d }) {
  const [abierto, setAbierto] = useState(false);
  return (
    <div style={{ marginTop: 8 }}>
      <button className="btn ghost" style={{ fontSize: 11, padding: '5px 10px' }} onClick={() => setAbierto(!abierto)}>
        {abierto ? '▾' : '▸'} puntaje de perfil: {d.puntaje_total} {d.es_deudor_principal && '★'}
      </button>
      {abierto && (
        <div className="table-wrap" style={{ maxHeight: 'none', marginTop: 8 }}>
          <table>
            <thead><tr><th>Factor</th><th>Puntaje base</th><th>Peso</th><th>Aporte</th></tr></thead>
            <tbody>
              {d.puntaje_desglose.map((f, i) => (
                <tr key={i}>
                  <td style={{ fontFamily: 'var(--sans)' }}>{f.factor}</td>
                  <td>{f.puntaje_base}</td>
                  <td>{(f.peso * 100).toFixed(0)}%</td>
                  <td style={{ color: f.aporte < 0 ? 'var(--red)' : 'var(--green)' }}>{f.aporte >= 0 ? '+' : ''}{f.aporte}</td>
                </tr>
              ))}
              <tr style={{ fontWeight: 700 }}>
                <td colSpan={3}>TOTAL</td><td style={{ color: 'var(--amber)' }}>{d.puntaje_total}</td>
              </tr>
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function Deudor({ d, letras }) {
  return (
    <div className={'deudor-card' + (d.es_deudor_principal ? ' principal' : '')}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
        <b>{d.es_deudor_principal ? '★ Deudor principal' : 'Codeudor'}</b>
        <span className="muted mono" style={{ fontSize: 11 }}>id {d.id_usuario} · tipo {d.tipo ?? '—'} · score {d.credit_score ?? '—'} · {d.educacion}</span>
      </div>
      <div className="table-wrap" style={{ maxHeight: 'none' }}>
        <table>
          <thead><tr><th>Horizonte</th>{MODELOS.map(m => <th key={m.id}>{m.label}</th>)}</tr></thead>
          <tbody>
            {['6M', '12M'].map(per => (
              <tr key={per}>
                <td><b>{per}</b></td>
                {MODELOS.map(m => <td key={m.id}><PredCell pred={d.predicciones[per]} modelo={m.id} letras={letras} /></td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <Desglose d={d} />
    </div>
  );
}

export default function Prediccion({ versionActiva, showToast, letras }) {
  const [files, setFiles] = useState([]);
  const [modo, setModo] = useState('titular');
  const [res, setRes] = useState(null);
  const [loading, setLoading] = useState(false);

  const predecir = async () => {
    if (!versionActiva) { showToast('Primero selecciona una versión de modelos'); return; }
    setLoading(true); setRes(null);
    try {
      const r = await api.predecir(versionActiva, files, modo);
      setRes(r.archivos); showToast(`Predicción lista para ${r.archivos.length} archivo(s)`);
    } catch (e) { showToast('Error: ' + e.message); }
    setLoading(false);
  };

  return (
    <div>
      <h2 className="section-title">Predicción</h2>
      <p className="section-desc">
        Sube uno o varios CSV de datapoints. Para cada simulación se marca el deudor principal (según el
        modo elegido), se muestra el desglose de su puntaje de perfil, y la predicción de las 4 familias
        de modelos en 6 y 12 meses.
      </p>

      <div className="card" style={{ marginBottom: 20 }}>
        <div className="dropzone" onClick={() => document.getElementById('multi').click()}>
          <input id="multi" type="file" multiple accept=".csv,.xlsx,.xls,.ods" style={{ display: 'none' }}
            onChange={e => setFiles(Array.from(e.target.files))} />
          {files.length
            ? <div><b>✓ {files.length} archivo(s) seleccionado(s)</b><div className="muted" style={{ fontSize: 11, marginTop: 4 }}>{files.map(f => f.name).join(', ').slice(0, 90)}</div></div>
            : <div>Subir uno o varios CSV (puedes seleccionar múltiples)<div className="muted" style={{ fontSize: 11, marginTop: 4 }}>CSV · XLSX · ODS</div></div>}
        </div>
        <div style={{ marginTop: 14, display: 'flex', alignItems: 'flex-end', gap: 14, flexWrap: 'wrap' }}>
          <label className="field" style={{ marginBottom: 0, minWidth: 280 }}>
            <span>Modo de deudor principal</span>
            <select value={modo} onChange={e => setModo(e.target.value)}>
              <option value="titular">Titular (id_type_debtor = 1)</option>
              <option value="perfil">Mejor perfil (pesos)</option>
            </select>
          </label>
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
                <h3>Simulación {sim.id_simulacion} <span className="tag">{sim.deudores.length} deudor(es)</span></h3>
                {sim.deudores.map((d, j) => <Deudor key={j} d={d} letras={letras} />)}
              </div>
            ))}
        </div>
      ))}
    </div>
  );
}

import React, { useState, useEffect } from 'react';
import { api } from '../../api';
import { nombre, fechaVersion } from './nombres';

const pct = (x) => (x == null ? '—' : `${(x * 100).toFixed(0)}%`);
const num = (x, d = 2) => (x == null ? '—' : Number(x).toFixed(d).replace('.', ','));

function Matriz({ m, clases }) {
  const max = Math.max(...m.flat(), 1);
  return (
    <table style={{ fontSize: 11 }}>
      <thead>
        <tr><th style={{ fontSize: 10 }}>real ↓ / predicho →</th>{clases.map(c => <th key={c} style={{ textAlign: 'center' }}>{c}</th>)}</tr>
      </thead>
      <tbody>
        {m.map((fila, i) => (
          <tr key={i}>
            <td style={{ fontWeight: 600 }}>{clases[i]}</td>
            {fila.map((v, j) => (
              <td key={j} style={{ textAlign: 'center', fontWeight: i === j ? 700 : 400,
                background: i === j ? `rgba(111,154,94,${0.15 + (v / max) * 0.5})` : `rgba(197,99,78,${(v / max) * 0.45})` }}>{num(v, 1)}</td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function Panel({ periodo, d }) {
  if (!d) return <div className="card"><h3>{periodo}</h3><p className="muted">Sin modelo para este horizonte.</p></div>;
  const maxAuc = Math.max(...d.por_indice.map(x => x.auc_solo), 0.6);
  return (
    <div className="card">
      <h3>{periodo === '6M' ? '6 meses' : '12 meses'} <span className="tag">n = {d.n} · No-Buena = {d.no_buena}</span>
        <span className="tag">Buena ≥ {d.corte_buena ?? 9}</span></h3>
      <div className="kpi-row">
        <div className="kpi"><div className="label">AUC</div><div className="value small">{num(d.auc)}</div></div>
        <div className="kpi"><div className="label">F1-macro</div><div className="value small">{num(d.f1_macro)}</div></div>
        <div className="kpi"><div className="label">Exactitud bal.</div><div className="value small">{num(d.balanced_accuracy)}</div></div>
      </div>
      <p className="muted" style={{ fontSize: 11, marginBottom: 12 }}>
        Validación cruzada 5 × {d.repeticiones} (créditos no vistos). Umbral del puntaje: <b className="mono">{num(d.umbral, 3)}</b> ·
        se marca el {pct(d.frac_marcado)} de mayor puntaje (1,5 × la tasa de No-Buena de entrenamiento).
        {periodo === '6M' && ' Ajustado sin transitorios (No-Buena a 6M que son Buena a 12M).'}
      </p>
      <div className="split" style={{ gap: 14 }}>
        <div>
          <div className="muted" style={{ fontSize: 11, marginBottom: 6 }}>Matriz de confusión (promedio por repetición)</div>
          <Matriz m={d.matriz_confusion} clases={d.clases} />
          <div className="muted" style={{ fontSize: 11, margin: '12px 0 6px' }}>Métricas por clase</div>
          <table>
            <thead><tr><th>Clase</th><th>Precisión</th><th>Recall</th><th>F1</th></tr></thead>
            <tbody>{d.por_clase.map(c => (
              <tr key={c.clase}><td>{c.clase}</td><td className="mono">{num(c.precision)}</td><td className="mono">{num(c.recall)}</td><td className="mono">{num(c.f1)}</td></tr>
            ))}</tbody>
          </table>
        </div>
        <div>
          <div className="muted" style={{ fontSize: 11, marginBottom: 6 }}>% No-Buena por banda de puntaje</div>
          <table>
            <thead><tr><th>Banda</th><th>Operaciones</th><th>% No-Buena</th></tr></thead>
            <tbody>{d.bandas.map(b => (
              <tr key={b.banda}><td>{b.banda}</td><td className="mono">{num(b.n_promedio, 0)}</td><td className="mono">{pct(b.tasa_no_buena)}</td></tr>
            ))}</tbody>
          </table>
          <div className="muted" style={{ fontSize: 11, margin: '12px 0 6px' }}>AUC de cada índice por sí solo (0,5 = azar)</div>
          {d.por_indice.map(x => (
            <div key={x.indice} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
              <div style={{ width: 150, fontSize: 11 }}>{nombre(x.indice)}</div>
              <div style={{ height: 7, width: `${Math.max(0, (x.auc_solo - 0.4) / (maxAuc - 0.4)) * 120}px`, minWidth: 2,
                background: x.auc_solo > 0.55 ? 'var(--amber)' : 'var(--muted-2)', borderRadius: 2 }} />
              <span className="mono" style={{ fontSize: 10 }}>{num(x.auc_solo)}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default function ModelosIndices({ versiones, versionActiva, setVersionActiva, recargarVersiones }) {
  const [detalle, setDetalle] = useState(null);
  const [cargando, setCargando] = useState(false);
  useEffect(() => {
    if (!versionActiva) return;
    setCargando(true);
    api.detalleModelos(versionActiva).then(d => { setDetalle(d); setCargando(false); }).catch(() => { setDetalle(null); setCargando(false); });
  }, [versionActiva]);

  return (
    <div>
      <h2 className="section-title">Modelos & versiones</h2>
      <p className="section-desc">
        Cada modelo suma índices de riesgo con signo fijo (sin coeficientes estimados). Las métricas vienen de validación cruzada
        sobre los créditos calificados, con la misma regla de decisión que se usa al predecir.
      </p>
      <div className="card" style={{ marginBottom: 20 }}>
        <h3>Versión activa</h3>
        {versiones.length === 0
          ? <p className="muted">Aún no hay modelos. Si acaba de arrancar, el primer entrenamiento tarda unos segundos;
              <button className="btn ghost" style={{ marginLeft: 8, fontSize: 11, padding: '4px 10px' }} onClick={recargarVersiones}>actualizar</button></p>
          : (
            <label className="field" style={{ maxWidth: 380 }}>
              <span>Seleccionar entrenamiento por fecha</span>
              <select value={versionActiva || ''} onChange={e => setVersionActiva(e.target.value)}>
                {versiones.map(v => <option key={v.version} value={v.version}>{fechaVersion(v.version)} · Buena ≥ {v.corte_buena ?? 9}</option>)}
              </select>
            </label>
          )}
      </div>
      {cargando && <div className="card"><span className="spinner" /> cargando modelos…</div>}
      {detalle && !cargando && (
        <div className="split">
          <Panel periodo="6M" d={detalle['6M']} />
          <Panel periodo="12M" d={detalle['12M']} />
        </div>
      )}
    </div>
  );
}

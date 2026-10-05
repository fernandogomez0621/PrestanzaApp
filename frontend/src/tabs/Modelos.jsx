import React, { useState, useEffect } from 'react';
import { api } from '../api';

const FAMILIAS = [
  { id: 'nueve_clases', nombre: '9 clases (calificación 0–10)', desc: 'Predice la letra exacta' },
  { id: 'tres_clases', nombre: '3 clases (Buena / Media / Riesgo)', desc: 'Agrupación ordinal' },
  { id: 'buena_no_buena', nombre: 'Buena vs No-Buena', desc: 'Todas las variables' },
  { id: 'buena_no_buena_shap', nombre: 'Buena vs No-Buena (SHAP con sentido)', desc: 'Solo variables con dirección lógica' },
];

// Matriz de confusión como heatmap
function MatrizConfusion({ matriz, clases }) {
  if (!matriz || !matriz.length) return null;
  const max = Math.max(...matriz.flat(), 1);
  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ fontSize: 11 }}>
        <thead>
          <tr>
            <th style={{ background: 'transparent' }}></th>
            <th colSpan={clases.length} style={{ textAlign: 'center', color: 'var(--muted)' }}>predicho →</th>
          </tr>
          <tr><th style={{ fontSize: 10 }}>real ↓</th>{clases.map(c => <th key={c} style={{ textAlign: 'center' }}>{c}</th>)}</tr>
        </thead>
        <tbody>
          {matriz.map((fila, i) => (
            <tr key={i}>
              <td style={{ fontWeight: 600, color: 'var(--paper-dim)' }}>{clases[i]}</td>
              {fila.map((v, j) => {
                const diag = i === j;
                const intensidad = v / max;
                return (
                  <td key={j} style={{
                    textAlign: 'center', fontWeight: diag ? 700 : 400,
                    background: diag ? `rgba(111,154,94,${0.15 + intensidad * 0.5})` : `rgba(197,99,78,${intensidad * 0.45})`,
                    color: 'var(--paper)',
                  }}>{v}</td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function DetalleFamilia({ comparacion }) {
  const [algoSel, setAlgoSel] = useState(0);
  if (!comparacion || !comparacion.length) return <p className="muted">Sin comparación disponible.</p>;
  const valid = comparacion.filter(r => !r.error);
  const r = valid[algoSel] || valid[0];
  if (!r) return <p className="muted">Sin resultados.</p>;

  return (
    <div>
      {/* tabla comparativa de algoritmos */}
      <div className="table-wrap" style={{ marginBottom: 14 }}>
        <table>
          <thead><tr><th>Algoritmo</th><th>F1-macro</th><th>Accuracy</th><th>Bal. acc</th><th></th></tr></thead>
          <tbody>
            {valid.map((a, i) => (
              <tr key={i} style={i === algoSel ? { background: 'var(--panel-2)' } : {}}>
                <td>{a.algoritmo} {i === 0 && <span className="badge ok" style={{ fontSize: 9 }}>mejor</span>}</td>
                <td className="mono" style={{ color: 'var(--amber)' }}>{a.f1_macro}</td>
                <td className="mono">{a.accuracy}</td>
                <td className="mono">{a.balanced_accuracy}</td>
                <td><button className="btn ghost" style={{ fontSize: 10, padding: '3px 8px' }} onClick={() => setAlgoSel(i)}>ver</button></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="split" style={{ gap: 14 }}>
        {/* matriz de confusión */}
        <div>
          <div className="muted" style={{ fontSize: 11, marginBottom: 6 }}>Matriz de confusión — {r.algoritmo} (validación cruzada)</div>
          <MatrizConfusion matriz={r.matriz_confusion} clases={r.clases} />
        </div>
        {/* métricas por clase */}
        <div>
          <div className="muted" style={{ fontSize: 11, marginBottom: 6 }}>Métricas por clase</div>
          <div className="table-wrap">
            <table>
              <thead><tr><th>Clase</th><th>Precisión</th><th>Recall</th><th>F1</th><th>n</th></tr></thead>
              <tbody>
                {r.por_clase.map((c, i) => (
                  <tr key={i}><td>{c.clase}</td><td className="mono">{c.precision}</td>
                    <td className="mono">{c.recall}</td><td className="mono">{c.f1}</td><td className="mono muted">{c.soporte}</td></tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* importancia de variables */}
      {r.importancia && r.importancia.length > 0 && (
        <div style={{ marginTop: 14 }}>
          <div className="muted" style={{ fontSize: 11, marginBottom: 6 }}>Importancia de variables — {r.algoritmo}</div>
          {r.importancia.slice(0, 8).map((v, i) => {
            const max = r.importancia[0].importancia || 1;
            return (
              <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                <div style={{ width: 220, fontSize: 11, fontFamily: 'var(--sans)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                  {v.variable} {v.direccion && <span className="muted">({v.direccion})</span>}
                </div>
                <div style={{ height: 7, width: `${(v.importancia / max) * 160}px`, minWidth: 2, background: 'var(--amber)', borderRadius: 2 }} />
                <span className="mono" style={{ fontSize: 10 }}>{v.importancia}</span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

function PanelModelos({ periodo, detalle }) {
  const [fam, setFam] = useState('tres_clases');
  if (!detalle) return <div className="card"><h3>{periodo}</h3><p className="muted">Sin modelos para este horizonte.</p></div>;
  const comp = (detalle.comparacion || {})[fam];
  const met = detalle.metricas[fam];

  return (
    <div className="card">
      <h3>{periodo === '6M' ? '6 meses' : '12 meses'} <span className="tag">n = {detalle.n}</span></h3>
      <div className="filterbar" style={{ marginBottom: 14 }}>
        <label className="field" style={{ marginBottom: 0, minWidth: 320 }}>
          <span>Familia de modelo</span>
          <select value={fam} onChange={e => setFam(e.target.value)}>
            {FAMILIAS.map(f => <option key={f.id} value={f.id}>{f.nombre}</option>)}
          </select>
        </label>
        {met && met.mejor_algoritmo && (
          <div style={{ fontSize: 12 }}>
            <span className="muted">mejor algoritmo:</span> <b style={{ color: 'var(--amber)' }}>{met.mejor_algoritmo}</b>
          </div>
        )}
      </div>
      <DetalleFamilia comparacion={comp} />
    </div>
  );
}

export default function Modelos({ versiones, versionActiva, setVersionActiva }) {
  const [detalle, setDetalle] = useState(null);
  const [cargando, setCargando] = useState(false);
  useEffect(() => {
    if (versionActiva) { setCargando(true); api.detalleModelos(versionActiva).then(d => { setDetalle(d); setCargando(false); }).catch(() => { setDetalle(null); setCargando(false); }); }
  }, [versionActiva]);

  return (
    <div>
      <h2 className="section-title">Modelos & versiones</h2>
      <p className="section-desc">
        Cada familia compara varios algoritmos (Regresión Logística, Random Forest, Gradient Boosting, KNN)
        y se queda con el mejor por F1-macro. Aquí ves la comparación, la matriz de confusión, las métricas
        por clase y la importancia de variables de cada uno.
      </p>

      <div className="card" style={{ marginBottom: 20 }}>
        <h3>Versión activa</h3>
        {versiones.length === 0
          ? <p className="muted">Aún no hay modelos entrenados. Ve a “Carga de datos” y reentrena.</p>
          : (
            <label className="field" style={{ maxWidth: 380 }}>
              <span>Seleccionar entrenamiento por fecha</span>
              <select value={versionActiva || ''} onChange={e => setVersionActiva(e.target.value)}>
                {versiones.map(v => {
                  const f = v.version;
                  const fecha = `${f.slice(0,4)}-${f.slice(4,6)}-${f.slice(6,8)} ${f.slice(9,11)}:${f.slice(11,13)}`;
                  return <option key={v.version} value={v.version}>{fecha}{v.modo_seleccion ? ` · ${v.modo_seleccion}` : ''}</option>;
                })}
              </select>
            </label>
          )}
      </div>

      {cargando && <div className="card"><span className="spinner" /> cargando modelos…</div>}
      {detalle && !cargando && (
        <div className="split">
          <PanelModelos periodo="6M" detalle={detalle['6M']} />
          <PanelModelos periodo="12M" detalle={detalle['12M']} />
        </div>
      )}
    </div>
  );
}

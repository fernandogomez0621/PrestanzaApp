import React, { useState, useEffect } from 'react';
import { api } from '../api';

const COLOR_CLASE = { Buena: 'var(--green)', Media: 'var(--amber)', Riesgo: 'var(--red)' };

export default function CalifConfig({ showToast }) {
  const [data, setData] = useState(null);
  const [letras, setLetras] = useState([]);
  const [cortes, setCortes] = useState({});
  const [guardando, setGuardando] = useState(false);

  useEffect(() => {
    api.getCalifConfig().then(r => { setData(r); setLetras(r.config.letras); setCortes(r.config.cortes); }).catch(() => {});
  }, []);

  if (!data) return <div className="card"><span className="spinner" /> cargando…</div>;

  const orden = data.orden_clases;

  // a qué clase cae un número según los cortes
  const claseDe = (num) => {
    const v = Number(num);
    for (const c of orden) if (cortes[c] != null && v >= cortes[c]) return c;
    return orden[orden.length - 1];
  };

  // editar un rango de letra
  const setRango = (i, campo, val) => {
    setLetras(letras.map((r, j) => j === i
      ? { ...r, [campo]: campo === 'letra' ? val : (val === '' ? null : Number(val)) }
      : r));
  };
  const agregar = () => setLetras([...letras, { letra: '', min: null, max: null }]);
  const quitar = (i) => setLetras(letras.filter((_, j) => j !== i));

  const guardar = async () => {
    setGuardando(true);
    try { await api.setCalifConfig({ letras, cortes }); showToast('Mapeo y cortes guardados. Reentrena para que los cortes afecten los modelos.'); }
    catch (e) { showToast('Error: ' + e.message); }
    setGuardando(false);
  };
  const restaurar = () => { setLetras(data.default.letras); setCortes(data.default.cortes); };

  return (
    <div>
      <h2 className="section-title">Calificaciones: letras y clases</h2>
      <p className="section-desc">
        Define el rango de calificación de cada letra (AAA, AA, A…) y qué números caen en cada clase
        (Buena / Media / Riesgo). El rango de letra es un intervalo <b>[desde, hasta)</b>: incluye el
        “desde” y excluye el “hasta”. Los cortes de clase se usan al entrenar, así que <b>tras cambiarlos
        hay que reentrenar</b>.
      </p>

      <div className="split">
        {/* Rangos de letras */}
        <div className="card">
          <h3>Rangos de letra <span className="tag">[desde, hasta)</span></h3>
          <p className="muted" style={{ fontSize: 11, marginBottom: 12 }}>
            Cada letra cubre un rango. Deja “desde” vacío para −∞ y “hasta” vacío para +∞.
            Así cualquier número (incluido 3 o 7) cae en su letra.
          </p>
          <div className="table-wrap" style={{ maxHeight: 420 }}>
            <table>
              <thead><tr><th>Letra</th><th>Desde</th><th>Hasta</th><th></th></tr></thead>
              <tbody>
                {letras.map((r, i) => (
                  <tr key={i}>
                    <td><input type="text" value={r.letra} style={{ width: 70 }}
                      onChange={e => setRango(i, 'letra', e.target.value)} /></td>
                    <td><input type="number" step="0.5" value={r.min ?? ''} placeholder="−∞" style={{ width: 80 }}
                      onChange={e => setRango(i, 'min', e.target.value)} /></td>
                    <td><input type="number" step="0.5" value={r.max ?? ''} placeholder="+∞" style={{ width: 80 }}
                      onChange={e => setRango(i, 'max', e.target.value)} /></td>
                    <td><button className="btn ghost" style={{ padding: '3px 8px', fontSize: 11 }} onClick={() => quitar(i)}>✕</button></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <button className="btn ghost" style={{ marginTop: 8, fontSize: 11, padding: '5px 10px' }} onClick={agregar}>+ letra</button>
        </div>

        {/* Cortes de clases */}
        <div className="card">
          <h3>Cortes de clases <span className="tag">umbral mínimo</span></h3>
          <p className="muted" style={{ fontSize: 12, marginBottom: 14 }}>
            Cada clase aplica si la calificación es <b>mayor o igual</b> a su umbral. Se evalúa de mejor a
            peor: Buena primero, luego Media, y el resto es Riesgo.
          </p>
          {orden.map(c => (
            <div key={c} style={{ display: 'flex', alignItems: 'center', gap: 14, marginBottom: 14 }}>
              <div style={{ flex: 1 }}>
                <span className="badge" style={{ color: COLOR_CLASE[c], borderColor: COLOR_CLASE[c] }}>{c}</span>
                <span className="muted" style={{ fontSize: 12, marginLeft: 8 }}>calificación ≥</span>
              </div>
              <input type="number" value={cortes[c] ?? 0} step="1" style={{ width: 90 }}
                disabled={c === orden[orden.length - 1]}
                onChange={e => setCortes({ ...cortes, [c]: Number(e.target.value) })} />
            </div>
          ))}
          <p className="muted" style={{ fontSize: 11, marginTop: 6 }}>
            (La última clase, {orden[orden.length - 1]}, recibe todo lo que no entra en las anteriores.)
          </p>

          {/* vista previa: qué letra y clase recibe cada número 0-10 */}
          <div style={{ marginTop: 16 }}>
            <div className="muted" style={{ fontSize: 11, marginBottom: 6 }}>Vista previa (número → letra → clase):</div>
            <div className="table-wrap" style={{ maxHeight: 200 }}>
              <table>
                <thead><tr><th>Núm.</th><th>Letra</th><th>Clase</th></tr></thead>
                <tbody>
                  {[10,9,8,7,6,5,4,3,2,1,0].map(n => {
                    const letra = (letras.find(r => (r.min == null || n >= r.min) && (r.max == null || n < r.max)) || {}).letra || '—';
                    return <tr key={n}><td className="mono">{n}</td><td>{letra}</td>
                      <td><span className="badge" style={{ color: COLOR_CLASE[claseDe(n)], borderColor: COLOR_CLASE[claseDe(n)] }}>{claseDe(n)}</span></td></tr>;
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>

      <div style={{ display: 'flex', gap: 12, marginTop: 20 }}>
        <button className="btn" onClick={guardar} disabled={guardando}>
          {guardando ? <span className="spinner" /> : 'Guardar'}
        </button>
        <button className="btn ghost" onClick={restaurar}>Restaurar por defecto</button>
      </div>
    </div>
  );
}

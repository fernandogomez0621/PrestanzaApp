import React, { useState, useEffect } from 'react';
import { api } from '../api';

// --- Editor de escalones de un factor numérico ---
function EditorEscalones({ nombre, label, escalones, neg, onChange }) {
  const set = (i, campo, val) => {
    const copia = escalones.map((e, j) => j === i ? { ...e, [campo]: val === '' ? null : Number(val) } : e);
    onChange(copia);
  };
  const agregar = () => onChange([...escalones, { min: null, max: null, valor: 0 }]);
  const quitar = (i) => onChange(escalones.filter((_, j) => j !== i));

  return (
    <div style={{ marginBottom: 18, paddingBottom: 14, borderBottom: '1px solid var(--line)' }}>
      <div style={{ fontWeight: 600, fontSize: 13, marginBottom: 8 }}>
        {label} {neg && <span className="badge no" style={{ fontSize: 9 }}>resta</span>}
        <span className="muted" style={{ fontSize: 11, marginLeft: 8, fontWeight: 400 }}>si el valor cae en el rango → da ese puntaje</span>
      </div>
      <div className="table-wrap" style={{ maxHeight: 'none' }}>
        <table>
          <thead><tr><th>Desde (min)</th><th>Hasta (max)</th><th>Puntaje</th><th></th></tr></thead>
          <tbody>
            {escalones.map((e, i) => (
              <tr key={i}>
                <td><input type="number" value={e.min ?? ''} placeholder="−∞"
                  onChange={ev => set(i, 'min', ev.target.value)} style={{ width: 90 }} /></td>
                <td><input type="number" value={e.max ?? ''} placeholder="+∞"
                  onChange={ev => set(i, 'max', ev.target.value)} style={{ width: 90 }} /></td>
                <td><input type="number" value={e.valor} step="0.5"
                  onChange={ev => set(i, 'valor', Number(ev.target.value))} style={{ width: 70 }} /></td>
                <td><button className="btn ghost" style={{ padding: '3px 8px', fontSize: 11 }} onClick={() => quitar(i)}>✕</button></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <button className="btn ghost" style={{ marginTop: 8, fontSize: 11, padding: '5px 10px' }} onClick={agregar}>+ escalón</button>
    </div>
  );
}

// --- Editor de categorías (educación) ---
function EditorCategorias({ mapa, onChange }) {
  const set = (k, v) => onChange({ ...mapa, [k]: Number(v) });
  return (
    <div style={{ marginBottom: 18 }}>
      <div style={{ fontWeight: 600, fontSize: 13, marginBottom: 8 }}>Nivel de educación
        <span className="muted" style={{ fontSize: 11, marginLeft: 8, fontWeight: 400 }}>cada nivel → su puntaje</span>
      </div>
      {Object.keys(mapa).filter(k => k !== '_default').map(k => (
        <div key={k} style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 6 }}>
          <div style={{ flex: 1, fontSize: 12, fontFamily: 'var(--sans)' }}>{k}</div>
          <input type="number" value={mapa[k]} step="1" onChange={e => set(k, e.target.value)} style={{ width: 70 }} />
        </div>
      ))}
    </div>
  );
}

export default function Pesos({ showToast }) {
  const [data, setData] = useState(null);
  const [valores, setValores] = useState({});
  const [reglas, setReglas] = useState(null);
  const [guardando, setGuardando] = useState(false);
  const [verReglas, setVerReglas] = useState(false);

  useEffect(() => {
    api.getPesos().then(r => { setData(r); setValores(r.pesos); }).catch(() => {});
    api.getReglas().then(r => setReglas(r)).catch(() => {});
  }, []);

  if (!data || !reglas) return <div className="card"><span className="spinner" /> cargando…</div>;

  const total = Object.values(valores).reduce((a, b) => a + Number(b), 0);
  const set = (k, v) => setValores({ ...valores, [k]: v });

  const guardar = async () => {
    setGuardando(true);
    try {
      await api.setPesos(valores);
      await api.setReglas(reglas.reglas);
      showToast('Pesos y subcriterios guardados. Se usan en modo “mejor perfil”.');
    } catch (e) { showToast('Error: ' + e.message); }
    setGuardando(false);
  };
  const restaurar = () => { setValores(data.default); setReglas({ ...reglas, reglas: JSON.parse(JSON.stringify(reglas.default)) }); };

  const setEscalones = (factor, nuevos) => {
    setReglas({ ...reglas, reglas: { ...reglas.reglas, escalones: { ...reglas.reglas.escalones, [factor]: nuevos } } });
  };
  const setCategorias = (nuevo) => {
    setReglas({ ...reglas, reglas: { ...reglas.reglas, categorias: { ...reglas.reglas.categorias, educacion: nuevo } } });
  };

  const NEG = data.negativos;

  return (
    <div>
      <h2 className="section-title">Pesos y subcriterios del deudor principal</h2>
      <p className="section-desc">
        Define cuánto pesa cada factor (los %) y, dentro de cada uno, cómo se convierte el valor crudo
        en un puntaje (los escalones). Se aplican cuando entrenas o predices en <b>modo “mejor perfil”</b>
        y en el desglose de Predicción.
      </p>

      <div className="card" style={{ maxWidth: 760, marginBottom: 20 }}>
        <h3>Pesos por factor <span className="tag">suman {(total * 100).toFixed(0)}%</span></h3>
        {Object.keys(data.labels).map(k => (
          <div key={k} style={{ display: 'flex', alignItems: 'center', gap: 14, marginBottom: 12 }}>
            <div style={{ flex: 1, fontSize: 13 }}>
              {data.labels[k]} {NEG.includes(k) && <span className="badge no" style={{ fontSize: 9 }}>resta</span>}
            </div>
            <input type="range" min="0" max="0.4" step="0.01" value={valores[k] ?? 0}
              onChange={e => set(k, parseFloat(e.target.value))} style={{ flex: 1 }} />
            <div className="mono" style={{ width: 52, textAlign: 'right', color: 'var(--amber)' }}>{((valores[k] ?? 0) * 100).toFixed(0)}%</div>
          </div>
        ))}
      </div>

      <div className="card" style={{ maxWidth: 760, marginBottom: 20 }}>
        <h3 style={{ cursor: 'pointer' }} onClick={() => setVerReglas(!verReglas)}>
          {verReglas ? '▾' : '▸'} Subcriterios (cómo se calcula el puntaje de cada factor)
        </h3>
        {verReglas && (
          <div style={{ marginTop: 10 }}>
            <EditorCategorias mapa={reglas.reglas.categorias.educacion} onChange={setCategorias} />
            {Object.keys(reglas.reglas.escalones).map(factor => (
              <EditorEscalones key={factor} nombre={factor}
                label={reglas.labels[factor] || factor}
                escalones={reglas.reglas.escalones[factor]}
                neg={reglas.negativos.includes(factor)}
                onChange={(nuevos) => setEscalones(factor, nuevos)} />
            ))}
          </div>
        )}
      </div>

      <div style={{ display: 'flex', gap: 12, maxWidth: 760 }}>
        <button className="btn" onClick={guardar} disabled={guardando}>
          {guardando ? <span className="spinner" /> : 'Guardar pesos y subcriterios'}
        </button>
        <button className="btn ghost" onClick={restaurar}>Restaurar valores por defecto</button>
      </div>
      <p className="muted" style={{ fontSize: 11, marginTop: 12, maxWidth: 760 }}>
        Nota: esto define cómo se elige el <b>deudor principal</b> (regla de negocio). Tras guardar, reentrena
        en modo “mejor perfil” para que el cambio afecte qué deudor se usa.
      </p>
    </div>
  );
}

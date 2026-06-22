import React, { useState, useEffect } from 'react';
import { api } from '../api';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, PieChart, Pie, Legend } from 'recharts';

const COLOR_CLASE = { Buena: '#10b981', Media: '#d9a441', Riesgo: '#b71f1f' };
const PALETA = ['#d9a441', '#10b981', '#6d8aa8', '#b71f1f', '#b08968', '#64748b', '#a8826d', '#7d9a8a', '#9a6d8a'];

// arma los datos del histograma para una variable (total + por clase)
function histData(data, v) {
  const h = data.histogramas && data.histogramas[v];
  if (!h) return [];
  return h.centros.map((centro, i) => ({
    centro,
    total: h.total[i],
    Buena: h.por_clase.Buena ? h.por_clase.Buena[i] : 0,
    Media: h.por_clase.Media ? h.por_clase.Media[i] : 0,
    Riesgo: h.por_clase.Riesgo ? h.por_clase.Riesgo[i] : 0,
  }));
}

// Barra de separación (eta2) y correlación
function FilaResumen({ r }) {
  const corr = r.spearman ?? 0;
  return (
    <tr>
      <td style={{ fontFamily: 'var(--sans)', fontSize: 12 }}>{r.variable}</td>
      <td>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <div style={{ height: 7, width: `${r.eta2 * 200}px`, minWidth: 2, background: 'var(--amber)', borderRadius: 2 }} />
          <span style={{ fontSize: 11 }}>{r.eta2}</span>
        </div>
      </td>
      <td style={{ color: corr > 0 ? 'var(--green)' : 'var(--red)' }}>{corr > 0 ? '+' : ''}{corr}</td>
      <td>{r.significativa ? <span className="badge ok">sí</span> : <span className="muted">no</span>}</td>
      <td className="mono" style={{ fontSize: 11 }}>{r.media_Buena ?? '—'}</td>
      <td className="mono" style={{ fontSize: 11 }}>{r.media_Media ?? '—'}</td>
      <td className="mono" style={{ fontSize: 11 }}>{r.media_Riesgo ?? '—'}</td>
    </tr>
  );
}

// Mini-boxplot por clase de una variable
function Boxplot({ nombre, datos }) {
  const clases = Object.keys(datos);
  if (!clases.length) return null;
  const all = clases.flatMap(c => [datos[c].min, datos[c].max]);
  const lo = Math.min(...all), hi = Math.max(...all);
  const rango = hi - lo || 1;
  const px = (v) => ((v - lo) / rango) * 100;
  return (
    <div style={{ marginBottom: 16 }}>
      <div style={{ fontSize: 12, fontFamily: 'var(--sans)', marginBottom: 6 }}>{nombre}</div>
      {clases.map(c => {
        const d = datos[c];
        return (
          <div key={c} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
            <span className="badge" style={{ width: 56, fontSize: 9, color: COLOR_CLASE[c], borderColor: COLOR_CLASE[c] }}>{c}</span>
            <div style={{ position: 'relative', flex: 1, height: 16 }}>
              {/* rango min-max */}
              <div style={{ position: 'absolute', top: 7, left: `${px(d.min)}%`, width: `${px(d.max) - px(d.min)}%`, height: 2, background: 'var(--line)' }} />
              {/* caja q1-q3 */}
              <div style={{ position: 'absolute', top: 3, left: `${px(d.q1)}%`, width: `${Math.max(px(d.q3) - px(d.q1), 1)}%`, height: 10, background: COLOR_CLASE[c], opacity: 0.35, borderRadius: 2 }} />
              {/* mediana */}
              <div style={{ position: 'absolute', top: 1, left: `${px(d.mediana)}%`, width: 2, height: 14, background: COLOR_CLASE[c] }} />
            </div>
            <span className="mono muted" style={{ fontSize: 10, width: 70, textAlign: 'right' }}>med {d.mediana}</span>
          </div>
        );
      })}
    </div>
  );
}

// Heatmap de correlación (solo top variables para que sea legible)
function MatrizCorr({ matriz, topVars }) {
  const idx = matriz.variables.map((v, i) => [v, i]).filter(([v]) => topVars.includes(v));
  const color = (v) => {
    const a = Math.abs(v);
    if (v > 0) return `rgba(111,154,94,${a})`;
    return `rgba(197,99,78,${a})`;
  };
  return (
    <div style={{ overflow: 'auto' }}>
      <table style={{ fontSize: 10 }}>
        <thead><tr><th></th>{idx.map(([v]) => <th key={v} style={{ writingMode: 'vertical-rl', padding: 4, fontFamily: 'var(--mono)', textTransform: 'none' }}>{v.slice(0, 16)}</th>)}</tr></thead>
        <tbody>
          {idx.map(([v1, i1]) => (
            <tr key={v1}>
              <td style={{ fontFamily: 'var(--mono)', fontSize: 10, whiteSpace: 'nowrap' }}>{v1.slice(0, 18)}</td>
              {idx.map(([v2, i2]) => (
                <td key={v2} style={{ background: color(matriz.valores[i1][i2]), textAlign: 'center', color: '#14110d', fontWeight: 600 }}>
                  {matriz.valores[i1][i2].toFixed(1)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function PanelEDA({ periodo, modo }) {
  const [data, setData] = useState(null);
  const [cargando, setCargando] = useState(true);
  const [varSel, setVarSel] = useState(null);
  const [histVar, setHistVar] = useState(null);
  const [histPorClase, setHistPorClase] = useState(false);

  useEffect(() => {
    setCargando(true);
    api.eda(periodo, modo).then(d => { setData(d); setVarSel(d.top_variables ? d.top_variables[0] : null); setHistVar(d.variables ? d.variables[0] : null); setCargando(false); })
      .catch(() => { setData({ disponible: false }); setCargando(false); });
  }, [periodo, modo]);

  if (cargando) return <div className="card"><span className="spinner" /> analizando {periodo}…</div>;
  if (!data.disponible) return <div className="card"><h3>{periodo}</h3><p className="muted">{data.mensaje || 'Sin datos. Sube calificaciones y datapoints.'}</p></div>;

  const promData = varSel && data.promedio_por_calificacion[varSel]
    ? Object.entries(data.promedio_por_calificacion[varSel]).map(([k, v]) => ({ calif: k, valor: v })).sort((a, b) => Number(a.calif) - Number(b.calif))
    : [];

  return (
    <div>
      <div className="kpi-row">
        <div className="kpi"><div className="label">Créditos</div><div className="value small">{data.n}</div></div>
        {['Buena', 'Media', 'Riesgo'].map(c => (
          <div className="kpi" key={c}><div className="label">{c}</div>
            <div className="value small" style={{ color: COLOR_CLASE[c] }}>{data.distribucion_clases[c] || 0}</div></div>
        ))}
      </div>

      <div className="card" style={{ marginBottom: 18 }}>
        <h3>Qué variables separan mejor las clases <span className="tag">ranking por η²</span></h3>
        <p className="muted" style={{ fontSize: 12, marginBottom: 12 }}>
          η² (eta²) mide cuánto separa cada variable entre Buena/Media/Riesgo (más alto = mejor).
          Spearman es la correlación con la calificación. “Sig.” marca si esa correlación es
          estadísticamente significativa (p&lt;0.05).
        </p>
        <div className="table-wrap">
          <table>
            <thead><tr><th>Variable</th><th>Separación (η²)</th><th>Spearman</th><th>Sig.</th><th>μ Buena</th><th>μ Media</th><th>μ Riesgo</th></tr></thead>
            <tbody>{data.resumen.map((r, i) => <FilaResumen key={i} r={r} />)}</tbody>
          </table>
        </div>
      </div>

      <div className="split" style={{ marginBottom: 18 }}>
        <div className="card">
          <h3>Distribución por clase <span className="tag">top variables</span></h3>
          <p className="muted" style={{ fontSize: 11, marginBottom: 12 }}>Caja = rango intercuartil (Q1–Q3); línea = mediana; barra fina = min–max.</p>
          {data.top_variables.slice(0, 8).map(v => <Boxplot key={v} nombre={v} datos={data.boxplots[v] || {}} />)}
        </div>

        <div className="card">
          <h3>Promedio por calificación</h3>
          <label className="field" style={{ maxWidth: 320 }}>
            <span>Variable</span>
            <select value={varSel || ''} onChange={e => setVarSel(e.target.value)}>
              {data.top_variables.map(v => <option key={v} value={v}>{v}</option>)}
            </select>
          </label>
          <div style={{ height: 240 }}>
            <ResponsiveContainer>
              <BarChart data={promData} margin={{ top: 6, right: 6, bottom: 0, left: 0 }}>
                <XAxis dataKey="calif" tick={{ fill: '#64748b', fontSize: 11 }} axisLine={{ stroke: '#e2e8f0' }} tickLine={false} />
                <YAxis tick={{ fill: '#64748b', fontSize: 10 }} axisLine={false} tickLine={false} width={48} />
                <Tooltip contentStyle={{ background: '#ffffff', border: '1px solid #e2e8f0', borderRadius: 4, fontSize: 12 }} cursor={{ fill: 'rgba(107,79,217,0.08)' }} />
                <Bar dataKey="valor" radius={[3, 3, 0, 0]} fill="#6b4fd9" />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p className="muted" style={{ fontSize: 11, textAlign: 'center' }}>eje X = calificación (0–10); barra = promedio de la variable</p>
        </div>
      </div>

      {/* HISTOGRAMAS de variables cuantitativas */}
      <div className="card" style={{ marginBottom: 18 }}>
        <h3>Histograma de variables cuantitativas</h3>
        <div className="filterbar">
          <label className="field" style={{ minWidth: 280, marginBottom: 0 }}>
            <span>Variable</span>
            <select value={histVar || ''} onChange={e => setHistVar(e.target.value)}>
              {data.variables.map(v => <option key={v} value={v}>{v}</option>)}
            </select>
          </label>
          <label className="field" style={{ marginBottom: 0 }}>
            <span>Vista</span>
            <select value={histPorClase ? 'clase' : 'total'} onChange={e => setHistPorClase(e.target.value === 'clase')}>
              <option value="total">Total</option>
              <option value="clase">Apilado por clase</option>
            </select>
          </label>
        </div>
        <div style={{ height: 260 }}>
          <ResponsiveContainer>
            <BarChart data={histData(data, histVar)} margin={{ top: 6, right: 6, bottom: 0, left: 0 }}>
              <XAxis dataKey="centro" tick={{ fill: '#64748b', fontSize: 10 }} axisLine={{ stroke: '#e2e8f0' }} tickLine={false} />
              <YAxis tick={{ fill: '#64748b', fontSize: 10 }} axisLine={false} tickLine={false} width={40} />
              <Tooltip contentStyle={{ background: '#ffffff', border: '1px solid #e2e8f0', borderRadius: 4, fontSize: 12 }} cursor={{ fill: 'rgba(107,79,217,0.08)' }} />
              {histPorClase
                ? ['Buena', 'Media', 'Riesgo'].map(c => <Bar key={c} dataKey={c} stackId="a" fill={COLOR_CLASE[c]} />)
                : <Bar dataKey="total" radius={[3, 3, 0, 0]} fill="#6b4fd9" />}
            </BarChart>
          </ResponsiveContainer>
        </div>
        <p className="muted" style={{ fontSize: 11, textAlign: 'center' }}>frecuencia de cada rango de valores{histPorClase ? ', dividida por clase' : ''}</p>
      </div>

      {/* DISTRIBUCIÓN de la calificación + TORTAS categóricas */}
      <div className="split" style={{ marginBottom: 18 }}>
        <div className="card">
          <h3>Distribución de la calificación</h3>
          <div style={{ height: 240 }}>
            <ResponsiveContainer>
              <BarChart data={Object.entries(data.distribucion_calificacion).map(([k, v]) => ({ calif: k, n: v })).sort((a,b)=>Number(a.calif)-Number(b.calif))} margin={{ top: 6, right: 6, bottom: 0, left: 0 }}>
                <XAxis dataKey="calif" tick={{ fill: '#64748b', fontSize: 11 }} axisLine={{ stroke: '#e2e8f0' }} tickLine={false} />
                <YAxis tick={{ fill: '#64748b', fontSize: 10 }} axisLine={false} tickLine={false} width={40} />
                <Tooltip contentStyle={{ background: '#ffffff', border: '1px solid #e2e8f0', borderRadius: 4, fontSize: 12 }} cursor={{ fill: 'rgba(107,79,217,0.08)' }} />
                <Bar dataKey="n" radius={[3, 3, 0, 0]} fill="#6b4fd9" />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p className="muted" style={{ fontSize: 11, textAlign: 'center' }}>cuántos créditos hay en cada calificación (0–10)</p>
        </div>

        {data.categoricas && data.categoricas.debtor_level_of_education && (
          <div className="card">
            <h3>Nivel de educación</h3>
            <div style={{ height: 240 }}>
              <ResponsiveContainer>
                <PieChart>
                  <Pie data={Object.entries(data.categoricas.debtor_level_of_education.global).map(([k, v]) => ({ name: k, value: v }))}
                    dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label={({ name }) => name}>
                    {Object.keys(data.categoricas.debtor_level_of_education.global).map((k, i) => (
                      <Cell key={i} fill={PALETA[i % PALETA.length]} />
                    ))}
                  </Pie>
                  <Tooltip contentStyle={{ background: '#ffffff', border: '1px solid #e2e8f0', borderRadius: 4, fontSize: 12 }} />
                </PieChart>
              </ResponsiveContainer>
            </div>
            <p className="muted" style={{ fontSize: 11, textAlign: 'center' }}>distribución del nivel educativo de los deudores principales</p>
          </div>
        )}
      </div>

      <div className="card">
        <h3>Correlación entre variables <span className="tag">top {Math.min(data.top_variables.length, 12)}</span></h3>
        <p className="muted" style={{ fontSize: 11, marginBottom: 12 }}>
          Verde = correlación positiva, rojo = negativa. Variables muy correlacionadas entre sí son
          redundantes (explican lo mismo) — útil para entender por qué el modelo a veces “se confunde”.
        </p>
        <MatrizCorr matriz={data.matriz_correlacion} topVars={data.top_variables} />
      </div>
    </div>
  );
}

export default function EDA() {
  const [periodo, setPeriodo] = useState('6M');
  const [modo, setModo] = useState('titular');

  return (
    <div>
      <h2 className="section-title">EDA · Variables vs calificación</h2>
      <p className="section-desc">
        Análisis exploratorio de las variables del modelo frente a la calificación del crédito, usando un
        perfil por crédito (el deudor principal). Cambia entre titular y mejor perfil para ver cómo se
        comportan los datos en cada criterio.
      </p>

      <div className="filterbar" style={{ marginBottom: 20 }}>
        <label className="field" style={{ minWidth: 160 }}>
          <span>Horizonte</span>
          <select value={periodo} onChange={e => setPeriodo(e.target.value)}>
            <option value="6M">6 meses</option>
            <option value="12M">12 meses</option>
          </select>
        </label>
        <label className="field" style={{ minWidth: 240 }}>
          <span>Deudor principal</span>
          <select value={modo} onChange={e => setModo(e.target.value)}>
            <option value="titular">Titular (id_type_debtor = 1)</option>
            <option value="perfil">Mejor perfil (pesos)</option>
          </select>
        </label>
      </div>

      <PanelEDA periodo={periodo} modo={modo} />
    </div>
  );
}

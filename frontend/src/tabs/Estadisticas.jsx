import React, { useState, useEffect } from 'react';
import { api } from '../api';
import { Clase } from '../components/ui';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';

const ORDEN_LETRAS = ['AAA', 'AA', 'A', 'A-', 'BBB', 'BB', 'B', 'CCC', 'CC', 'C'];
const COLOR_CLASE = { Buena: '#10b981', Media: '#d9a441', Riesgo: '#b71f1f' };

function PanelHorizonte({ periodo }) {
  const [data, setData] = useState(null);
  const [filtroClase, setFiltroClase] = useState('Todas');
  const [filtroEdu, setFiltroEdu] = useState('Todas');

  useEffect(() => { api.estadisticas(periodo).then(setData).catch(() => setData({ disponible: false })); }, [periodo]);

  if (!data) return <div className="card"><span className="spinner" /> cargando {periodo}…</div>;
  if (!data.disponible) return <div className="card"><h3>{periodo}</h3><p className="muted">{data.mensaje || 'Sin datos. Sube calificaciones y datapoints, luego reentrena.'}</p></div>;

  const chartData = ORDEN_LETRAS.filter(l => data.distribucion_letras[l])
    .map(l => ({ letra: l, n: data.distribucion_letras[l] }));
  const colorLetra = (l) => ['AAA', 'AA'].includes(l) ? '#10b981' : ['A', 'A-', 'BBB'].includes(l) ? '#d9a441' : '#b71f1f';

  const edus = ['Todas', ...Array.from(new Set(data.tabla.map(r => r.debtor_level_of_education).filter(Boolean)))];
  const filas = data.tabla.filter(r =>
    (filtroClase === 'Todas' || r.clase3 === filtroClase) &&
    (filtroEdu === 'Todas' || r.debtor_level_of_education === filtroEdu));

  return (
    <div className="card">
      <h3>{periodo === '6M' ? '6 meses' : '12 meses'} <span className="tag">{data.n_creditos} créditos</span></h3>

      <div className="kpi-row">
        {['Buena', 'Media', 'Riesgo'].map(c => (
          <div className="kpi" key={c}>
            <div className="label">{c}</div>
            <div className="value small" style={{ color: COLOR_CLASE[c] }}>{data.distribucion_3clases[c] || 0}</div>
          </div>
        ))}
      </div>

      <div style={{ height: 180, marginBottom: 8 }}>
        <ResponsiveContainer>
          <BarChart data={chartData} margin={{ top: 6, right: 6, bottom: 0, left: -18 }}>
            <XAxis dataKey="letra" tick={{ fill: '#64748b', fontSize: 11, fontFamily: 'IBM Plex Mono' }} axisLine={{ stroke: '#e2e8f0' }} tickLine={false} />
            <YAxis tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} tickLine={false} />
            <Tooltip contentStyle={{ background: '#ffffff', border: '1px solid #e2e8f0', borderRadius: 4, color: '#1e293b', fontSize: 12 }} cursor={{ fill: 'rgba(107,79,217,0.08)' }} />
            <Bar dataKey="n" radius={[3, 3, 0, 0]}>
              {chartData.map((e, i) => <Cell key={i} fill={colorLetra(e.letra)} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="muted" style={{ fontSize: 11, textAlign: 'center', marginBottom: 16 }}>distribución por calificación (letra)</div>

      <div className="filterbar">
        <label className="field"><span>Filtrar por clase</span>
          <select value={filtroClase} onChange={e => setFiltroClase(e.target.value)}>
            {['Todas', 'Buena', 'Media', 'Riesgo'].map(c => <option key={c}>{c}</option>)}
          </select>
        </label>
        <label className="field"><span>Filtrar por educación</span>
          <select value={filtroEdu} onChange={e => setFiltroEdu(e.target.value)}>
            {edus.map(c => <option key={c}>{c}</option>)}
          </select>
        </label>
        <span className="muted" style={{ fontSize: 12 }}>{filas.length} de {data.tabla.length}</span>
      </div>

      <div className="table-wrap" style={{ maxHeight: 280 }}>
        <table>
          <thead><tr><th>Simulación</th><th>Score</th><th>Educación</th><th>Monto</th><th>Calif.</th><th>Clase</th></tr></thead>
          <tbody>
            {filas.map((r, i) => (
              <tr key={i}>
                <td>{r.id_simulacion}</td>
                <td>{r.debtor_credit_score}</td>
                <td style={{ fontFamily: 'var(--sans)' }}>{r.debtor_level_of_education}</td>
                <td>{r.original_amount ? Number(r.original_amount).toLocaleString() : '—'}</td>
                <td>{r.letra}</td>
                <td><Clase v={r.clase3} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default function Estadisticas() {
  return (
    <div>
      <h2 className="section-title">Calificaciones</h2>
      <p className="section-desc">
        Cuántos créditos calificados hay por horizonte, su distribución (AAA, AA, A…) y el perfil
        de cada uno. Filtra por clase o nivel educativo. 6 meses a la izquierda, 12 meses a la derecha.
      </p>
      <div className="split">
        <PanelHorizonte periodo="6M" />
        <PanelHorizonte periodo="12M" />
      </div>
    </div>
  );
}

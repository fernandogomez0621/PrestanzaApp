import React, { useState, useEffect } from 'react';
import { api } from '../api';

function PanelShap({ periodo, detalle }) {
  if (!detalle) return <div className="card"><h3>{periodo}</h3><p className="muted">Sin modelo.</p></div>;
  const shap = detalle.shap || [];
  const maxImp = Math.max(...shap.map(s => s.importancia_shap), 0.001);
  const conSentido = detalle.variables_con_sentido || [];

  return (
    <div className="card">
      <h3>{periodo === '6M' ? '6 meses' : '12 meses'} <span className="tag">{conSentido.length} con sentido</span></h3>
      <p className="muted" style={{ fontSize: 12, marginBottom: 12 }}>
        Importancia SHAP (barra) y dirección. Verde = la dirección coincide con la lógica de crédito;
        rojo = el modelo la usa al revés (señal de ruido/colinealidad).
      </p>
      <div className="table-wrap">
        <table>
          <thead><tr><th>Variable</th><th>Importancia</th><th>Dirección modelo</th><th>¿Sentido?</th></tr></thead>
          <tbody>
            {shap.map((s, i) => (
              <tr key={i}>
                <td style={{ fontFamily: 'var(--sans)', fontSize: 12 }}>{s.variable}</td>
                <td style={{ minWidth: 120 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <div style={{ height: 7, width: `${(s.importancia_shap / maxImp) * 70}px`, background: s.tiene_sentido ? 'var(--green)' : 'var(--red)', borderRadius: 2 }} />
                    <span style={{ fontSize: 11 }}>{s.importancia_shap.toFixed(3)}</span>
                  </div>
                </td>
                <td style={{ fontSize: 11 }}>{s.direccion}</td>
                <td>{s.esperado == null
                  ? <span className="muted">n/a</span>
                  : <span className={'badge ' + (s.tiene_sentido ? 'ok' : 'no')}>{s.tiene_sentido ? 'sí' : 'invertida'}</span>}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default function Shap({ versionActiva }) {
  const [detalle, setDetalle] = useState(null);
  useEffect(() => {
    if (versionActiva) api.detalleModelos(versionActiva).then(setDetalle).catch(() => setDetalle(null));
  }, [versionActiva]);

  return (
    <div>
      <h2 className="section-title">Interpretabilidad (SHAP)</h2>
      <p className="section-desc">
        A qué le da peso el modelo Buena/No-Buena y en qué dirección. Las variables marcadas como
        “invertida” contradicen la lógica crediticia (p. ej. premiar score bajo) y son las que se
        excluyen del modelo “SHAP con sentido”.
      </p>
      {detalle
        ? <div className="split"><PanelShap periodo="6M" detalle={detalle['6M']} /><PanelShap periodo="12M" detalle={detalle['12M']} /></div>
        : <div className="card"><p className="muted">Selecciona una versión de modelos en la pestaña “Modelos”.</p></div>}
    </div>
  );
}

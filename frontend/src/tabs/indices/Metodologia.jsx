import React, { useState, useEffect } from 'react';
import { api } from '../../api';
import { nombre } from './nombres';

function Composicion({ titulo, defs }) {
  if (!defs) return null;
  return (
    <div className="card">
      <h3>{titulo}</h3>
      <div className="table-wrap" style={{ maxHeight: 'none' }}>
        <table>
          <thead><tr><th>Índice</th><th>Variables (signo)</th></tr></thead>
          <tbody>
            {Object.entries(defs).map(([k, vars]) => (
              <tr key={k}>
                <td style={{ fontFamily: 'var(--sans)', fontWeight: 600 }}>{nombre(k)}</td>
                <td style={{ fontSize: 11 }}>
                  {Object.entries(vars).map(([v, s], i) => (
                    <span key={v}>{i > 0 && ', '}{v} <b style={{ color: s > 0 ? 'var(--red)' : 'var(--green)' }}>({s > 0 ? '+' : '−'})</b></span>
                  ))}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default function Metodologia({ versionActiva }) {
  const [d, setD] = useState(null);
  const c = d?.['12M']?.corte_buena ?? d?.['6M']?.corte_buena;
  useEffect(() => { if (versionActiva) api.detalleModelos(versionActiva).then(setD).catch(() => setD(null)); }, [versionActiva]);
  return (
    <div>
      <h2 className="section-title">Cómo se construyen</h2>
      <div className="card" style={{ marginBottom: 20, fontSize: 13, lineHeight: 1.7 }}>
        <p><b>0. Etiqueta.</b> Buena si la calificación es mayor o igual al corte de la clase Buena{c != null ? ` (hoy ${c})` : ''};
          No-Buena en caso contrario. El corte se define en “Letras & clases” de la app principal y lo comparten las dos apps.</p>
        <p><b>1. Estandarización.</b> Cada variable se lleva a una escala común con parámetros del entrenamiento.
          12M: logaritmo con signo y z-score, z = (slog(x) − media) / desviación. 6M: z = (x − mediana) / rango intercuartílico.</p>
        <p><b>2. Índice.</b> Promedio de sus variables estandarizadas multiplicadas por su signo: (+) si un valor más alto implica más riesgo,
          (−) si implica menos. Si falta una variable, el promedio se toma sobre las disponibles.</p>
        <p><b>3. Puntaje.</b> Suma simple de los índices; más alto = más riesgo. No hay pesos estimados.</p>
        <p><b>4. Decisión.</b> Se marca como No-Buena (revisar) el 1,5 × la tasa de No-Buena del entrenamiento con mayor puntaje.
          Las bandas Bajo / Medio / Alto son los terciles del puntaje de entrenamiento.</p>
        <p className="muted" style={{ fontSize: 12, marginTop: 8 }}>
          Al reentrenar solo se recalculan escalas, umbral y terciles; la composición y los signos son fijos. Los índices se diseñaron
          con los créditos actuales, así que deben confirmarse con cosechas nuevas antes de usarse para decidir.</p>
      </div>
      <div className="split">
        <Composicion titulo="6 meses (7 índices)" defs={d?.['6M']?.definiciones} />
        <Composicion titulo="12 meses (11 índices)" defs={d?.['12M']?.definiciones} />
      </div>
    </div>
  );
}

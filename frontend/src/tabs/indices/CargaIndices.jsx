import React, { useState } from 'react';
import { api } from '../../api';
import { FileDrop } from '../../components/ui';
import { CalifUploader } from '../CargaDatos';

export default function CargaIndices({ showToast, recargarVersiones, setVersionActiva }) {
  const [dpFile, setDpFile] = useState(null);
  const [dpInfo, setDpInfo] = useState(null);
  const [entrenando, setEntrenando] = useState(false);

  const subirDp = async (f) => {
    setDpFile(f);
    try { const r = await api.subirDatapoints(f); setDpInfo(r); showToast(`Datapoints: ${r.simulaciones} simulaciones`); }
    catch (e) { showToast('Error: ' + e.message); }
  };

  const reentrenar = async () => {
    setEntrenando(true);
    showToast('Ajustando los modelos por índices de 6M y 12M…');
    try {
      const r = await api.entrenar('titular');
      showToast(`✓ Entrenamiento completo — versión ${r.version}`);
      await recargarVersiones(); setVersionActiva(r.version);
    } catch (e) { showToast('Error al entrenar: ' + e.message); }
    setEntrenando(false);
  };

  return (
    <div>
      <h2 className="section-title">Carga de datos</h2>
      <p className="section-desc">
        Los datos son los mismos de la aplicación principal (puerto 8501): lo que se cargue aquí también queda disponible allá.
        Solo se usan las operaciones con calificación (créditos formalizados).
      </p>
      <div className="split">
        <CalifUploader periodo="6M" showToast={showToast} />
        <CalifUploader periodo="12M" showToast={showToast} />
      </div>
      <div className="card" style={{ marginTop: 20 }}>
        <h3>Datapoints <span className="tag">perfil de deudores</span></h3>
        <FileDrop label="Subir archivo de datapoints" file={dpFile} onFile={subirDp} />
        {dpInfo && (
          <div className="kpi-row" style={{ marginTop: 16 }}>
            <div className="kpi"><div className="label">Filas</div><div className="value small">{dpInfo.filas.toLocaleString()}</div></div>
            <div className="kpi"><div className="label">Simulaciones</div><div className="value small">{dpInfo.simulaciones}</div></div>
          </div>
        )}
      </div>
      <div className="card" style={{ marginTop: 20, borderColor: 'var(--amber)' }}>
        <h3>Reentrenar modelos por índices</h3>
        <p className="muted" style={{ marginBottom: 14, fontSize: 13 }}>
          Recalcula las escalas de cada variable, el umbral y los terciles con los datos vigentes, evalúa con validación cruzada
          y guarda una nueva versión fechada. La composición de los índices y sus signos no cambian.
        </p>
        <button className="btn" onClick={reentrenar} disabled={entrenando}>
          {entrenando ? <><span className="spinner" /> Entrenando…</> : 'Actualizar datos y reentrenar'}
        </button>
      </div>
    </div>
  );
}

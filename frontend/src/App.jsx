import React, { useState, useEffect } from 'react';
import { api } from './api';
import { useToast } from './components/ui';
import CargaDatos from './tabs/CargaDatos';
import Estadisticas from './tabs/Estadisticas';
import Modelos from './tabs/Modelos';
import Prediccion from './tabs/Prediccion';
import Shap from './tabs/Shap';
import Pesos from './tabs/Pesos';
import CalifConfig from './tabs/CalifConfig';
import EDA from './tabs/EDA';
import Interpretacion from './tabs/Interpretacion';

const TABS = [
  { id: 'carga', label: 'Carga de datos' },
  { id: 'estadisticas', label: 'Calificaciones' },
  { id: 'eda', label: 'EDA / Análisis' },
  { id: 'calif_config', label: 'Letras & clases' },
  { id: 'modelos', label: 'Modelos & versiones' },
  { id: 'interpretacion', label: 'Cómo leer métricas' },
  { id: 'pesos', label: 'Pesos deudor principal' },
  { id: 'prediccion', label: 'Predicción' },
  { id: 'shap', label: 'Interpretabilidad (SHAP)' },
];

export default function App() {
  const [tab, setTab] = useState('carga');
  const [versiones, setVersiones] = useState([]);
  const [versionActiva, setVersionActiva] = useState(null);
  const [letras, setLetras] = useState({});
  const [toast, showToast] = useToast();

  const recargarVersiones = async () => {
    try {
      const r = await api.versiones();
      setVersiones(r.versiones);
      if (r.versiones.length && !versionActiva) setVersionActiva(r.versiones[0].version);
    } catch (e) { /* backend aún no arranca */ }
  };
  useEffect(() => { recargarVersiones(); api.getCalifConfig().then(r=>setLetras(r.config.letras)).catch(()=>{}); }, []);

  const props = { showToast, versiones, versionActiva, setVersionActiva, recargarVersiones, letras };

  return (
    <div className="app">
      <header className="header">
        <div>
          <div className="brand">Prestanza<span className="dot">.</span></div>
        </div>
        <div className="subtitle">Modelo Prospectivo de Riesgo Crediticio</div>
        <div className="version-pill">
          modelos activos: <b>{versionActiva || 'ninguno'}</b>
        </div>
      </header>

      <nav className="tabs">
        {TABS.map(t => (
          <button key={t.id} className={'tab' + (tab === t.id ? ' active' : '')} onClick={() => setTab(t.id)}>
            {t.label}
          </button>
        ))}
      </nav>

      <main className="content">
        {tab === 'carga' && <CargaDatos {...props} />}
        {tab === 'estadisticas' && <Estadisticas {...props} />}
        {tab === 'modelos' && <Modelos {...props} />}
        {tab === 'eda' && <EDA {...props} />}
        {tab === 'calif_config' && <CalifConfig {...props} />}
        {tab === 'interpretacion' && <Interpretacion {...props} />}
        {tab === 'pesos' && <Pesos {...props} />}
        {tab === 'prediccion' && <Prediccion {...props} />}
        {tab === 'shap' && <Shap {...props} />}
      </main>
      {toast}
    </div>
  );
}

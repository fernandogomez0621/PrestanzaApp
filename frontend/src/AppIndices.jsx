import React, { useState, useEffect } from 'react';
import { api } from './api';
import { useToast } from './components/ui';
import CargaIndices from './tabs/indices/CargaIndices';
import ModelosIndices from './tabs/indices/ModelosIndices';
import PrediccionIndices from './tabs/indices/PrediccionIndices';
import Metodologia from './tabs/indices/Metodologia';

const TABS = [
  { id: 'carga', label: 'Carga de datos' },
  { id: 'modelos', label: 'Modelos & versiones' },
  { id: 'prediccion', label: 'Predicción' },
  { id: 'metodologia', label: 'Cómo se construyen' },
];

export default function AppIndices() {
  const [tab, setTab] = useState('modelos');
  const [versiones, setVersiones] = useState([]);
  const [versionActiva, setVersionActiva] = useState(null);
  const [toast, showToast] = useToast();

  const recargarVersiones = async () => {
    try {
      const r = await api.versiones();
      setVersiones(r.versiones);
      if (r.versiones.length && !versionActiva) setVersionActiva(r.versiones[0].version);
    } catch (e) { /* backend aún no arranca */ }
  };
  useEffect(() => { recargarVersiones(); }, []);

  const props = { showToast, versiones, versionActiva, setVersionActiva, recargarVersiones };

  return (
    <div className="app">
      <header className="header">
        <div><div className="brand">Prestanza<span className="dot">.</span></div></div>
        <div className="subtitle">Modelos de riesgo por índices · 6 y 12 meses</div>
        <div className="version-pill">modelos activos: <b>{versionActiva || 'ninguno'}</b></div>
      </header>
      <nav className="tabs">
        {TABS.map(t => (
          <button key={t.id} className={'tab' + (tab === t.id ? ' active' : '')} onClick={() => setTab(t.id)}>{t.label}</button>
        ))}
      </nav>
      <main className="content">
        {tab === 'carga' && <CargaIndices {...props} />}
        {tab === 'modelos' && <ModelosIndices {...props} />}
        {tab === 'prediccion' && <PrediccionIndices {...props} />}
        {tab === 'metodologia' && <Metodologia {...props} />}
      </main>
      {toast}
    </div>
  );
}

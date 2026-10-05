import React, { useRef, useState } from 'react';

export function FileDrop({ label, file, onFile, accept = '.csv,.xlsx,.xls,.ods' }) {
  const ref = useRef();
  return (
    <div className={'dropzone' + (file ? ' has-file' : '')} onClick={() => ref.current.click()}>
      <input ref={ref} type="file" accept={accept}
        onChange={(e) => e.target.files[0] && onFile(e.target.files[0])} />
      {file
        ? <div><b>✓ {file.name}</b><div className="muted" style={{ fontSize: 11, marginTop: 4 }}>Clic para cambiar</div></div>
        : <div>{label}<div className="muted" style={{ fontSize: 11, marginTop: 4 }}>CSV · XLSX · ODS</div></div>}
    </div>
  );
}

const claseBadge = (v) => {
  const k = String(v || '').toLowerCase();
  if (k === 'buena') return 'buena';
  if (k === 'media') return 'media';
  if (k === 'riesgo') return 'riesgo';
  if (k === 'no-buena') return 'nobuena';
  return 'media';
};
export function Clase({ v }) {
  if (!v) return <span className="muted">—</span>;
  return <span className={'badge ' + claseBadge(v)}>{v}</span>;
}

export function useToast() {
  const [msg, setMsg] = useState(null);
  const show = (m) => { setMsg(m); setTimeout(() => setMsg(null), 3500); };
  const node = msg ? <div className="toast">{msg}</div> : null;
  return [node, show];
}

// Mapea un número de calificación a su letra, según rangos [min,max) o dict legado
export function numeroALetra(valor, letras) {
  if (valor == null || letras == null) return null;
  const v = Number(valor);
  if (Array.isArray(letras)) {
    const r = letras.find(x => (x.min == null || v >= x.min) && (x.max == null || v < x.max));
    return r ? r.letra : null;
  }
  // formato viejo dict número->letra
  return letras[String(v)] || null;
}

import React from 'react';

export default function Interpretacion() {
  return (
    <div>
      <h2 className="section-title">Cómo leer las métricas</h2>
      <p className="section-desc">
        Guía rápida para interpretar la matriz de confusión y las métricas en el contexto de riesgo crediticio.
      </p>

      {/* Matriz de confusión */}
      <div className="card" style={{ marginBottom: 18, maxWidth: 820 }}>
        <h3>Matriz de confusión</h3>
        <p style={{ fontSize: 13, marginBottom: 12 }}>
          Las <b>filas son la realidad</b> (lo que de verdad era el crédito) y las <b>columnas la predicción</b>
          (lo que dijo el modelo). La diagonal son aciertos; lo de fuera son errores.
        </p>
        <div className="table-wrap" style={{ maxWidth: 460, marginBottom: 14 }}>
          <table>
            <thead><tr><th>real ↓ / predijo →</th><th>No-Buena</th><th>Buena</th></tr></thead>
            <tbody>
              <tr><td style={{ fontWeight: 600 }}>No-Buena</td>
                <td style={{ background: 'rgba(111,154,94,0.25)', textAlign: 'center' }}>acierto</td>
                <td style={{ background: 'rgba(197,99,78,0.3)', textAlign: 'center' }}>falso positivo</td></tr>
              <tr><td style={{ fontWeight: 600 }}>Buena</td>
                <td style={{ background: 'rgba(197,99,78,0.3)', textAlign: 'center' }}>falso negativo</td>
                <td style={{ background: 'rgba(111,154,94,0.25)', textAlign: 'center' }}>acierto</td></tr>
            </tbody>
          </table>
        </div>
        <p style={{ fontSize: 13, marginBottom: 6 }}>
          <span className="badge nobuena">Falso positivo</span> &nbsp;Era <b>No-Buena</b> y dijo Buena → le apruebas
          crédito a un mal pagador. Es el error que te hace <b>perder plata</b>.
        </p>
        <p style={{ fontSize: 13 }}>
          <span className="badge media">Falso negativo</span> &nbsp;Era <b>Buena</b> y dijo No-Buena → rechazas a un
          buen cliente. Es el error que te hace <b>perder negocio</b>.
        </p>
        <p className="muted" style={{ fontSize: 12, marginTop: 12 }}>
          En crédito el falso positivo suele importar más: un impago duele más que un buen cliente que se fue.
          En 3 clases es igual pero con más casillas — mira sobre todo la fila de <b>Riesgo</b>: si esos casos
          se predicen como Buena/Media, el modelo no está detectando a los riesgosos.
        </p>

        <div style={{ marginTop: 14, padding: '12px 14px', background: 'var(--morado-soft)', borderRadius: 'var(--radius-sm)' }}>
          <div style={{ fontWeight: 600, fontSize: 13, marginBottom: 4, color: 'var(--morado)' }}>
            Recomendación de lectura (3 clases): mira a qué lado de la diagonal cae el error
          </div>
          <p style={{ fontSize: 12.5, margin: 0 }}>
            Con las clases ordenadas (Buena &gt; Media &gt; Riesgo), lo importante es <b>hacia qué lado</b> se
            equivoca: errores <b>por encima</b> de la diagonal (predice mejor de lo que es, p. ej. un Riesgo
            que llama Buena) son los <b>peligrosos</b> — aprobar a quien fallará. Errores <b>por debajo</b>
            (predice peor de lo que es) son <b>conservadores</b> — rechazar a un bueno. La esquina superior
            derecha (real malo → predicho bueno) es la que más cuesta.
          </p>
        </div>
      </div>

      {/* Métricas */}
      <div className="card" style={{ marginBottom: 18, maxWidth: 820 }}>
        <h3>Las métricas</h3>
        <div style={{ display: 'grid', gap: 12 }}>
          <Metr nombre="Recall (de una clase)" texto="De los que REALMENTE eran de esa clase, ¿cuántos atrapó? (se lee en la fila). Recall de Riesgo = 0 significa que no detecta ningún riesgoso." />
          <Metr nombre="Precisión (de una clase)" texto="De los que el modelo PREDIJO como esa clase, ¿cuántos acertó? (se lee en la columna). Precisión baja = muchas falsas alarmas." />
          <Metr nombre="F1" texto="Equilibra precisión y recall en un solo número (su media armónica). Útil cuando importan los dos a la vez." />
          <Metr nombre="F1-macro" texto="Promedio del F1 de todas las clases dándoles el mismo peso. Por eso castiga cuando una clase (ej. Riesgo) va mal, aunque las demás vayan bien." />
          <Metr nombre="Accuracy" texto="% total de aciertos. Engaña con clases desbalanceadas: si casi todo es Buena, acertar siempre 'Buena' da accuracy alta pero es inútil." />
          <Metr nombre="Balanced accuracy" texto="Como accuracy pero promediando el acierto de cada clase por igual. Más honesta cuando hay desbalance." />
        </div>
      </div>

      {/* Qué mirar */}
      <div className="card" style={{ maxWidth: 820, borderColor: 'var(--amber)' }}>
        <h3>Qué mirar en este modelo</h3>
        <p style={{ fontSize: 13 }}>
          Para riesgo crediticio, las dos cosas que cuestan dinero: el <b>recall de Riesgo / No-Buena</b>
          (¿se me escapan los malos?) y los <b>falsos positivos</b> (¿estoy aprobando malos?). Si el recall
          de Riesgo es bajo, el modelo no sirve para frenar a los riesgosos — aunque su accuracy se vea bien.
        </p>
      </div>
    </div>
  );
}

function Metr({ nombre, texto }) {
  return (
    <div style={{ borderLeft: '3px solid var(--amber)', paddingLeft: 14 }}>
      <div style={{ fontWeight: 600, fontSize: 13 }}>{nombre}</div>
      <div className="muted" style={{ fontSize: 12.5 }}>{texto}</div>
    </div>
  );
}

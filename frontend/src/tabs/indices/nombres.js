// Nombres legibles de los índices
export const NOMBRE_INDICE = {
  mora_titular: 'Mora del titular', mora_codeudores: 'Mora de codeudores', peor_deudor: 'Peor deudor',
  carga_titular: 'Carga del titular', carga_hogar: 'Carga del hogar', respaldo: 'Respaldo',
  score_titular: 'Score del titular', score_codeudor: 'Score de codeudores', estabilidad: 'Estabilidad',
  litigios: 'Litigios', analista: 'Analista',
  capacidad_titular: 'Capacidad del titular', 'tamaño_crédito': 'Tamaño del crédito', score: 'Score',
  'educación': 'Educación', 'garantía_casa': 'Garantía casa',
};
export const nombre = (k) => NOMBRE_INDICE[k] || k;
export const fechaVersion = (f) => `${f.slice(0, 4)}-${f.slice(4, 6)}-${f.slice(6, 8)} ${f.slice(9, 11)}:${f.slice(11, 13)}`;

import React from 'react'
import { createRoot } from 'react-dom/client'
import './styles.css'
import App from './App.jsx'
import AppIndices from './AppIndices.jsx'

// VITE_MODO=indices compila la versión que solo tiene los modelos por índices (puerto 8502)
const Raiz = import.meta.env.VITE_MODO === 'indices' ? AppIndices : App;
createRoot(document.getElementById('root')).render(<Raiz />)

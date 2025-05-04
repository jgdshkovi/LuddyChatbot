// ✅ src/main.jsx
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App.jsx';
import './styles.css';

console.log('✅ React root injected');

const rootElement = document.createElement('div');
rootElement.id = 'luddy-chatbot-root';
rootElement.style.zIndex = '999999';
rootElement.style.position = 'fixed';
rootElement.style.bottom = '1rem';
rootElement.style.right = '1rem';
document.body.appendChild(rootElement);

ReactDOM.createRoot(rootElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
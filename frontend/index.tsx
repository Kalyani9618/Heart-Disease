import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css'; // Import Tailwind CSS
import App from './App';

// Mock API responses for development/mobile
const mockApiResponses: Record<string, (url: string, options?: RequestInit) => Response | null> = {
  '/api/provider/status': () => new Response(JSON.stringify({
    ollama_available: false,
    openrouter_available: true,
    current_provider: 'openrouter',
    ollama_url: 'http://localhost:11434',
    openrouter_configured: true,
  }), { headers: { 'Content-Type': 'application/json' }, status: 200 }),
  
  '/api/provider/available': () => new Response(JSON.stringify({
    available_providers: [
      {
        name: 'ollama',
        label: 'Ollama (Local)',
        description: 'Local Ollama instance',
        available: false,
      },
      {
        name: 'openrouter',
        label: 'OpenRouter',
        description: 'OpenRouter API',
        available: true,
      },
    ],
  }), { headers: { 'Content-Type': 'application/json' }, status: 200 }),

  '/api/provider/select': (url: string, options?: RequestInit) => {
    try {
      const body = JSON.parse(options?.body as string);
      return new Response(JSON.stringify({
        selected_provider: body.provider || 'openrouter',
      }), { headers: { 'Content-Type': 'application/json' }, status: 200 });
    } catch {
      return new Response(JSON.stringify({ error: 'Invalid request' }), { status: 400 });
    }
  },
};

// Intercept fetch calls to mock API endpoints (dev only)
if (import.meta.env.DEV) {
  const originalFetch = window.fetch;
  window.fetch = function(input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
    const urlString = typeof input === 'string' ? input : input.toString();
    
    // Extract the path from the URL
    let pathToMatch = urlString;
    try {
      const urlObj = new URL(urlString, window.location.href);
      pathToMatch = urlObj.pathname + urlObj.search;
    } catch {
      pathToMatch = urlString;
    }
    
    // Check if this is a mock API endpoint
    for (const [mockPath, mockHandler] of Object.entries(mockApiResponses)) {
      if (pathToMatch.includes(mockPath)) {
        console.log(`[Mock API] Intercepting ${mockPath}: ${pathToMatch}`);
        const response = mockHandler(pathToMatch, init);
        if (response) {
          return Promise.resolve(response);
        }
      }
    }
    
    // Fall back to original fetch for all other requests
    return originalFetch.apply(this, arguments as any);
  };
}

// Show loading indicator immediately
const rootElement = document.getElementById('root');
if (rootElement) {
  rootElement.innerHTML = '<div style="display:flex;justify-content:center;align-items:center;height:100vh;background:#101922;color:white;font-family:sans-serif;"><div style="text-align:center;"><div style="font-size:24px;margin-bottom:16px;">Loading Cardio AI...</div><div id="loading-status">Initializing...</div></div></div>';
}

// Error handler
window.onerror = function(msg, url, lineNo, columnNo, error) {
  console.error('Global error:', msg, url, lineNo, columnNo, error);
  const statusEl = document.getElementById('loading-status');
  if (statusEl) {
    // Use textContent instead of innerHTML to prevent XSS
    statusEl.textContent = '';
    const errorDiv = document.createElement('div');
    errorDiv.style.color = 'red';
    errorDiv.textContent = 'Error: ' + String(msg);
    const detailDiv = document.createElement('div');
    detailDiv.style.fontSize = '12px';
    detailDiv.style.marginTop = '8px';
    detailDiv.textContent = String(url || '') + ':' + String(lineNo || '');
    statusEl.appendChild(errorDiv);
    statusEl.appendChild(detailDiv);
  }
  return false;
};

// Mount React app
if (!rootElement) {
  throw new Error("Could not find root element to mount to");
}

console.log('Mounting React app to root element');
const statusEl = document.getElementById('loading-status');
if (statusEl) statusEl.textContent = 'Starting React...';

// Check if CSS is loaded
setTimeout(() => {
  const computedStyle = window.getComputedStyle(document.body);
  console.log('Body styles loaded:', {
    backgroundColor: computedStyle.backgroundColor,
    fontFamily: computedStyle.fontFamily,
  });
}, 100);

const root = ReactDOM.createRoot(rootElement);
root.render(React.createElement(App));

console.log('React app mounted successfully');

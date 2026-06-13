import React, { useState, useEffect, useCallback, createContext, useContext } from 'react';

export interface ToastMessage {
  id: string;
  message: string;
  type: 'success' | 'error' | 'info' | 'warning';
  duration?: number;
}

interface ToastContextType {
  showToast: (message: string, type?: ToastMessage['type'], duration?: number) => void;
}

const ToastContext = createContext<ToastContextType | null>(null);

export const useToast = () => {
  const context = useContext(ToastContext);
  if (!context) {
    // Fallback to console if not wrapped in ToastProvider
    return {
      showToast: (message: string, type: ToastMessage['type'] = 'info') => {
        console.log(`[Toast ${type}] ${message}`);
      }
    };
  }
  return context;
};

const ToastItem: React.FC<{ toast: ToastMessage; onClose: (id: string) => void }> = ({ toast, onClose }) => {
  useEffect(() => {
    const timer = setTimeout(() => onClose(toast.id), toast.duration || 3000);
    return () => clearTimeout(timer);
  }, [toast.id, toast.duration, onClose]);

  const icons: Record<string, string> = {
    success: 'check_circle',
    error: 'error',
    info: 'info',
    warning: 'warning',
  };

  const colors: Record<string, string> = {
    success: 'bg-green-500/90 border-green-400/50',
    error: 'bg-red-500/90 border-red-400/50',
    info: 'bg-blue-500/90 border-blue-400/50',
    warning: 'bg-yellow-500/90 border-yellow-400/50',
  };

  return (
    <div
      className={`flex items-center gap-3 px-4 py-3 rounded-xl border shadow-lg backdrop-blur-sm text-white text-sm font-medium max-w-[90vw] animate-in slide-in-from-top-2 fade-in duration-200 ${colors[toast.type]}`}
      role="alert"
    >
      <span className="material-symbols-outlined text-lg shrink-0">{icons[toast.type]}</span>
      <span className="flex-1 min-w-0">{toast.message}</span>
      <button onClick={() => onClose(toast.id)} className="shrink-0 opacity-70 hover:opacity-100 transition-opacity">
        <span className="material-symbols-outlined text-sm">close</span>
      </button>
    </div>
  );
};

export const ToastProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [toasts, setToasts] = useState<ToastMessage[]>([]);

  const showToast = useCallback((message: string, type: ToastMessage['type'] = 'info', duration = 3000) => {
    const id = Date.now().toString() + Math.random().toString(36).slice(2, 5);
    setToasts(prev => [...prev.slice(-4), { id, message, type, duration }]); // Max 5 toasts
  }, []);

  const removeToast = useCallback((id: string) => {
    setToasts(prev => prev.filter(t => t.id !== id));
  }, []);

  return (
    <ToastContext.Provider value={{ showToast }}>
      {children}
      {/* Toast container - fixed at top for Android */}
      {toasts.length > 0 && (
        <div className="fixed top-4 left-4 right-4 z-[9999] flex flex-col items-center gap-2 pointer-events-none">
          {toasts.map(toast => (
            <div key={toast.id} className="pointer-events-auto">
              <ToastItem toast={toast} onClose={removeToast} />
            </div>
          ))}
        </div>
      )}
    </ToastContext.Provider>
  );
};

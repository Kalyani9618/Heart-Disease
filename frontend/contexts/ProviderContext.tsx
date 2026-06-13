import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';

type Provider = 'ollama' | 'openrouter';

interface ProviderStatus {
  ollama_available: boolean;
  openrouter_available: boolean;
  current_provider: Provider;
  ollama_url: string;
  openrouter_configured: boolean;
}

interface ProviderContextType {
  selectedProvider: Provider;
  setSelectedProvider: (provider: Provider) => Promise<void>;
  providerStatus: ProviderStatus | null;
  isLoading: boolean;
  error: string | null;
  availableProviders: Array<{
    name: Provider;
    label: string;
    description: string;
    available: boolean;
  }>;
}

const ProviderContext = createContext<ProviderContextType | undefined>(undefined);

export const ProviderProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [selectedProvider, setSelectedProviderState] = useState<Provider>('openrouter');
  const [providerStatus, setProviderStatus] = useState<ProviderStatus | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [availableProviders, setAvailableProviders] = useState<Array<{
    name: Provider;
    label: string;
    description: string;
    available: boolean;
  }>>([]);

  // Load provider status on mount
  useEffect(() => {
    const loadProviderStatus = async () => {
      try {
        setIsLoading(true);
        setError(null);

        // Get current provider and status
        const [statusRes, availableRes] = await Promise.all([
          fetch('/api/provider/status'),
          fetch('/api/provider/available')
        ]);

        if (statusRes.ok) {
          const status = await statusRes.json() as ProviderStatus;
          setProviderStatus(status);
          setSelectedProviderState(status.current_provider);
        }

        if (availableRes.ok) {
          const data = await availableRes.json();
          setAvailableProviders(data.available_providers || []);
        }
      } catch (err) {
        console.error('Failed to load provider status:', err);
        setError(err instanceof Error ? err.message : 'Failed to load provider status');
      } finally {
        setIsLoading(false);
      }
    };

    loadProviderStatus();
  }, []);

  const setSelectedProvider = async (provider: Provider) => {
    try {
      setIsLoading(true);
      setError(null);

      const response = await fetch('/api/provider/select', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          provider,
        }),
      });

      if (!response.ok) {
        throw new Error(`Failed to select provider: ${response.statusText}`);
      }

      const data = await response.json();
      setSelectedProviderState(data.selected_provider);

      // Update status
      const statusRes = await fetch('/api/provider/status');
      if (statusRes.ok) {
        const status = await statusRes.json() as ProviderStatus;
        setProviderStatus(status);
      }

      // Save to localStorage
      localStorage.setItem('selected_llm_provider', provider);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to select provider';
      setError(message);
      throw err;
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <ProviderContext.Provider
      value={{
        selectedProvider,
        setSelectedProvider,
        providerStatus,
        isLoading,
        error,
        availableProviders,
      }}
    >
      {children}
    </ProviderContext.Provider>
  );
};

export const useProvider = () => {
  const context = useContext(ProviderContext);
  if (!context) {
    throw new Error('useProvider must be used within a ProviderProvider');
  }
  return context;
};

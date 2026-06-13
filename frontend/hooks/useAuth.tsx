import React, { useState, useEffect, createContext, useContext } from 'react';
import { apiClient } from '../services/apiClient';
import { authService } from '../services/authService';

interface User {
    id: string;
    name: string;
    email: string;
    role: 'patient' | 'doctor';
}

interface AuthContextType {
    user: User | null;
    isAuthenticated: boolean;
    login: (email: string, password: string) => Promise<void>;
    logout: () => void;
    loading: boolean;
    refreshUser: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | null>(null);

export const AuthProvider = ({ children }: { children: React.ReactNode }) => {
    const [user, setUser] = useState<User | null>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        // Check for existing session on mount
        const initAuth = async () => {
            const token = authService.getToken();

            if (token && !authService.isTokenExpired()) {
                try {
                    // Verify token is still valid and get user data
                    const userData = await apiClient.me();
                    setUser(userData as User);
                } catch (error) {
                    console.error('Failed to restore session:', error);
                    authService.clearAuth();
                }
            }

            setLoading(false);
        };

        initAuth();
    }, []);

    const login = async (email: string, password: string) => {
        setLoading(true);
        try {
            const response = await apiClient.login(email, password);

            // Store tokens
            authService.setToken(response.token);
            if (response.refresh_token) {
                authService.setRefreshToken(response.refresh_token);
            }

            // Store user data
            authService.setUser(response.user);
            setUser(response.user as User);

            // Store user_id for appointment system
            if (response.user?.id) {
                localStorage.setItem('user_id', response.user.id);
            }

            console.log('[Auth] Login successful');
        } catch (error) {
            console.error('[Auth] Login failed:', error);
            throw error;
        } finally {
            setLoading(false);
        }
    };

    const logout = () => {
        // Call logout API (fire and forget)
        apiClient.logout().catch(err => {
            console.error('[Auth] Logout API error:', err);
        });

        // Clear local auth state
        authService.clearAuth();
        setUser(null);

        console.log('[Auth] Logged out');
    };

    const refreshUser = async () => {
        try {
            const userData = await apiClient.me();
            setUser(userData as User);
            authService.setUser(userData);
        } catch (error) {
            console.error('[Auth] Failed to refresh user:', error);
            // Token invalid, clear auth
            authService.clearAuth();
            setUser(null);
            throw error;
        }
    };

    return (
        <AuthContext.Provider value={{
            user,
            isAuthenticated: !!user,
            login,
            logout,
            loading,
            refreshUser
        }}>
            {children}
        </AuthContext.Provider>
    );
};

export const useAuth = () => {
    const context = useContext(AuthContext);
    if (!context) {
        throw new Error('useAuth must be used within an AuthProvider');
    }
    return context;
};

/**
 * Authentication Service
 * Handles JWT token storage, validation, and refresh logic
 */

const TOKEN_KEY = 'auth_token';
const TOKEN_EXPIRY_KEY = 'auth_token_expiry';
const REFRESH_TOKEN_KEY = 'refresh_token';
const USER_KEY = 'auth_user';

interface TokenPayload {
    exp: number;
    user_id: string;
    email: string;
    [key: string]: any;
}

export const authService = {
    /**
     * Store JWT access token
     */
    setToken(token: string): void {
        try {
            localStorage.setItem(TOKEN_KEY, token);

            // Decode JWT payload to get expiry
            const payload = this.decodeToken(token);
            if (payload?.exp) {
                localStorage.setItem(TOKEN_EXPIRY_KEY, payload.exp.toString());
            }
        } catch (error) {
            console.error('[AuthService] Failed to store token:', error);
        }
    },

    /**
     * Get current access token
     */
    getToken(): string | null {
        return localStorage.getItem(TOKEN_KEY);
    },

    /**
     * Check if current token is expired
     */
    isTokenExpired(): boolean {
        const expiry = localStorage.getItem(TOKEN_EXPIRY_KEY);
        if (!expiry) return true;

        const expiryTime = parseInt(expiry) * 1000; // Convert to milliseconds
        const now = Date.now();

        // Add 60 second buffer to refresh before actual expiry
        return now >= (expiryTime - 60000);
    },

    /**
     * Decode JWT token payload
     */
    decodeToken(token: string): TokenPayload | null {
        try {
            const parts = token.split('.');
            if (parts.length !== 3) return null;

            const payload = JSON.parse(atob(parts[1]));
            return payload;
        } catch (error) {
            console.error('[AuthService] Failed to decode token:', error);
            return null;
        }
    },

    /**
     * Store refresh token
     */
    setRefreshToken(token: string): void {
        localStorage.setItem(REFRESH_TOKEN_KEY, token);
    },

    /**
     * Get refresh token
     */
    getRefreshToken(): string | null {
        return localStorage.getItem(REFRESH_TOKEN_KEY);
    },

    /**
     * Store user data
     */
    setUser(user: any): void {
        localStorage.setItem(USER_KEY, JSON.stringify(user));
    },

    /**
     * Get stored user data
     */
    getUser(): any | null {
        const userStr = localStorage.getItem(USER_KEY);
        if (!userStr) return null;

        try {
            return JSON.parse(userStr);
        } catch {
            return null;
        }
    },

    /**
     * Get current user's ID from stored user data or decoded token
     */
    getUserId(): string | null {
        // Try from stored user data first
        const user = this.getUser();
        if (user?.user_id) return user.user_id;
        if (user?.id) return user.id;

        // Fallback: extract from JWT token
        const token = this.getToken();
        if (token) {
            const payload = this.decodeToken(token);
            if (payload?.user_id) return payload.user_id;
        }

        return null;
    },

    /**
     * Clear all authentication data
     */
    clearAuth(): void {
        localStorage.removeItem(TOKEN_KEY);
        localStorage.removeItem(TOKEN_EXPIRY_KEY);
        localStorage.removeItem(REFRESH_TOKEN_KEY);
        localStorage.removeItem(USER_KEY);
    },

    /**
     * Check if user is authenticated (has valid token)
     */
    isAuthenticated(): boolean {
        const token = this.getToken();
        if (!token) return false;

        return !this.isTokenExpired();
    },

    /**
     * Get authorization header value
     */
    getAuthHeader(): string | null {
        const token = this.getToken();
        if (!token) return null;

        return `Bearer ${token}`;
    },
};

export default authService;

import {createContext, useCallback, useContext, useEffect, useState} from "react";
import {TOKEN_KEY, REFRESH_KEY} from "../../api/api";
export const AuthContext = createContext(null);

export const useAuth = () => useContext(AuthContext);

export  const AuthProvider = ({ children, navigate }) => {
    export const [isAuthenticated, setIsAuthenticated] = useState(false);
    const [user, setUser] = useState(null);
    export const [isLoading, setIsLoading] = useState(true);

    const loadUserFromToken = useCallback(async () => {
        const token = localStorage.getItem(TOKEN_KEY);
        if (token) {
            try {
                const response = await secureFetch('/auth/users/me/', { method: 'GET' });
                const userData = await response.json();
                setUser(userData);
                setIsAuthenticated(true);
            } catch (error) {
                console.error('Error loading user data:', error);
                handleLogout(); // Вихід, якщо токен недійсний
            }
        }
        setIsLoading(false);
    }, []);

    useEffect(() => {
        loadUserFromToken();
    }, [loadUserFromToken]);

    const handleLogin = async (email, password) => {
        try {
            const response = await secureFetch('/auth/jwt/create/', {
                method: 'POST',
                body: JSON.stringify({ email, password }),
            }, false); // Не включаємо auth токен при логіні

            const data = await response.json();
            localStorage.setItem(TOKEN_KEY, data.access);
            localStorage.setItem(REFRESH_KEY, data.refresh);

            // Після успішного входу завантажуємо дані користувача
            await loadUserFromToken();
            navigate('dashboard');
            return true;
        } catch (e) {
            console.error('Login failed:', e.message);
            return false;
        }
    };

    const handleLogout = () => {
        localStorage.removeItem(TOKEN_KEY);
        localStorage.removeItem(REFRESH_KEY);
        setIsAuthenticated(false);
        setUser(null);
        navigate('login');
    };

     const value = {
        isAuthenticated,
        user,
        isLoading,
        login: handleLogin,
        logout: handleLogout,
        loadUserFromToken,
    };

    if (isLoading) {
        // Оновлено для темної теми
        return <div className="min-h-screen flex items-center justify-center bg-gray-900">
            <RefreshCcw className="animate-spin text-indigo-500 h-8 w-8" />
        </div>;
    }

    return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

// import React, {useEffect, useState} from 'react';
// import {API_BASE_URL, apiRequest, REFRESH_KEY, TOKEN_KEY} from "./api/api.js";
// import {Header} from "./components/Header";
// import LoginPage from "./pages/LoginPage";
// import ProfilePage from "./pages/Profile";
// import DashboardPage from "./pages/DashboardPage";
// import RegisterPage from "./pages/RegisterPage";
// import {isAuthenticated, isLoading, useAuth} from "./features/Auth/AuthProvider";
//
// // =============================================================================
// // 1. КОНСТАНТИ ТА API НАЛАШТУВАННЯ
// // =============================================================================
//
// // Хелпер для експоненційного відступу
// const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));
// const MAX_RETRIES = 3;
//
// /**
//  * Функція для безпечного fetch із токеном та експоненційним відступом.
//  * @param {string} endpoint - API endpoint (відносний до API_BASE_URL)
//  * @param {object} options - Налаштування fetch
//  * @param {boolean} includeAuth - Чи додавати Access Token до заголовка
//  */
// export const secureFetch = async (endpoint, options = {}, includeAuth = true) => {
//     const url = `${API_BASE_URL}${endpoint}`;
//     const token = localStorage.getItem(TOKEN_KEY);
//
//     const headers = {
//         'Content-Type': 'application/json',
//         ...options.headers,
//     };
//
//     if (includeAuth && token) {
//         headers['Authorization'] = `Bearer ${token}`;
//     }
//
//     for (let i = 0; i < MAX_RETRIES; i++) {
//         try {
//             const response = await fetch(url, { ...options, headers });
//
//             if (response.status === 401 && includeAuth) {
//                 // Спробувати оновити токен, якщо це не запит на оновлення
//                 if (endpoint !== '/auth/jwt/refresh/') {
//                     const success = await refreshAccessToken();
//                     if (success) {
//                         // Повторити запит з новим токеном
//                         const newToken = localStorage.getItem(TOKEN_KEY);
//                         headers['Authorization'] = `Bearer ${newToken}`;
//                         return await fetch(url, {...options, headers});
//                     }
//                 }
//             }
//
//             if (!response.ok) {
//                 const errorData = await response.json().catch(() => ({}));
//                 new Error(errorData.detail || response.statusText);
//             }
//
//             return response;
//         } catch (error) {
//             if (i < MAX_RETRIES - 1) {
//                 const delay = Math.pow(2, i) * 1000;
//                 await sleep(delay);
//             } else {
//                 throw error; // Викидаємо помилку після останньої спроби
//             }
//         }
//     }
// };
//
// const refreshAccessToken = async () => {
//     const refreshToken = localStorage.getItem(REFRESH_KEY);
//     if (!refreshToken) return false;
//
//     try {
//         const response = await secureFetch('/auth/jwt/refresh/', {
//             method: 'POST',
//             body: JSON.stringify({ refresh: refreshToken }),
//         }, false); // Не включаємо токен, бо його немає або він недійсний
//
//         if (response.ok) {
//             const data = await response.json();
//             localStorage.setItem(TOKEN_KEY, data.access);
//             return true;
//         } else {
//             console.error('Refresh token failed');
//             return false;
//         }
//     } catch (e) {
//         console.error('Refresh token API call failed', e);
//         return false;
//     }
// };
// // =============================================================================
// // 4. ГОЛОВНИЙ ДОДАТОК (РОУТЕР)
// // =============================================================================
// // Огортаємо App у AuthProvider, щоб забезпечити доступ до контексту
// // Протектед роут
// function ProtectedRoute({ children }) {
//   const { isAuthenticated } = useAuth();
//   if (!isAuthenticated) return <Navigate to="/login" replace />;
//   return children;
// }
//
// export const App = () => {
//     const [user, SetUser] = useState(null)
//
//     useEffect(() => {
//         const token = localStorage.getItem('accessToken')
//         if (!token) return;
//         apiRequest('/auth/users/me/', 'GET', null, token)
//             .then((data) => setUser(data))
//             .catch(() => localStorage.removeItem('accessToken'))
//     }, []);
//     // return user ? <Dashboard user={user} /> : <LoginPage/>;
//
//     const renderPage = () => {
//         if (isLoading) return null; // Завантаження обробляється в AuthProvider
//
//         switch (currentPage) {
//             case 'login':
//                 return <LoginPage navigate={navigate} />;
//             case 'register':
//                 return <RegisterPage navigate={navigate} />;
//             case 'dashboard':
//                 if (isAuthenticated) return <DashboardPage />;
//                 break;
//             case 'profile':
//                 if (isAuthenticated) return <ProfilePage />;
//                 break;
//             default:
//                 return <LoginPage navigate={navigate} />;
//         }
//     };
//
//     return (
//         // Оновлено головний фон для темної теми
//         <div className="min-h-screen bg-gray-900 font-sans">
//             <Header navigate={navigate} />
//             <main className="flex-grow">
//                 {renderPage()}
//             </main>
//         </div>
//     );
// };
//
//
// // Додаємо стилі Tailwind CSS
// export default function AppWrapper() {
//     return (
//         <>
//             <script src="https://cdn.tailwindcss.com"></script>
//             <RootApp />
//         </>
//     );
// }

import React, { useState, useEffect } from "react";
import { apiRequest } from "./api/api";
import Header from "./components/Header.jsx";
import LoginPage from "./pages/LoginPage.jsx";
import RegisterPage from "./pages/RegisterPage.jsx";
import DashboardPage from "./pages/DashboardPage.jsx";

const App = () => {
  const [currentPage, setCurrentPage] = useState("login");
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [user, setUser] = useState(null);

  const navigate = (page) => setCurrentPage(page);

  const handleLoginSuccess = (userData) => {
    setIsAuthenticated(true);
    setUser(userData || null);
    setCurrentPage("dashboard");
  };

  const handleLogout = () => {
    localStorage.removeItem("accessToken");
    setIsAuthenticated(false);
    setUser(null);
    navigate("login");
  };

  useEffect(() => {
    const token = localStorage.getItem("accessToken");
    if (!token) return;
    apiRequest("/auth/users/me/", "GET", null, token)
      .then((data) => {
        setUser(data);
        setIsAuthenticated(true);
        setCurrentPage("dashboard");
      })
      .catch(() => {
        localStorage.removeItem("accessToken");
        setIsAuthenticated(false);
        setCurrentPage("login");
      });
  }, []);

  const renderPage = () => {
    if (isAuthenticated) {
      if (currentPage === "dashboard") return <DashboardPage user={user} />;
      return <DashboardPage user={user} />;
    } else {
      if (currentPage === "register")
        return <RegisterPage navigate={navigate} />;
      return <LoginPage navigate={navigate} onSuccess={handleLoginSuccess} />;
    }
  };

  return (
    <div className="min-h-screen bg-gray-900">
      <Header navigate={navigate} isAuthenticated={isAuthenticated} onLogout={handleLogout} />
      <main className="flex-grow flex items-center justify-center p-6">
        {renderPage()}
      </main>
    </div>
  );
};

export default App;

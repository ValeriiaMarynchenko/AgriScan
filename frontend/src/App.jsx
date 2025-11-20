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

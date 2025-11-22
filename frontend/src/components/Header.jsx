import React from "react";

const Header = ({ navigate, isAuthenticated, onLogout }) => {
  return (
    <header className="fixed top-0 right-0 p-4 text-green-400 text-lg z-50 bg-black/60 backdrop-blur-sm shadow-[0_0_15px_#00ff99]">
      <div className="flex items-center justify-end space-x-6">
        <div
          className="font-bold cursor-pointer hover:text-green-300 transition"
          onClick={() => navigate("dashboard")}
        >
          AgriScan
        </div>
        <nav className="space-x-4">
          {isAuthenticated ? (
            <>
              <button className="hover:text-green-300 transition" onClick={() => navigate("dashboard")}>
                Dashboard
              </button>
              <button onClick={onLogout} className="text-red-400 hover:text-red-300 transition">
                Вийти
              </button>
            </>
          ) : (
            <>
              <button className="hover:text-green-300 transition" onClick={() => navigate("login")}>
                Увійти
              </button>
              <button className="hover:text-green-300 transition" onClick={() => navigate("register")}>
                Реєстрація
              </button>
            </>
          )}
        </nav>
      </div>
    </header>
  );
};

export default Header;
// // /profile         //go to profile
// // /dashboard       //go to main page
// // /login           //go to login page
//
// //TODO Навігаційна Панель. Відображається на головних
// // сторінках. Містить посилання на Профіль/Головну та
// // кнопку "Вихід", яка викликає logout() з useAuth.
//
//
// import {Home} from "lucide-react";
//
// export const Header = ({ navigate, isAuthenticated, handleLogout }) => {
//     if (!isAuthenticated) return null;
//
//     return (
//         <header className="bg-gray-800 shadow-lg shadow-gray-700/30 sticky top-0 z-10">
//             <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex justify-between items-center">
//                 <h1
//                     className="text-2xl font-bold text-indigo-400 cursor-pointer"
//                     onClick={() => navigate('dashboard')}
//                 >
//                     AgroMonitor AI
//                 </h1>
//                 <nav className="flex items-center space-x-4">
//                     <button
//                         onClick={() => navigate('dashboard')}
//                         className="flex items-center space-x-2 text-gray-300 hover:text-indigo-400 p-2 rounded-lg transition"
//                         title="Головна"
//                     >
//                         <Home className="w-5 h-5" />
//                         <span className="hidden sm:inline">Dashboard</span>
//                     </button>
//                     <button
//                         onClick={handleLogout}
//                         className="flex items-center space-x-2 bg-red-700 text-white px-3 py-2 rounded-lg hover:bg-red-800 transition shadow-md"
//                         title="Вихід"
//                     >
//                         <LogIn className="w-5 h-5" />
//                         <span className="hidden sm:inline">Вихід</span>
//                     </button>
//                 </nav>
//             </div>
//         </header>
//     );
// };
import React from "react";

const Header = ({ navigate, isAuthenticated, onLogout }) => {
  return (
    <header className="bg-gray-800 text-white p-4 shadow">
      <div className="max-w-6xl mx-auto flex justify-between items-center">
        <div className="font-bold text-lg cursor-pointer" onClick={() => navigate("dashboard")}>AgriScan</div>
        <nav className="space-x-4">
          {isAuthenticated ? (
            <>
              <button onClick={() => navigate("dashboard")}>Dashboard</button>
              <button onClick={onLogout} className="text-red-400">Вийти</button>
            </>
          ) : (
            <>
              <button onClick={() => navigate("login")}>Увійти</button>
              <button onClick={() => navigate("register")}>Реєстрація</button>
            </>
          )}
        </nav>
      </div>
    </header>
  );
};

export default Header;

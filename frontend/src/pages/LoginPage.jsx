// import React, { useState } from "react";
// import { apiRequest } from "../api/api";
// import {Card, inputClass} from "../components/Card";
//
// const LoginPage = ({ handleLoginSuccess }) => {
//     const [email, setEmail] = useState("");
//     const [password, setPassword] = useState("");
//     const [error, setError] = useState("");
//     const [isSubmitting, setIsSubmitting] = useState(false);
//
//     const handleLogin = async (e) => {
//         e.preventDefault();
//         setError("");
//         setIsSubmitting(true);
//
//         try {
//             // 🔹 Запит JWT токенів
//             const { access, refresh } = await apiRequest("/auth/jwt/create/", "POST", {
//             email,
//             password,
//             });
//
//             localStorage.setItem("accessToken", access);
//             localStorage.setItem("refreshToken", refresh);
//
//             // 🔹 Отримуємо інформацію про користувача
//             const user = await apiRequest("/auth/users/me/", "GET", null, access);
//
//             handleLoginSuccess(user);
//         } catch (err) {
//             setError(err.message || "Невірний логін або пароль.");
//         } finally {
//             setIsSubmitting(false);
//         }
// };
//
//     return (
//         // Головний контейнер з глибоким темним фоном
//         <div className="min-h-screen flex flex-col items-center justify-center bg-gray-900 p-4">
//             <Card title="Вхід до Системи">
//                 <form onSubmit={handleLogin} className="space-y-6">
//                     <div>
//                         <label className="block text-sm font-medium text-gray-300 mb-1">Електронна пошта</label>
//                         <input
//                             type="email"
//                             required
//                             value={email}
//                             onChange={(e) => setEmail(e.target.value)}
//                             placeholder="ваша@пошта.com"
//                             className={inputClass}
//                         />
//                     </div>
//                     <div>
//                         <label className="block text-sm font-medium text-gray-300 mb-1">Пароль</label>
//                         <input
//                             type="password"
//                             required
//                             value={password}
//                             onChange={(e) => setPassword(e.target.value)}
//                             placeholder="********"
//                             className={inputClass}
//                         />
//                     </div>
//
//                     {/* Повідомлення про помилку в темному стилі */}
//                     {error && (
//                         <div className="text-sm text-red-400 p-3 bg-red-900 border border-red-700 rounded-lg flex items-center">
//                             {error}
//                         </div>
//                     )}
//
//                     <button
//                         type="submit"
//                         disabled={isSubmitting}
//                         className="w-full flex justify-center items-center py-3 px-4 border border-transparent rounded-lg shadow-lg text-white font-semibold bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 focus:ring-offset-gray-900 disabled:opacity-50 transition duration-200"
//                     >
//                         {isSubmitting ? (
//                             <>
//                                 <LogIn className="w-5 h-5 mr-2 animate-pulse" />
//                                 Вхід...
//                             </>
//                         ) : (
//                             <>
//                                 <LogIn className="w-5 h-5 mr-2" />
//                                 Увійти
//                             </>
//                         )}
//                     </button>
//                 </form>
//
//                 {/* Додаткові посилання */}
//                 <div className="mt-4 text-center space-y-2">
//                     <p className="text-sm text-gray-400">
//                         <span
//                             onClick={() => console.log('Перехід до відновлення пароля')}
//                             className="font-medium text-indigo-400 hover:text-indigo-300 cursor-pointer transition"
//                         >
//                             Забули пароль?
//                         </span>
//                     </p>
//                     <p className="text-sm text-gray-400">
//                         Немає облікового запису?{' '}
//                         <span
//                             onClick={() => navigate('register')}
//                             className="font-medium text-green-400 hover:text-green-300 cursor-pointer transition"
//                         >
//                             Зареєструватися
//                         </span>
//                     </p>
//                 </div>
//             </Card>
//         </div>
//     );
// };
//
// export default LoginPage;
import React, { useState } from "react";
import Card from "../components/Card.jsx";
import { apiRequest } from "../api/api";

const LoginPage = ({ navigate, onSuccess }) => {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      const data = await apiRequest("/auth/jwt/create/", "POST", { email, password });
      // Djoser/simplejwt return: { access: "...", refresh: "..." }
      if (data.access) {
        localStorage.setItem("accessToken", data.access);
        // Отримати дані користувача
        const user = await apiRequest("/auth/users/me/", "GET", null, data.access);
        onSuccess && onSuccess(user);
      } else {
        throw new Error("No access token received");
      }
    } catch (err) {
      setError(err.message || "Помилка входу");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center p-4">
      <Card title="Увійти в акаунт">
        {error && <div className="text-sm text-red-400 bg-red-900 p-3 rounded-lg mb-3">{error}</div>}
        <form onSubmit={handleSubmit} className="space-y-4">
          <input type="email" required placeholder="Електронна пошта" value={email} onChange={(e) => setEmail(e.target.value)} className="w-full p-3 bg-gray-700 text-white rounded-lg" />
          <input type="password" required placeholder="Пароль" value={password} onChange={(e) => setPassword(e.target.value)} className="w-full p-3 bg-gray-700 text-white rounded-lg" />
          <button type="submit" disabled={loading} className="w-full bg-indigo-600 hover:bg-indigo-700 text-white py-3 rounded-lg font-semibold">
            {loading ? "Вхід..." : "Увійти"}
          </button>
        </form>
        <div className="mt-4 text-center">
          <p className="text-sm text-gray-400">
            Немає акаунту?{" "}
            <span onClick={() => navigate("register")} className="text-indigo-400 hover:text-indigo-300 cursor-pointer">Зареєструватися</span>
          </p>
        </div>
      </Card>
    </div>
  );
};

export default LoginPage;

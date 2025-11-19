// //POST /auth/users/
//
// import React, { useState } from "react";
// import { apiRequest } from "../api/api";
// import { Card, inputClass } from '../components/Card'
//
//
// const RegisterPage = ({ navigate }) => {
//     const [email, setEmail] = useState("");
//     const [password, setPassword] = useState("");
//     const [rePassword, setRePassword] = useState("");
//     const [error, setError] = useState("");
//     const [message, setMessage] = useState("");
//     const [isSubmitting, setIsSubmitting] = useState(false);
//
//     const handleRegister = async (e) => {
//     e.preventDefault();
//     setError("");
//     setMessage("");
//     setIsSubmitting(true);
//
//     if (password.length < 8) {
//           setError("Пароль має містити щонайменше 8 символів.");
//           setIsSubmitting(false);
//           return;
//     }
//     if (password !== rePassword) {
//           setError("Паролі не збігаються.");
//           setIsSubmitting(false);
//           return;
//     }
//
//     try {
//         await apiRequest("/auth/users/", "POST", {
//             email,
//             password,
//             re_password: rePassword, // Djoser вимагає це поле, якщо USER_CREATE_PASSWORD_RETYPE = True
//         });
//
//         setMessage("Реєстрація успішна! Тепер увійдіть.");
//         setTimeout(() => navigate("login"), 2000);
//     } catch (err) {
//         setError(err.message);
//     } finally {
//         setIsSubmitting(false);
//     }
//     };
//
//
//     return (
//         // Головний контейнер з глибоким темним фоном
//         <div className="min-h-screen flex flex-col items-center justify-center bg-gray-900 p-4">
//             <Card title="Створити Обліковий Запис">
//                 {/* Повідомлення про успіх */}
//                 {message && (
//                     <div className="text-sm text-green-400 p-3 bg-green-900 border border-green-700 rounded-lg flex items-center">
//                         {message}
//                     </div>
//                 )}
//
//                 {/* Повідомлення про помилку */}
//                 {error && (
//                     <div className="text-sm text-red-400 p-3 bg-red-900 border border-red-700 rounded-lg flex items-center">
//                         {error}
//                     </div>
//                 )}
//
//                 <form onSubmit={handleRegister} className="space-y-6">
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
//                         <label className="block text-sm font-medium text-gray-300 mb-1">Пароль (мін. 8 символів)</label>
//                         <input
//                             type="password"
//                             required
//                             value={password}
//                             onChange={(e) => setPassword(e.target.value)}
//                             placeholder="********"
//                             className={inputClass}
//                         />
//                     </div>
//                     <div>
//                         <label className="block text-sm font-medium text-gray-300 mb-1">Повторіть Пароль</label>
//                         <input
//                             type="password"
//                             required
//                             value={rePassword}
//                             onChange={(e) => setRePassword(e.target.value)}
//                             placeholder="********"
//                             className={inputClass}
//                         />
//                          {password !== rePassword && rePassword.length > 0 && (
//                             <p className="mt-1 text-xs text-red-400">Паролі не збігаються.</p>
//                         )}
//                     </div>
//
//                     <button
//                         type="submit"
//                         disabled={isSubmitting || password !== rePassword || password.length < 8}
//                         className="w-full flex justify-center items-center py-3 px-4 border border-transparent rounded-lg shadow-lg text-white font-semibold bg-green-600 hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500 focus:ring-offset-gray-900 disabled:opacity-50 transition duration-200"
//                     >
//                         {isSubmitting ? (
//                             <>
//                                 <UserPlus className="w-5 h-5 mr-2 animate-pulse" />
//                                 Реєстрація...
//                             </>
//                         ) : (
//                             <>
//                                 <UserPlus className="w-5 h-5 mr-2" />
//                                 Зареєструватися
//                             </>
//                         )}
//                     </button>
//                 </form>
//
//                 {/* Посилання на вхід */}
//                 <div className="mt-4 text-center">
//                     <p className="text-sm text-gray-400">
//                         Вже є обліковий запис?{' '}
//                         <span
//                             onClick={() => navigate('login')}
//                             className="font-medium text-indigo-400 hover:text-indigo-300 cursor-pointer transition"
//                         >
//                             Увійти
//                         </span>
//                     </p>
//                 </div>
//             </Card>
//         </div>
//     );
// };
//
// export default RegisterPage;
import React, { useState } from "react";
import Card from "../components/Card.jsx";
import { apiRequest } from "../api/api";

const RegisterPage = ({ navigate }) => {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [rePassword, setRePassword] = useState("");
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleRegister = async (e) => {
    e.preventDefault();
    setError("");
    setMessage("");
    setIsSubmitting(true);

    if (password.length < 8) {
      setError("Пароль має містити щонайменше 8 символів.");
      setIsSubmitting(false);
      return;
    }
    if (password !== rePassword) {
      setError("Паролі не збігаються.");
      setIsSubmitting(false);
      return;
    }

    try {
      await apiRequest("/auth/users/", "POST", {
        email,
        password,
        re_password: rePassword,
      });

      setMessage("Реєстрація успішна! Тепер увійдіть.");
      setTimeout(() => navigate("login"), 1500);
    } catch (err) {
      setError(err.message || "Помилка реєстрації");
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-4">
      <Card title="Створити Обліковий Запис">
        {message && <div className="text-sm text-green-400 bg-green-900 p-3 rounded-lg mb-3">{message}</div>}
        {error && <div className="text-sm text-red-400 bg-red-900 p-3 rounded-lg mb-3">{error}</div>}

        <form onSubmit={handleRegister} className="space-y-4">
          <input type="email" required placeholder="Електронна пошта" value={email} onChange={(e) => setEmail(e.target.value)} className="w-full p-3 bg-gray-700 text-white rounded-lg" />
          <input type="password" required placeholder="Пароль (мін. 8 символів)" value={password} onChange={(e) => setPassword(e.target.value)} className="w-full p-3 bg-gray-700 text-white rounded-lg" />
          <input type="password" required placeholder="Повторіть пароль" value={rePassword} onChange={(e) => setRePassword(e.target.value)} className="w-full p-3 bg-gray-700 text-white rounded-lg" />
          <button type="submit" disabled={isSubmitting} className="w-full bg-green-600 hover:bg-green-700 text-white py-3 rounded-lg font-semibold">
            {isSubmitting ? "Реєстрація..." : "Зареєструватися"}
          </button>
        </form>

        <div className="mt-4 text-center">
          <p className="text-sm text-gray-400">
            Вже є обліковий запис?{" "}
            <span onClick={() => navigate("login")} className="text-indigo-400 hover:text-indigo-300 cursor-pointer">
              Увійти
            </span>
          </p>
        </div>
      </Card>
    </div>
  );
};

export default RegisterPage;

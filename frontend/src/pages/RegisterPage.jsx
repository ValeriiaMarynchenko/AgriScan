import React, { useState } from "react";
// Припускаємо, що Card та apiRequest доступні
import Card from "../components/Card.jsx";
import { apiRequest } from "../api/api";


const RegisterPage = ({ navigate }) => {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [rePassword, setRePassword] = useState("");
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [fullName, setFullName] = useState("");

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
    
    const registrationEndpoint = "/api/v1/auth/users/";

    try {
      await apiRequest(registrationEndpoint, "POST", {
        email,
        password,
        re_password: rePassword,
        full_name: fullName,
      });

      setMessage("Реєстрація успішна! Тепер увійдіть.");
      setTimeout(() => {
          // Припускаємо, що navigate("login") працює
          if (navigate) navigate("login");
          else console.log("Navigation to login skipped.");
      }, 1500);

    } catch (err) {
      // Якщо це помилка "Failed to fetch", вона може бути обгорнута тут.
      setError(err.message || "Помилка реєстрації. Перевірте підключення до API (CORS/URL).");
    } finally {
      setIsSubmitting(false);
    }
  };

  // Заглушка для navigate
  const internalNavigate = (target) => {
      console.log(`Navigating to: ${target}`);
      alert(`Успішно зареєстровано. Навігація до: ${target}`);
  };
  const finalNavigate = navigate || internalNavigate;


  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-4 bg-gray-900">
      <Card title="Створити Обліковий Запис">
        {message && <div className="text-sm text-green-400 bg-green-900 p-3 rounded-lg mb-3">{message}</div>}
        {error && <div className="text-sm text-red-400 bg-red-900 p-3 rounded-lg mb-3">{error}</div>}

        <form onSubmit={handleRegister} className="space-y-4">
          <input type="text" required placeholder="Повне ім'я" value={fullName} onChange={(e) => setFullName(e.target.value)} className="w-full p-3 bg-gray-700 text-white rounded-lg focus:ring-indigo-500 focus:border-indigo-500 border border-gray-600" aria-label="Повне ім'я"/>
          <input type="email" required placeholder="Електронна пошта" value={email} onChange={(e) => setEmail(e.target.value)} className="w-full p-3 bg-gray-700 text-white rounded-lg focus:ring-indigo-500 focus:border-indigo-500 border border-gray-600" aria-label="Електронна пошта"/>
          <input type="password" required placeholder="Пароль (мін. 8 символів)" value={password} onChange={(e) => setPassword(e.target.value)} className="w-full p-3 bg-gray-700 text-white rounded-lg focus:ring-indigo-500 focus:border-indigo-500 border border-gray-600" aria-label="Пароль"/>
          <input type="password" required placeholder="Повторіть пароль" value={rePassword} onChange={(e) => setRePassword(e.target.value)} className="w-full p-3 bg-gray-700 text-white rounded-lg focus:ring-indigo-500 focus:border-indigo-500 border border-gray-600" aria-label="Повторіть пароль"/>
          <button type="submit" disabled={isSubmitting} className="w-full bg-indigo-600 hover:bg-indigo-700 text-white py-3 rounded-lg font-semibold transition duration-200 disabled:opacity-50">
            {isSubmitting ? "Реєстрація..." : "Зареєструватися"}
          </button>
        </form>

        <div className="mt-4 text-center">
          <p className="text-sm text-gray-400">
            Вже є обліковий запис?{" "}
            <span onClick={() => finalNavigate("login")} className="text-indigo-400 hover:text-indigo-300 cursor-pointer">
              Увійти
            </span>
          </p>
        </div>
      </Card>
    </div>
  );
};

export default RegisterPage;
// //POST /auth/users/

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

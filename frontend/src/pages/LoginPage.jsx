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

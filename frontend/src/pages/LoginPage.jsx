import React, { useState } from "react";
import Card from "../components/Card.jsx"; // Припускаємо, що Card доступний
import { apiRequest } from "../api/api"; // Припускаємо, що apiRequest доступний


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

      if (data.access_token) {

        // Зберігаємо токен
        localStorage.setItem("accessToken", data.access_token);
        const accessToken = data.access_token;

        // 2. Отримати дані користувача (використовуємо новий токен)
        const user = await apiRequest("/auth/users/me/", "GET", null, accessToken);

        // 3. Успіх
        onSuccess && onSuccess(user);

      } else {
        // Логіка для випадку, якщо API успішно відповів, але токена немає
        throw new Error("API не повернув токен доступу.");
      }
    } catch (err) {
      // Додаємо більш детальне повідомлення для "Failed to fetch"
      let errorMessage = err.message || "Невідома помилка входу.";

      // Якщо це помилка мережі, повідомте користувача про можливі проблеми з CORS/URL.
      if (errorMessage.includes("Failed to fetch") || errorMessage.includes("NetworkError")) {
          errorMessage = "Помилка підключення до сервера (Failed to fetch). Перевірте, чи запущений бекенд і чи правильно налаштований API_BASE_URL.";
      }

      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  const internalNavigate = (target) => console.log(`Navigating to: ${target}`);
  const finalNavigate = navigate || internalNavigate;


  return (
    <div className="min-h-screen flex items-center justify-center p-4 bg-gray-900">
      <Card title="Увійти в акаунт">
        {error && <div className="text-sm text-red-400 bg-red-900 p-3 rounded-lg mb-3">{error}</div>}
        <form onSubmit={handleSubmit} className="space-y-4">
          <input type="email" required placeholder="Електронна пошта" value={email} onChange={(e) => setEmail(e.target.value)} className="w-full p-3 bg-gray-700 text-white rounded-lg focus:ring-indigo-500 focus:border-indigo-500 border border-gray-600" aria-label="Електронна пошта"/>
          <input type="password" required placeholder="Пароль" value={password} onChange={(e) => setPassword(e.target.value)} className="w-full p-3 bg-gray-700 text-white rounded-lg focus:ring-indigo-500 focus:border-indigo-500 border border-gray-600" aria-label="Пароль"/>
          <button type="submit" disabled={loading} className="w-full bg-indigo-600 hover:bg-indigo-700 text-white py-3 rounded-lg font-semibold transition duration-200 disabled:opacity-50">
            {loading ? "Вхід..." : "Увійти"}
          </button>
        </form>
        <div className="mt-4 text-center">
          <p className="text-sm text-gray-400">
            Немає акаунту?{" "}
            <span onClick={() => finalNavigate("register")} className="text-indigo-400 hover:text-indigo-300 cursor-pointer">Зареєструватися</span>
          </p>
        </div>
      </Card>
    </div>
  );
};

export default LoginPage;
export const TOKEN_KEY = 'agriscan_access_token';
export const REFRESH_KEY = 'agriscan_refresh_token';

export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export async function apiRequest(path, method = "GET", body = null, token = null) {
  const headers = { "Content-Type": "application/json" };
  if (token) headers["Authorization"] = `Bearer ${token}`;

  const res = await fetch(`${API_BASE_URL}${path}`, {
    method,
    headers,
    body: body ? JSON.stringify(body) : null,
  });

  if (!res.ok) {
    // намагаємось прочитати JSON помилку, інакше кидаємо статус
    let errorText = res.statusText;
    try {
      const errJson = await res.json();
      // Djoser/DRF часто повертає { "detail": "..." } або поле з помилками
      errorText = errJson.detail || JSON.stringify(errJson);
    } catch (e) {}
    throw new Error(errorText || `HTTP ${res.status}`);
  }

  if (res.status === 204) return null;
  return res.json();
}

/**
 * secureFetch - обгортка над стандартним fetch.
 * Автоматично додає токен авторизації та базовий URL.
 * * @param {string} endpoint - частина шляху, наприклад '/auth/users/me/'
 * @param {object} options - налаштування запиту (method, body, headers...)
 * @param {boolean} requiresAuth - чи додавати токен (за замовчуванням true)
 */
export const secureFetch = async (endpoint, options = {}, requiresAuth = true) => {
    // 1. Формуємо повний URL
    // Видаляємо зайві слеші, щоб не було http://loc..//api
    const url = `${BASE_URL}${endpoint}`;

    // 2. Налаштовуємо заголовки
    const headers = {
        'Content-Type': 'application/json',
        ...options.headers, // Додаємо кастомні заголовки, якщо вони передані
    };

    // 3. Додаємо токен, якщо потрібно
    if (requiresAuth) {
        const token = localStorage.getItem(TOKEN_KEY);
        if (token) {
            headers['Authorization'] = `Bearer ${token}`;
        }
    }

    // 4. Формуємо конфігурацію запиту
    const config = {
        ...options,
        headers,
    };

    // 5. Виконуємо запит
    const response = await fetch(url, config);

    // 6. (Опціонально) Обробка 401 Unauthorized (якщо токен протух)
    if (response.status === 401 && requiresAuth) {
        // Тут можна додати логіку оновлення токена (refresh token flow)
        // Або просто очистити сторінку
        // localStorage.removeItem(TOKEN_KEY);
        // window.location.href = '/login';
        throw new Error('Unauthorized');
    }

    if (!response.ok) {
        // Спробуємо отримати текст помилки від сервера
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Request failed with status ${response.status}`);
    }

    return response;
};

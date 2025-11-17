//
//
//
// // Base URL для вашого Django бекенду
// export const API_BASE_URL = "http://localhost/api"; //TODO change to real and add it to .conf
//
//
// // Key для зберігання токенів у локальному сховищі
// export const TOKEN_KEY = 'accessToken';  //TODO change and add it to .conf
// export const REFRESH_KEY = 'refreshToken';  //TODO change and add it to .conf
//
//
// export async function apiRequest(endpoint, method = "GET", body = null, token = null) {
//     const headers = {"Content-Type": "application/json",};
//     if (token) headers["Authorization"] = `Bearer ${token}`;
//
//     const response = await fetch(`${API_BASE_URL}${endpoint}`, {
//         method,
//         headers,
//         body: body ? JSON.stringify(body) : null,
//     });
//
//     const data = await response.json().catch(() => ({}));
//
//     if (!response.ok) {
//         throw new Error(data.detail || data.error || "Помилка запиту");
//     }
//
//     return data;
// }
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

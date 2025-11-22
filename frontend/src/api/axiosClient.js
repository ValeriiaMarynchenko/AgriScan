import axios from 'axios';

// Отримуємо URL залежно від збирача (Vite або CRA)
const baseURL = import.meta.env.VITE_API_URL || process.env.REACT_APP_API_URL;

const axiosClient = axios.create({
  baseURL: baseURL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Автоматичне додавання токена до кожного запиту (якщо він є)
axiosClient.interceptors.request.use((config) => {
  const token = localStorage.getItem('token'); // Або sessionStorage
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Глобальна обробка помилок
axiosClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response && error.response.status === 401) {
      // Якщо токен недійсний — можна перенаправити на логін
      console.error('Unauthorized, logging out...');
      // localStorage.removeItem('token');
      // window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

export default axiosClient;
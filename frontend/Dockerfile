# СТАДІЯ 1: ЗБІРКА (BUILDER)
# Використовуємо офіційний образ Node.js для встановлення залежностей та збірки.
FROM node:20-alpine AS builder

# Встановлюємо робочу директорію в контейнері.
WORKDIR /app

# Копіюємо файли конфігурації залежностей.
COPY package*.json ./

# Встановлюємо залежності.
RUN npm install

# Копіюємо решту коду застосунку.
COPY . .

# Запускаємо команду збірки.
# Для Vite це створить теку 'dist'.
RUN npm run build

# СТАДІЯ 2: СЕРВІС (PRODUCTION)
# Використовуємо легкий образ Nginx для обслуговування статичних файлів.
FROM nginx:stable-alpine AS production

# Копіюємо кастомний конфігураційний файл Nginx для підтримки SPA.
# Цей файл (frontend/nginx.conf) потрібен для коректної маршрутизації React.
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Копіюємо зібрані статичні файли з першої стадії ('builder')
# до директорії Nginx. Зверніть увагу: шлях змінено на /app/dist для Vite.
COPY --from=builder /app/dist /usr/share/nginx/html

# Відкриваємо порт 80, на якому працюватиме Nginx.
EXPOSE 80

# Команда для запуску Nginx.
CMD ["nginx", "-g", "daemon off;"]
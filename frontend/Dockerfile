FROM ubuntu:latest
LABEL authors="vlerk"

ENTRYPOINT ["top", "-b"]

# --- Этап 1: Сборка приложения ---
FROM node:18-alpine AS build

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm install

COPY . .
# Создание production-сборки
RUN rm /etc/nginx/conf.d/default.conf

# --- Этап 2: Запуск с помощью Nginx (или просто Node, если нужно) ---
FROM nginx:alpine

# Копируем собранный код в директорию Nginx
COPY --from=build /app/dist /usr/share/nginx/html
#'dist' может быть 'build'

# Копируем кастомную конфигурацию Nginx (если есть)
# COPY nginx.conf /etc/nginx/conf.d/default.conf

# Сервер запускается командой из базового образа Nginx
CMD ["nginx", "-g", "daemon off;"]
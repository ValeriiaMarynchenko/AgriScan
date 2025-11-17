project-agro-monitoring/
├── backend/          # Django/Flask API
├── frontend/         # React/Vue.js застосунок
├── ai_service/       # Сервіс Комп'ютерного Зору (FastAPI/Flask)
├── infra/            # Конфігурації розгортання (Docker, Kubernetes)
├── docs/             # Документація проєкту
├── .gitignore
├── README.md
└── docker-compose.yml



frontend/src/
├── api/             # Функції для взаємодії з Django API (Axios)
├── components/      # Перевикористовувані компоненти (Buttons, Inputs)
├── features/
│   ├── Auth/        # Компоненти для авторизації, реєстрації, профілю
│   └── Analysis/    # Компоненти для завантаження фото та відображення результатів
├── pages/           # Компоненти, прив'язані до маршрутів (Login, Register, Dashboard)
├── context/         # Стан користувача (AuthContext)
└── App.html           # Головний роутинг



backend/
├── venv/             # Віртуальне оточення Python
├── project_core/     # Головна директорія проєкту (settings.py, urls.py)
│   ├── settings/     # Конфігурації (base.py, prod.py, dev.py)
│   └── urls.py
├── apps/             # Директорія для логічних додатків
│   ├── users/        # Керування користувачами та Auth (JWT, OAuth)
│   │   ├── migrations/
│   │   ├── models.py
│   │   └── views.py (API Endpoints)
│   ├── fields/       # Керування полями та геопросторовими даними (PostGIS)
│   ├── analysis/     # Логіка для запуску завдань ШІ та зберігання метаданих
│   └── reports/      # Генерація звітів
├── requirements.txt  # Залежності Python
├── manage.py         # Скрипт керування Django
└── Dockerfile


ai_service/
├── venv/
├── models/           # Збережені вагові коефіцієнти моделей (e.g., model.pth)
├── src/
│   ├── processing/   # Функції попередньої/постобробки зображень (GDAL, OpenCV)
│   ├── core/         # Основна логіка моделі (класи моделей U-Net)
│   ├── api.py        # FastAPI/Flask Endpoints для виклику моделі
│   └── utils.py
├── data/             # Приклади, або конфігурація для завантаження даних
├── requirements.txt
└── Dockerfile


infra/
├── docker/
│   ├── db/           # Конфігурація для PostgreSQL/PostGIS
│   └── nginx/        # Конфігурація для реверс-проксі
├── k8s/              # Конфігурації Kubernetes (якщо використовується)
└── logging/          # Конфігурація для логування (ELK stack, Prometheus)



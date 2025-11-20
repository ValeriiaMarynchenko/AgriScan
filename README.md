project-agro-monitoring/
├── backend/          # Django/Flask API
├── frontend/         # React/Vue.js застосунок
├── ai_service/       # Сервіс Комп'ютерного Зору (FastAPI/Flask)
├── infra/            # Конфігурації розгортання (Docker, Kubernetes)
├── docs/             # Документація проєкту
├── .gitignore
├── README.md
└── docker-compose.yml



frontend/
├───public
├───srs
│    │   App.css
│    │   App.jsx
│    │   index.css
│    │   index.html
│    │   main.jsx
│    │   
│    ├───api
│    │       api.js
│    │       
│    ├───assets
│    │       react.svg
│    │       
│    ├───components
│    │       Card.jsx
│    │       Header.jsx
│    │       
│    ├───context
│    │       AuthContext.js
│    │       
│    ├───features
│    │   ├───Analysis
│    │   │       analysisService.js
│    │   │       authService.js
│    │   │       UploadForm.js
│    │   │       
│    │   └───Auth
│    │           AuthProvider.jsx
│    │           
│    └───pages
│            DashboardPage.jsx
│            LoginPage.jsx
│            Profile.jsx
│            RegisterPage.jsx
├─── .gitignoge
├─── Dockerfile
├─── eslint.config.js
├─── index.html
├─── nginx.conf
├─── package.json
├─── package-lock.json
├─── README.md
└─── vite.config.js




backend/
├── venv/
├── models/
│   ├── auth_service.py
│   └── user.py
├── dependencies.py
├── tasks.py
├── settings
├── requirements.txt
├── main.py
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
│   ├── db/           # Конфігурація для MongoDB
│   └── nginx/        # Конфігурація для реверс-проксі
└── logging/          # Конфігурація для логування (ELK stack, Prometheus)



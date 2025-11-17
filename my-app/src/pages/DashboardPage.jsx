// TODO Головна Сторінка. Сторінка,
//  на яку користувач потрапляє після входу.
//  Містить Header.jsx та форму UploadForm.js.

import {useState} from "react";
import {CheckCircle, XCircle} from "lucide-react";

const DashboardPage = () => {
    const [file, setFile] = useState(null);
    const [previewUrl, setPreviewUrl] = useState(null);
    const [analysisStatus, setAnalysisStatus] = useState('IDLE'); // IDLE, UPLOADING, PROCESSING, COMPLETED, FAILED
    const [resultUrl, setResultUrl] = useState(null);
    const [message, setMessage] = useState('Завантажте знімок поля для початку аналізу.');
    const [isSubmitting, setIsSubmitting] = useState(false);

    // Симуляція асинхронного процесу аналізу
    const simulateAnalysis = () => {
        setAnalysisStatus('UPLOADING');
        setMessage('Знімок завантажується...');
        setIsSubmitting(true);

        // 1. Симуляція завантаження (1.5 секунди)
        setTimeout(() => {
            setAnalysisStatus('PROCESSING');
            setMessage('Дані отримано. AI-аналіз розпочато (це може зайняти до 5 секунд)...');

            // 2. Симуляція обробки (5 секунд)
            setTimeout(() => {
                // 3. Симуляція завершення
                setAnalysisStatus('COMPLETED');
                setMessage('Аналіз успішно завершено! Результати готові.');
                setIsSubmitting(false);

                // Симуляція отримання результату (приклад placeholder зображення)
                setResultUrl('https://placehold.co/600x400/10b981/ffffff?text=NDVI+MASK');

            }, 5000); // 5 секунд на обробку

        }, 1500); // 1.5 секунди на завантаження
    };

    const handleFileChange = (e) => {
        const selectedFile = e.target.files[0];
        setFile(selectedFile);

        if (selectedFile) {
            setPreviewUrl(URL.createObjectURL(selectedFile));
            setAnalysisStatus('IDLE');
            setResultUrl(null);
            setMessage(`Файл "${selectedFile.name}" готовий до аналізу.`);
        } else {
            setPreviewUrl(null);
            setMessage('Завантажте знімок поля для початку аналізу.');
        }
    };

    const handleAnalysisSubmit = () => {
        if (!file) {
            setMessage('Будь ласка, оберіть файл зображення.');
            return;
        }
        simulateAnalysis();
    };

    const getStatusColor = (status) => {
        switch (status) {
            case 'COMPLETED': return 'bg-green-900 border-green-700 text-green-400';
            case 'FAILED': return 'bg-red-900 border-red-700 text-red-400';
            case 'UPLOADING':
            case 'PROCESSING': return 'bg-yellow-900 border-yellow-700 text-yellow-400';
            default: return 'bg-blue-900 border-blue-700 text-blue-400';
        }
    };

    return (
        <div className="min-h-screen bg-gray-900 text-white p-8">
            <div className="max-w-7xl mx-auto">
                <h2 className="text-4xl font-bold text-indigo-400 mb-8 border-b border-gray-700 pb-4">
                    Dashboard: AI Аналіз Полів
                </h2>

                <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                    {/* 1. Панель керування та завантаження */}
                    <div className="lg:col-span-1 bg-gray-800 p-6 rounded-xl shadow-xl shadow-gray-700/50 space-y-6 h-min sticky top-24">
                        <h3 className="text-xl font-semibold text-white border-b border-gray-700 pb-2 flex items-center">
                            <Upload className="w-5 h-5 mr-2" /> Завантаження знімка
                        </h3>

                        <label
                            htmlFor="file-upload"
                            className="block w-full text-center py-3 border-2 border-dashed border-indigo-600 rounded-lg cursor-pointer text-gray-300 hover:border-indigo-500 hover:text-white transition bg-gray-700/50"
                        >
                            {file ? `Обрано: ${file.name}` : 'Натисніть або перетягніть файл'}
                        </label>
                        <input
                            id="file-upload"
                            type="file"
                            accept="image/*"
                            onChange={handleFileChange}
                            className="hidden"
                        />

                        {previewUrl && (
                            <div className="mt-4 border border-gray-700 rounded-lg overflow-hidden">
                                <img src={previewUrl} alt="Попередній перегляд" className="w-full h-auto" />
                            </div>
                        )}

                        <button
                            onClick={handleAnalysisSubmit}
                            disabled={!file || isSubmitting}
                            className="w-full flex items-center justify-center py-3 px-4 rounded-xl shadow-md text-white font-bold transition disabled:opacity-50"
                            style={{ backgroundColor: isSubmitting ? '#f59e0b' : '#10b981' }}
                        >
                            {isSubmitting ? (
                                <RefreshCcw className="w-5 h-5 mr-2 animate-spin" />
                            ) : (
                                <Send className="w-5 h-5 mr-2" />
                            )}
                            {isSubmitting ? 'Аналізується...' : 'Почати Аналіз'}
                        </button>
                    </div>

                    {/* 2. Панель статусу та результатів */}
                    <div className="lg:col-span-2 space-y-6">
                         <h3 className="text-2xl font-semibold text-white border-b border-gray-700 pb-2">
                            Статус та Результати
                        </h3>

                        {/* Повідомлення про статус */}
                        <div className={`p-4 rounded-xl border-2 font-medium ${getStatusColor(analysisStatus)}`}>
                            <div className="flex items-center">
                                {analysisStatus === 'COMPLETED' ? <CheckCircle className="w-5 h-5 mr-2" /> :
                                 analysisStatus === 'FAILED' ? <XCircle className="w-5 h-5 mr-2" /> :
                                 isSubmitting ? <RefreshCcw className="w-5 h-5 mr-2 animate-spin" /> :
                                 <Upload className="w-5 h-5 mr-2" />}
                                {message}
                            </div>
                        </div>

                        {/* Відображення результату */}
                        {resultUrl && analysisStatus === 'COMPLETED' && (
                            <div className="border-4 border-green-500 rounded-xl overflow-hidden shadow-xl shadow-green-900/50">
                                <h4 className="text-xl font-bold bg-green-700 text-white p-3">Результат (Маска NDVI) </h4>
                                <img src={resultUrl} alt="Результат аналізу" className="w-full h-auto" />
                            </div>
                        )}

                        {/* Додаткові дані аналізу (симуляція) */}
                        {analysisStatus === 'COMPLETED' && (
                            <div className="bg-gray-800 p-6 rounded-xl shadow-lg space-y-4 shadow-gray-700/30">
                                <h4 className="text-xl font-bold text-white">Ключові показники</h4>
                                <ul className="space-y-2 text-gray-300">
                                    <li className="flex justify-between border-b border-gray-700 pb-1">
                                        <span className="font-medium">Загальна площа:</span>
                                        <span className="text-indigo-400">12.5 Га</span>
                                    </li>
                                    <li className="flex justify-between border-b border-gray-700 pb-1">
                                        <span className="font-medium">Середній індекс NDVI:</span>
                                        <span className="text-green-400">0.82 (Високий)</span>
                                    </li>
                                    <li className="flex justify-between border-b border-gray-700 pb-1">
                                        <span className="font-medium">Зони з низькою вегетацією:</span>
                                        <span className="text-red-400">1.1 Га (8.8%)</span>
                                    </li>
                                </ul>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default DashboardPage;

// Додаємо стилі Tailwind CSS
// export default function AppWrapper() {
//     return (
//         <>
//             <script src="https://cdn.tailwindcss.com"></script>
//             <App />
//         </>
//     );
// }
// }
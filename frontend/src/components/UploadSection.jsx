import React, { useState } from 'react';
import { useAnalyzeImage } from '../hooks/useScan'; // Імпорт нашого хука

const UploadSection = () => {
  const [file, setFile] = useState(null);

  // Витягуємо функцію mutate і стан завантаження
  const { mutate: analyze, isPending, isError, error, data } = useAnalyzeImage();

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleSubmit = () => {
    if (!file) return;

    const formData = new FormData();
    formData.append('image', file); // 'image' має співпадати з назвою поля на бекенді

    analyze(formData); // Викликаємо мутацію
  };

  return (
    <div className="upload-container">
      <input type="file" onChange={handleFileChange} />

      <button onClick={handleSubmit} disabled={isPending || !file}>
        {isPending ? 'Аналізуємо...' : 'Розпочати аналіз'}
      </button>

      {isError && <p style={{color: 'red'}}>Помилка: {error.message}</p>}

      {data && (
        <div className="result">
          <h3>Результат: {data.result}</h3>
          <p>Впевненість: {data.confidence}%</p>
        </div>
      )}
    </div>
  );
};

export default UploadSection;
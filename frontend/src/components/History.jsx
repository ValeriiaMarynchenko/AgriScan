import React from 'react';
import { useScanHistory } from '../hooks/useScan';

const History = () => {
  const { data: history, isLoading, isError } = useScanHistory();

  if (isLoading) return <div className="loader">Завантаження історії...</div>;
  if (isError) return <div className="error">Не вдалося завантажити історію.</div>;

  return (
    <div className="history-list">
      <h2>Історія сканувань</h2>
      <ul>
        {history?.map((item) => (
          <li key={item.id}>
            <span>{item.date}</span> - <strong>{item.disease_name}</strong>
          </li>
        ))}
      </ul>
    </div>
  );
};

export default History;
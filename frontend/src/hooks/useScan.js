import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import axiosClient from '../api/axiosClient';

// 1. Отримання історії сканувань
export const useScanHistory = () => {
  return useQuery({
    queryKey: ['scanHistory'],
    queryFn: async () => {
      const { data } = await axiosClient.get('/scans/history'); // Ваш ендпоінт
      return data;
    },
  });
};

// 2. Відправка фото на аналіз
export const useAnalyzeImage = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (formData) => {
      // Важливо: для файлів axios сам встановить Content-Type,
      // але краще явно передавати FormData
      const { data } = await axiosClient.post('/scans/analyze', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      return data;
    },
    onSuccess: () => {
      // Після успішного аналізу оновлюємо історію автоматично
      queryClient.invalidateQueries(['scanHistory']);
    },
  });
};
//
// // Базові класи для інпутів у темному стилі (загальні)
// export const inputClass = "mt-1 block w-full px-4 py-3 border border-gray-600 rounded-lg shadow-inner focus:ring-indigo-500 focus:border-indigo-500 bg-gray-700 text-white placeholder-gray-400 transition duration-150";
//
// // Компонент Card для обгортки вмісту форми (загальний для входу та реєстрації)
// export const Card = ({ children, title }) => (
//     // Темний фон картки, округлені кути та тінь
//     <div className="w-full max-w-md bg-gray-800 p-8 rounded-xl shadow-2xl shadow-indigo-900/50 space-y-6 transform hover:scale-[1.01] transition duration-300">
//         <h2 className="text-3xl font-extrabold text-indigo-400 text-center flex items-center justify-center space-x-2">
//             <Zap className="w-7 h-7" />
//             <span>{title}</span>
//         </h2>
//         {children}
//     </div>
// );

import React from "react";
import { Zap } from "lucide-react";

const Card = ({ children, title }) => (
  <div className="p-8 bg-[#111111] border border-green-400 rounded-lg text-green-400">
    <h2 className="text-3xl font-extrabold text-indigo-400 text-center flex items-center justify-center space-x-2">
      <Zap className="w-7 h-7" />
      <span>{title}</span>
    </h2>
    {children}
  </div>
);

export default Card;

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

// src/api/endpoints.js
const ENDPOINTS = {
  AUTH: {
    LOGIN: '/auth/login',
    REGISTER: '/auth/register',
  },
  CROPS: {
    GET_ALL: '/crops',
    GET_ONE: (id) => `/crops/${id}`,
  },
  SCAN: '/scan/analyze',
};

export default ENDPOINTS;
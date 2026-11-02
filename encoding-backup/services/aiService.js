import axios from "axios";

const api = axios.create({
  baseURL: "http://localhost:5000/api",
  timeout: 30000,
  headers: {
    "Content-Type": "application/json"
  }
});

export const aiService = {
  getHealth: async () => {
    const response = await api.get("/ai/health");
    return response.data;
  },

  getGPU: async () => {
    const response = await api.get("/ai/gpu");
    return response.data;
  },

  predictEarthquake: async (payload) => {
    const response = await api.post(
      "/ai/predictions/earthquake",
      payload
    );
    return response.data;
  },

  generateCounts: async (payload) => {
    const response = await api.post(
      "/ai/generator/counts",
      payload
    );
    return response.data;
  },

  generateScenarios: async (payload) => {
    const response = await api.post(
      "/ai/scenarios/generate",
      payload
    );
    return response.data;
  }
};

export default api;

import axios from "axios";

const API_BASE = "http://localhost:8000/api";

export const getClientStats = async (clientId) => {
  const res = await axios.get(`${API_BASE}/stats/${clientId}`);
  return res.data;
};

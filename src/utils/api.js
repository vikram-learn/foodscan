/* small axios wrapper for frontend -> backend calls */
import axios from "axios";

const API_BASE = process.env.REACT_APP_API_URL || "http://127.0.0.1:8000";

export async function postScanImage(file) {
  const form = new FormData();
  form.append("file", file);
  const res = await axios.post(`${API_BASE}/api/v1/image`, form, {
    headers: { "Content-Type": "multipart/form-data" },
    timeout: 60_000
  });
  return res.data;
}

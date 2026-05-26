import axios from 'axios';
import type { InferenceResponse, GradCAMResponse } from '@/types/inference';

// Ngrok free tier shows an interstitial HTML page unless this header is sent
const ngrokHeaders = { 'ngrok-skip-browser-warning': '1' };

export async function checkHealth(apiUrl: string): Promise<boolean> {
  try {
    const res = await axios.get(`${apiUrl}/api/health`, {
      timeout: 5000,
      headers: ngrokHeaders,
    });
    return res.data?.status === 'ok';
  } catch {
    return false;
  }
}

export async function runInference(
  apiUrl: string,
  file: File,
  temperature: number = 1.0
): Promise<InferenceResponse> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('temperature', temperature.toString());

  const res = await axios.post(`${apiUrl}/api/inference`, formData, {
    headers: { 'Content-Type': 'multipart/form-data', ...ngrokHeaders },
    timeout: 60000,
  });
  return res.data;
}

export async function recomputeDistribution(
  apiUrl: string,
  file: File,
  temperature: number
): Promise<InferenceResponse> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('temperature', temperature.toString());

  const res = await axios.post(`${apiUrl}/api/distribution`, formData, {
    headers: { 'Content-Type': 'multipart/form-data', ...ngrokHeaders },
    timeout: 30000,
  });
  return res.data;
}

export async function runGradCAM(
  apiUrl: string,
  file: File,
): Promise<GradCAMResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const res = await axios.post(`${apiUrl}/api/gradcam`, formData, {
    headers: { 'Content-Type': 'multipart/form-data', ...ngrokHeaders },
    timeout: 120000,
  });
  return res.data;
}

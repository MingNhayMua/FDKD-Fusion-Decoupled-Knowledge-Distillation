import axios from 'axios';
import type { InferenceResponse, GradCAMData } from '@/types/inference';

export async function checkHealth(apiUrl: string): Promise<boolean> {
  try {
    const res = await axios.get(`${apiUrl}/api/health`, { timeout: 5000 });
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
    headers: { 'Content-Type': 'multipart/form-data' },
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
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 30000,
  });
  return res.data;
}

export async function getGradCAM(
  apiUrl: string,
  file: File
): Promise<GradCAMData> {
  const formData = new FormData();
  formData.append('file', file);

  const res = await axios.post(`${apiUrl}/api/gradcam`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 60000,
  });
  return res.data.gradcam;
}

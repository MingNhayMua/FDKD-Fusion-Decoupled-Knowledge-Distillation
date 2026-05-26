import { create } from 'zustand';
import type { InferenceResponse } from '@/types/inference';

interface AppState {
  // Connection
  apiUrl: string;
  isConnected: boolean;
  setApiUrl: (url: string) => void;
  setConnected: (connected: boolean) => void;

  // Image
  imageFile: File | null;
  imagePreview: string | null;
  setImageFile: (file: File | null) => void;
  setImagePreview: (preview: string | null) => void;

  // Inference
  inferenceResult: InferenceResponse | null;
  isLoading: boolean;
  temperature: number;
  setInferenceResult: (result: InferenceResponse | null) => void;
  setLoading: (loading: boolean) => void;
  setTemperature: (temp: number) => void;

  // DKD tab
  activeTab: 'tckd' | 'nckd' | 'dark';
  setActiveTab: (tab: 'tckd' | 'nckd' | 'dark') => void;
}

export const useStore = create<AppState>((set) => ({
  apiUrl: '',
  isConnected: false,
  setApiUrl: (url) => set({ apiUrl: url.replace(/\/$/, '') }),
  setConnected: (connected) => set({ isConnected: connected }),

  imageFile: null,
  imagePreview: null,
  setImageFile: (file) => set({ imageFile: file }),
  setImagePreview: (preview) => set({ imagePreview: preview }),

  inferenceResult: null,
  isLoading: false,
  temperature: 1.0,
  setInferenceResult: (result) => set({ inferenceResult: result }),
  setLoading: (loading) => set({ isLoading: loading }),
  setTemperature: (temp) => set({ temperature: temp }),

  activeTab: 'tckd',
  setActiveTab: (tab) => set({ activeTab: tab }),
}));

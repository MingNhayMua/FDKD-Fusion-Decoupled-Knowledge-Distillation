"use client";

import { useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { motion } from "framer-motion";
import { Upload, ImageIcon, Loader2 } from "lucide-react";
import { useStore } from "@/hooks/useStore";
import { runInference } from "@/services/api";

export default function ImageUpload() {
  const {
    apiUrl, isConnected, imagePreview, setImageFile, setImagePreview,
    setInferenceResult, setGradcamResult, isLoading, setLoading, temperature,
  } = useStore();

  const onDrop = useCallback(async (files: File[]) => {
    const file = files[0];
    if (!file) return;

    setImageFile(file);
    const reader = new FileReader();
    reader.onload = () => setImagePreview(reader.result as string);
    reader.readAsDataURL(file);

    if (!isConnected || !apiUrl) return;

    setLoading(true);
    setGradcamResult(null);
    try {
      const result = await runInference(apiUrl, file, temperature);
      setInferenceResult(result);
    } catch (err) {
      console.error("Inference failed:", err);
    } finally {
      setLoading(false);
    }
  }, [apiUrl, isConnected, temperature, setImageFile, setImagePreview, setInferenceResult, setGradcamResult, setLoading]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "image/*": [".png", ".jpg", ".jpeg", ".webp"] },
    maxFiles: 1,
    disabled: !isConnected,
  });

  return (
    <section className="section-container py-8">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
      >
        <h2 className="text-xl font-semibold text-center mb-6">
          <ImageIcon className="inline w-5 h-5 mr-2 text-indigo-400" />
          Upload Image
        </h2>

        <div className="max-w-2xl mx-auto">
          <div
            {...getRootProps()}
            className={`glass-card p-10 text-center cursor-pointer transition-all border-2 border-dashed
              ${isDragActive ? "border-indigo-500 bg-indigo-500/5" : "border-white/10 hover:border-white/20"}
              ${!isConnected ? "opacity-50 cursor-not-allowed" : ""}
            `}
          >
            <input {...getInputProps()} />

            {isLoading ? (
              <div className="flex flex-col items-center gap-3">
                <Loader2 className="w-10 h-10 text-indigo-400 animate-spin" />
                <p className="text-sm text-slate-400">Running inference through T → A → S pipeline...</p>
              </div>
            ) : imagePreview ? (
              <div className="flex flex-col items-center gap-4">
                <img
                  src={imagePreview}
                  alt="Uploaded"
                  className="max-h-48 rounded-lg border border-white/10"
                />
                <p className="text-xs text-slate-500">Click or drag to replace</p>
              </div>
            ) : (
              <div className="flex flex-col items-center gap-3">
                <Upload className="w-10 h-10 text-slate-500" />
                <p className="text-sm text-slate-400">
                  {isConnected
                    ? "Drag & drop an image, or click to browse"
                    : "Connect to backend first"}
                </p>
                <p className="text-xs text-slate-600">
                  Tiny ImageNet compatible (64×64 or any size — auto-resized)
                </p>
              </div>
            )}
          </div>
        </div>
      </motion.div>
    </section>
  );
}

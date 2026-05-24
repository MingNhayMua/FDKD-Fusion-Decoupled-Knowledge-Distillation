"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Eye, Loader2 } from "lucide-react";
import { useStore } from "@/hooks/useStore";
import { getGradCAM } from "@/services/api";

const MODELS = [
  { key: "teacher", label: "Teacher (Swin-B)", color: "#6366f1" },
  { key: "assistant", label: "Assistant (ResNet-152)", color: "#8b5cf6" },
  { key: "student", label: "Student (ResNet-18)", color: "#10b981" },
] as const;

export default function GradCAMViewer() {
  const { apiUrl, imageFile, gradcamResult, setGradcamResult } = useStore();
  const [loading, setLoading] = useState(false);
  const [opacity, setOpacity] = useState(0.6);

  const handleGenerate = async () => {
    if (!apiUrl || !imageFile) return;
    setLoading(true);
    try {
      const result = await getGradCAM(apiUrl, imageFile);
      setGradcamResult(result);
    } catch (err) {
      console.error("Grad-CAM failed:", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="section-container">
      <h2 className="text-2xl font-bold text-center mb-2">
        <Eye className="inline w-6 h-6 mr-2 text-cyan-400" />
        Attention Visualization (Grad-CAM)
      </h2>
      <p className="text-sm text-slate-500 text-center mb-6">
        Where each model focuses when making predictions
      </p>

      {!gradcamResult ? (
        <div className="text-center">
          <button
            onClick={handleGenerate}
            disabled={loading || !apiUrl || !imageFile}
            className="px-6 py-3 bg-cyan-600 hover:bg-cyan-500 disabled:bg-slate-700
                       rounded-xl text-sm font-medium transition-colors inline-flex items-center gap-2"
          >
            {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Eye className="w-4 h-4" />}
            Generate Grad-CAM Heatmaps
          </button>
        </div>
      ) : (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
          {/* Opacity control */}
          <div className="flex items-center justify-center gap-3 mb-6">
            <span className="text-xs text-slate-500">Heatmap Opacity</span>
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={opacity}
              onChange={(e) => setOpacity(parseFloat(e.target.value))}
              className="w-32 accent-cyan-400"
            />
            <span className="text-xs font-mono text-cyan-400">{opacity.toFixed(1)}</span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {MODELS.map((m, i) => {
              const src = gradcamResult[m.key as keyof typeof gradcamResult];
              return (
                <motion.div
                  key={m.key}
                  className="glass-card p-4 text-center"
                  style={{ borderColor: `${m.color}30` }}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.15 }}
                >
                  <div className="text-sm font-semibold mb-3" style={{ color: m.color }}>
                    {m.label}
                  </div>
                  {src ? (
                    <img
                      src={`data:image/png;base64,${src}`}
                      alt={`${m.label} Grad-CAM`}
                      className="w-full rounded-lg"
                      style={{ opacity }}
                    />
                  ) : (
                    <div className="h-48 flex items-center justify-center text-slate-600 text-sm">
                      Not available
                    </div>
                  )}
                </motion.div>
              );
            })}
          </div>
        </motion.div>
      )}
    </section>
  );
}

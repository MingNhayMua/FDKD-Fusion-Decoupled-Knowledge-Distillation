"use client";

import { useState, useCallback } from "react";
import { motion } from "framer-motion";
import { ScanEye } from "lucide-react";
import { useStore } from "@/hooks/useStore";
import { runGradCAM } from "@/services/api";
import type { GradCAMResponse } from "@/types/inference";

const LABELS: Record<string, string> = {
  teacher: "Teacher (Swin-B)", dkd: "DKD", takd: "TAKD", baseline: "Baseline",
};
const COLORS: Record<string, string> = {
  teacher: "#6366f1", dkd: "#8b5cf6", takd: "#f59e0b", baseline: "#10b981",
};

export default function GradCAMSection() {
  const { inferenceResult, apiUrl, imageFile } = useStore();
  const [gradcamData, setGradcamData] = useState<GradCAMResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  console.log("[GradCAM] rendering, hasResult:", !!inferenceResult, "hasApiUrl:", !!apiUrl, "hasImage:", !!imageFile, "hasGradcam:", !!gradcamData, "loading:", loading);

  const handleRun = useCallback(async () => {
    console.log("[GradCAM] button clicked, apiUrl:", apiUrl, "imageFile:", !!imageFile);
    if (!apiUrl || !imageFile) {
      console.log("[GradCAM] missing apiUrl or imageFile, aborting");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      console.log("[GradCAM] calling API...");
      const result = await runGradCAM(apiUrl, imageFile);
      console.log("[GradCAM] API success, heatmaps:", Object.keys(result.heatmaps), "data len:", JSON.stringify(result.heatmaps).length);
      setGradcamData(result);
      console.log("[GradCAM] state set, entries:", Object.entries(result.heatmaps).map(([k, v]) => `${k}:${v.length}`));
    } catch (err: unknown) {
      console.error("[GradCAM] API error:", err);
      setError(err instanceof Error ? err.message : "GradCAM failed");
    } finally {
      setLoading(false);
    }
  }, [apiUrl, imageFile]);

  if (!inferenceResult) return null;

  return (
    <section className="section-container">
      <h2 className="text-2xl font-bold text-center mb-2">GradCAM Visualization</h2>
      <p className="text-sm text-slate-500 text-center mb-6">
        See which image regions each model focuses on for its prediction
      </p>

      <div className="flex justify-center mb-8">
        <motion.button
          onClick={handleRun}
          disabled={loading}
          className="flex items-center gap-2 px-6 py-3 bg-indigo-600/20 border border-indigo-500/30
                     rounded-xl text-indigo-300 font-medium hover:bg-indigo-600/30
                     disabled:opacity-50 disabled:cursor-not-allowed transition-all"
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
        >
          <ScanEye className={`w-5 h-5 ${loading ? "animate-pulse" : ""}`} />
          {loading ? "Computing GradCAM..." : "Generate GradCAM"}
        </motion.button>
      </div>

      {error && (
        <div className="text-center text-rose-400 text-sm mb-4">{error}</div>
      )}

      {gradcamData && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {Object.entries(gradcamData.heatmaps).map(([role, img], i) => (
            <motion.div
              key={role}
              className="glass-card p-4"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.1 }}
            >
              <div
                className="flex items-center gap-2 mb-3"
                style={{ color: COLORS[role] || "#94a3b8" }}
              >
                <div
                  className="w-3 h-3 rounded-full"
                  style={{ background: COLORS[role] || "#94a3b8" }}
                />
                <span className="font-semibold text-sm">
                  {LABELS[role] || role}
                </span>
              </div>
              <div className="rounded-xl overflow-hidden border border-white/10">
                <img
                  src={`data:image/png;base64,${img}`}
                  alt={`GradCAM ${role}`}
                  className="w-full h-auto"
                />
              </div>
            </motion.div>
          ))}
        </div>
      )}
    </section>
  );
}

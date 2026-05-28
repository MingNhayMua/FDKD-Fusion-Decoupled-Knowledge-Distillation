"use client";

import { useState, useCallback } from "react";
import { motion } from "framer-motion";
import { Thermometer } from "lucide-react";
import { useStore } from "@/hooks/useStore";
import { recomputeDistribution } from "@/services/api";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend,
} from "recharts";

const COLORS: Record<string, string> = {
  teacher: "#6366f1", dkd: "#8b5cf6", takd: "#f59e0b", baseline: "#10b981",
};
const LABELS: Record<string, string> = {
  teacher: "Teacher", dkd: "DKD", takd: "TAKD", baseline: "Baseline",
};

export default function DistributionSection() {
  const {
    inferenceResult, setInferenceResult, temperature, setTemperature,
    apiUrl, imageFile,
  } = useStore();
  const [updating, setUpdating] = useState(false);

  const handleTempChange = useCallback(
    async (newTemp: number) => {
      setTemperature(newTemp);
      if (!apiUrl || !imageFile) return;
      setUpdating(true);
      try {
        const result = await recomputeDistribution(apiUrl, imageFile, newTemp);
        setInferenceResult({
          ...inferenceResult!,
          temperature: newTemp,
          models: { ...inferenceResult!.models, ...result.models },
          dkd: result.dkd,
          metrics: result.metrics,
        });
      } catch (err) {
        console.error("Temperature update failed:", err);
      } finally {
        setUpdating(false);
      }
    },
    [apiUrl, imageFile, inferenceResult, setInferenceResult, setTemperature]
  );

  if (!inferenceResult) return null;

  const modelKeys = Object.keys(inferenceResult.models);
  const teacherData = inferenceResult.models.teacher?.topk?.slice(0, 15);
  if (!teacherData) return null;

  const chartData = teacherData.map((t) => {
    const row: Record<string, string | number> = {
      name: t.class.length > 12 ? t.class.slice(0, 11) + "\u2026" : t.class,
    };
    for (const key of modelKeys) {
      const fullProbs = inferenceResult.models[key]?.full_probs;
      const prob = fullProbs?.[t.class_id] ?? 0;
      row[LABELS[key] || key] = +(prob * 100).toFixed(2);
    }
    return row;
  });

  return (
    <section className="section-container">
      <h2 className="text-2xl font-bold text-center mb-2">
        Probability Distributions
      </h2>
      <p className="text-sm text-slate-500 text-center mb-6">
        How probability structure varies across models
      </p>

      <motion.div
        className="glass-card p-5 max-w-lg mx-auto mb-8"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
      >
        <div className="flex items-center gap-3 mb-3">
          <Thermometer className="w-4 h-4 text-orange-400" />
          <span className="text-sm font-medium">Temperature Scaling</span>
          <span className="ml-auto font-mono text-lg text-orange-400">
            T = {temperature.toFixed(1)}
          </span>
        </div>
        <input
          type="range" min="0.5" max="20" step="0.5"
          value={temperature}
          onChange={(e) => handleTempChange(parseFloat(e.target.value))}
          className="w-full accent-orange-400 h-2 bg-white/10 rounded-lg cursor-pointer"
        />
        <div className="flex justify-between text-xs text-slate-600 mt-1">
          <span>Sharp (T=0.5)</span>
          <span>Soft (T=20)</span>
        </div>
        <p className="text-xs text-slate-500 mt-2">
          p = softmax(z / T) &mdash; Higher T reveals more dark knowledge
        </p>
        {updating && (
          <p className="text-xs text-indigo-400 mt-1 animate-pulse">Recomputing...</p>
        )}
      </motion.div>

      <motion.div
        className="glass-card p-6"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <div className="h-[400px]">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} layout="vertical" margin={{ left: 10, right: 30 }}>
              <XAxis
                type="number" domain={[0, "auto"]}
                tick={{ fontSize: 11, fill: "#94a3b8" }}
                tickFormatter={(v) => `${v}%`}
              />
              <YAxis type="category" dataKey="name" width={100}
                tick={{ fontSize: 11, fill: "#94a3b8" }} />
              <Tooltip
                contentStyle={{ background: "#1e1b4b", border: "1px solid #312e81", borderRadius: 8, fontSize: 12 }}
                formatter={(v) => [`${v}%`]}
              />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              {modelKeys.map((key) => (
                <Bar
                  key={key}
                  dataKey={LABELS[key] || key}
                  fill={COLORS[key] || "#94a3b8"}
                  radius={[0, 3, 3, 0]}
                  maxBarSize={14}
                />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </div>
      </motion.div>
    </section>
  );
}

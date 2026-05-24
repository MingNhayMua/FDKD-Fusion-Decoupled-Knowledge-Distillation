"use client";

import { useState, useCallback } from "react";
import { motion } from "framer-motion";
import { Thermometer } from "lucide-react";
import { useStore } from "@/hooks/useStore";
import { recomputeDistribution } from "@/services/api";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, Legend,
} from "recharts";

const COLORS = { teacher: "#6366f1", assistant: "#8b5cf6", student: "#10b981" };

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
          teacher: { ...inferenceResult!.teacher, ...result.teacher },
          assistant: { ...inferenceResult!.assistant, ...result.assistant },
          student: { ...inferenceResult!.student, ...result.student },
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

  // Build combined distribution data (top 15 by teacher)
  const teacherTopK = inferenceResult.teacher.topk.slice(0, 15);
  const chartData = teacherTopK.map((t) => {
    const aMatch = inferenceResult.assistant.topk.find((a) => a.class_id === t.class_id);
    const sMatch = inferenceResult.student.topk.find((s) => s.class_id === t.class_id);
    return {
      name: t.class.length > 12 ? t.class.slice(0, 11) + "…" : t.class,
      Teacher: +(t.prob * 100).toFixed(2),
      Assistant: aMatch ? +(aMatch.prob * 100).toFixed(2) : 0,
      Student: sMatch ? +(sMatch.prob * 100).toFixed(2) : 0,
    };
  });

  return (
    <section className="section-container">
      <h2 className="text-2xl font-bold text-center mb-2">
        Probability Distributions
      </h2>
      <p className="text-sm text-slate-500 text-center mb-6">
        How the teacher&apos;s probability structure transfers progressively
      </p>

      {/* Temperature Slider */}
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
          type="range"
          min="0.5"
          max="20"
          step="0.5"
          value={temperature}
          onChange={(e) => handleTempChange(parseFloat(e.target.value))}
          className="w-full accent-orange-400 h-2 bg-white/10 rounded-lg cursor-pointer"
        />
        <div className="flex justify-between text-xs text-slate-600 mt-1">
          <span>Sharp (T=0.5)</span>
          <span>Soft (T=20)</span>
        </div>
        <p className="text-xs text-slate-500 mt-2">
          p = softmax(z / T) — Higher T reveals more dark knowledge
        </p>
        {updating && (
          <p className="text-xs text-indigo-400 mt-1 animate-pulse">Recomputing...</p>
        )}
      </motion.div>

      {/* Combined Distribution Chart */}
      <motion.div
        className="glass-card p-6"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <div className="h-[400px]">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} layout="vertical" margin={{ left: 10, right: 30 }}>
              <XAxis
                type="number"
                domain={[0, "auto"]}
                tick={{ fontSize: 11, fill: "#94a3b8" }}
                tickFormatter={(v) => `${v}%`}
              />
              <YAxis
                type="category"
                dataKey="name"
                width={100}
                tick={{ fontSize: 11, fill: "#94a3b8" }}
              />
              <Tooltip
                contentStyle={{
                  background: "#1e1b4b",
                  border: "1px solid #312e81",
                  borderRadius: 8,
                  fontSize: 12,
                }}
                formatter={(v) => [`${v}%`]}
              />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              <Bar dataKey="Teacher" fill={COLORS.teacher} radius={[0, 3, 3, 0]} maxBarSize={14} />
              <Bar dataKey="Assistant" fill={COLORS.assistant} radius={[0, 3, 3, 0]} maxBarSize={14} />
              <Bar dataKey="Student" fill={COLORS.student} radius={[0, 3, 3, 0]} maxBarSize={14} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </motion.div>
    </section>
  );
}

"use client";

import { motion } from "framer-motion";
import { useStore } from "@/hooks/useStore";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
} from "recharts";

const MODEL_CONFIG = {
  teacher: { label: "Teacher", sub: "Swin-B · 86.95M", color: "#6366f1", glow: "glow-teacher" },
  assistant: { label: "Assistant", sub: "ResNet-152 · 58.55M", color: "#8b5cf6", glow: "glow-assistant" },
  student: { label: "Student", sub: "ResNet-18 · 11.28M", color: "#10b981", glow: "glow-student" },
};

export default function ModelComparison() {
  const { inferenceResult } = useStore();
  if (!inferenceResult) return null;

  const models = ["teacher", "assistant", "student"] as const;

  return (
    <section className="section-container">
      <h2 className="text-2xl font-bold text-center mb-2">Model Predictions</h2>
      <p className="text-sm text-slate-500 text-center mb-8">
        Progressive knowledge transfer: Teacher → Assistant → Student
      </p>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {models.map((key, i) => {
          const data = inferenceResult[key];
          const cfg = MODEL_CONFIG[key];
          if (!data || data.error) return null;

          const chartData = data.topk.slice(0, 5).map((p) => ({
            name: p.class.length > 15 ? p.class.slice(0, 14) + "…" : p.class,
            prob: +(p.prob * 100).toFixed(1),
          }));

          return (
            <motion.div
              key={key}
              className="glass-card glass-card-hover p-6"
              style={{ borderColor: `${cfg.color}30` }}
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.15 }}
            >
              {/* Header */}
              <div className="flex items-center gap-3 mb-4">
                <div className="w-3 h-3 rounded-full" style={{ background: cfg.color }} />
                <div>
                  <div className="font-semibold" style={{ color: cfg.color }}>{cfg.label}</div>
                  <div className="text-xs text-slate-500">{cfg.sub}</div>
                </div>
              </div>

              {/* Prediction */}
              <div className="mb-4">
                <div className="text-xs text-slate-500 mb-1">Top Prediction</div>
                <div className="text-lg font-bold">{data.predicted_class}</div>
                <div className="flex items-center gap-2 mt-1">
                  <div className="flex-1 h-2 bg-white/5 rounded-full overflow-hidden">
                    <motion.div
                      className="h-full rounded-full"
                      style={{ background: cfg.color }}
                      initial={{ width: 0 }}
                      animate={{ width: `${data.confidence * 100}%` }}
                      transition={{ duration: 1, delay: i * 0.2 }}
                    />
                  </div>
                  <span className="text-sm font-mono" style={{ color: cfg.color }}>
                    {(data.confidence * 100).toFixed(1)}%
                  </span>
                </div>
              </div>

              {/* Top-5 Chart */}
              <div className="h-[160px] mt-2">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartData} layout="vertical" margin={{ left: 0, right: 10 }}>
                    <XAxis type="number" domain={[0, 100]} hide />
                    <YAxis
                      type="category"
                      dataKey="name"
                      width={90}
                      tick={{ fontSize: 10, fill: "#94a3b8" }}
                    />
                    <Tooltip
                      contentStyle={{ background: "#1e1b4b", border: "1px solid #312e81", borderRadius: 8, fontSize: 12 }}
                      formatter={(v) => [`${v}%`, "Confidence"]}
                    />
                    <Bar dataKey="prob" radius={[0, 4, 4, 0]} maxBarSize={16}>
                      {chartData.map((_, idx) => (
                        <Cell key={idx} fill={idx === 0 ? cfg.color : `${cfg.color}60`} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>

              {/* Stats */}
              <div className="grid grid-cols-2 gap-2 mt-4 text-xs">
                <div className="bg-white/5 rounded-lg p-2">
                  <div className="text-slate-500">Entropy</div>
                  <div className="font-mono font-semibold">{data.entropy.toFixed(3)}</div>
                </div>
                <div className="bg-white/5 rounded-lg p-2">
                  <div className="text-slate-500">Confidence</div>
                  <div className="font-mono font-semibold">{(data.confidence * 100).toFixed(1)}%</div>
                </div>
              </div>
            </motion.div>
          );
        })}
      </div>
    </section>
  );
}

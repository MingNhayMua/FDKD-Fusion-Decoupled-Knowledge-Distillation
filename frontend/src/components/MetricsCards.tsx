"use client";

import { motion } from "framer-motion";
import { useStore } from "@/hooks/useStore";
import { TrendingDown, GitCompare, Activity } from "lucide-react";

const COLORS: Record<string, string> = {
  teacher: "#6366f1", dkd: "#8b5cf6", takd: "#f59e0b", baseline: "#10b981",
};
const LABELS: Record<string, string> = {
  teacher: "Teacher", dkd: "DKD", takd: "TAKD", baseline: "Baseline",
};

interface MetricCardProps {
  icon: React.ReactNode;
  title: string;
  items: { label: string; value: number; color: string; format?: string }[];
  insight: string;
  delay: number;
}

function MetricCard({ icon, title, items, insight, delay }: MetricCardProps) {
  return (
    <motion.div
      className="glass-card glass-card-hover p-6"
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay }}
    >
      <div className="flex items-center gap-2 mb-4">
        {icon}
        <h3 className="text-sm font-semibold">{title}</h3>
      </div>

      <div className="space-y-3">
        {items.map((item) => (
          <div key={item.label}>
            <div className="flex justify-between text-xs mb-1">
              <span className="text-slate-400">{item.label}</span>
              <span className="font-mono font-bold" style={{ color: item.color }}>
                {item.format === "pct"
                  ? `${(item.value * 100).toFixed(1)}%`
                  : item.value.toFixed(4)}
              </span>
            </div>
            <div className="h-1.5 bg-white/5 rounded-full overflow-hidden">
              <motion.div
                className="h-full rounded-full"
                style={{ background: item.color }}
                initial={{ width: 0 }}
                animate={{
                  width: `${item.format === "pct" ? item.value * 100 : Math.min(item.value * 100, 100)}%`,
                }}
                transition={{ duration: 1, delay: delay + 0.2 }}
              />
            </div>
          </div>
        ))}
      </div>

      <p className="text-[11px] text-slate-500 mt-4 leading-relaxed">{insight}</p>
    </motion.div>
  );
}

function getPairs(modelKeys: string[]) {
  const pairs: [string, string][] = [];
  for (let i = 0; i < modelKeys.length; i++) {
    for (let j = i + 1; j < modelKeys.length; j++) {
      pairs.push([modelKeys[i], modelKeys[j]]);
    }
  }
  return pairs;
}

export default function MetricsCards() {
  const { inferenceResult } = useStore();
  if (!inferenceResult?.metrics) return null;

  const modelKeys = Object.keys(inferenceResult.models);
  const pairs = getPairs(modelKeys);
  const m = inferenceResult.metrics;

  const klItems = pairs.map(([a, b]) => ({
    label: `${LABELS[a] || a} \u2192 ${LABELS[b] || b}`,
    value: (m[`kl_${a}_${b}`] ?? m[`kl_${b}_${a}`]) as number,
    color: COLORS[b] || "#94a3b8",
  }));

  const cosItems = pairs.map(([a, b]) => ({
    label: `${LABELS[a] || a} \u2194 ${LABELS[b] || b}`,
    value: (m[`cosine_${a}_${b}`] ?? m[`cosine_${b}_${a}`]) as number,
    color: COLORS[b] || "#94a3b8",
    format: "pct",
  }));

  const entropyItems = modelKeys.map((key) => ({
    label: LABELS[key] || key,
    value: m[`entropy_${key}`] as number,
    color: COLORS[key] || "#94a3b8",
  }));

  return (
    <section className="section-container">
      <h2 className="text-2xl font-bold text-center mb-2">Research Metrics</h2>
      <p className="text-sm text-slate-500 text-center mb-8">
        Quantitative analysis of knowledge transfer quality
      </p>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <MetricCard
          icon={<TrendingDown className="w-5 h-5 text-rose-400" />}
          title="KL Divergence"
          items={klItems}
          insight="Lower KL divergence = better distribution matching across progressive stages."
          delay={0}
        />
        <MetricCard
          icon={<GitCompare className="w-5 h-5 text-blue-400" />}
          title="Cosine Similarity"
          items={cosItems}
          insight="Higher cosine similarity means probability distributions are better aligned."
          delay={0.15}
        />
        <MetricCard
          icon={<Activity className="w-5 h-5 text-amber-400" />}
          title="Distribution Entropy"
          items={entropyItems}
          insight="Lower entropy = sharper predictions. KD should preserve teacher's entropy level."
          delay={0.3}
        />
      </div>
    </section>
  );
}

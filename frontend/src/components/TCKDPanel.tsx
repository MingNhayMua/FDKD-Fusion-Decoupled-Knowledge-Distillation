"use client";

import { motion } from "framer-motion";
import type { TCKDData } from "@/types/inference";

interface Props { data: TCKDData; }

const COLORS: Record<string, string> = {
  teacher: "#6366f1", dkd: "#8b5cf6", takd: "#f59e0b", baseline: "#10b981",
};
const LABELS: Record<string, string> = {
  teacher: "Teacher", dkd: "DKD", takd: "TAKD", baseline: "Baseline",
};

function ConfidenceGauge({ label, value, color, delay }: {
  label: string; value: number; color: string; delay: number;
}) {
  const pct = value * 100;
  const circumference = 2 * Math.PI * 45;
  const offset = circumference - (pct / 100) * circumference;

  return (
    <div className="flex flex-col items-center">
      <div className="relative w-24 h-24">
        <svg className="w-full h-full -rotate-90" viewBox="0 0 100 100">
          <circle cx="50" cy="50" r="45" fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth="8" />
          <motion.circle
            cx="50" cy="50" r="45" fill="none" stroke={color} strokeWidth="8"
            strokeLinecap="round"
            strokeDasharray={circumference}
            initial={{ strokeDashoffset: circumference }}
            animate={{ strokeDashoffset: offset }}
            transition={{ duration: 1.2, delay, ease: "easeOut" }}
          />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="text-base font-mono font-bold" style={{ color }}>
            {pct.toFixed(1)}%
          </span>
        </div>
      </div>
      <span className="text-xs font-medium mt-1" style={{ color }}>{label}</span>
    </div>
  );
}

export default function TCKDPanel({ data }: Props) {
  const roleKeys = Object.keys(data.confidences);

  // Collect alignment pairs
  const alignPairs: { key: string; label: string; value: number; color: string }[] = [];
  for (let i = 0; i < roleKeys.length; i++) {
    for (let j = i + 1; j < roleKeys.length; j++) {
      const a = roleKeys[i], b = roleKeys[j];
      const valKey = `${a}_${b}_alignment`;
      if (typeof data[valKey] === "number") {
        alignPairs.push({
          key: valKey,
          label: `${LABELS[a] || a} \u2192 ${LABELS[b] || b}`,
          value: data[valKey] as number,
          color: COLORS[b] || "#94a3b8",
        });
      }
    }
  }

  return (
    <div className="glass-card p-8">
      <h3 className="text-lg font-bold mb-1">Target Class Knowledge Distillation</h3>
      <p className="text-sm text-slate-500 mb-6">
        How confident is each model about the target class &quot;<span className="text-indigo-400">{data.target_class}</span>&quot;?
      </p>

      {/* Confidence Gauges */}
      <div className="flex flex-wrap justify-center gap-6 mb-8">
        {roleKeys.map((key, i) => (
          <ConfidenceGauge
            key={key}
            label={LABELS[key] || key}
            value={data.confidences[key]}
            color={COLORS[key] || "#94a3b8"}
            delay={i * 0.15}
          />
        ))}
      </div>

      {/* Alignment Bars */}
      <div className="max-w-md mx-auto">
        <h4 className="text-sm font-semibold text-slate-300 mb-3">Confidence Alignment</h4>
        {alignPairs.map((pair) => (
          <div className="mb-3" key={pair.key}>
            <div className="flex justify-between text-xs mb-1">
              <span className="text-slate-400">{pair.label}</span>
              <span className="font-mono" style={{ color: pair.color }}>
                {(pair.value * 100).toFixed(1)}%
              </span>
            </div>
            <div className="h-2 bg-white/5 rounded-full overflow-hidden">
              <motion.div
                className="h-full rounded-full"
                style={{ background: pair.color }}
                initial={{ width: 0 }}
                animate={{ width: `${pair.value * 100}%` }}
                transition={{ duration: 1 }}
              />
            </div>
          </div>
        ))}
      </div>

      <div className="mt-6 p-4 bg-indigo-500/5 rounded-xl text-xs text-slate-400 leading-relaxed">
        <strong className="text-indigo-400">TCKD Insight:</strong> Progressive knowledge transfer
        through intermediate models helps preserve the teacher&apos;s confidence better than
        direct transfer, reducing the capacity gap at each stage.
      </div>
    </div>
  );
}

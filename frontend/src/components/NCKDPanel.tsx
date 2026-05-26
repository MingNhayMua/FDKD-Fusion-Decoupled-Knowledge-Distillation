"use client";

import { motion } from "framer-motion";
import type { NCKDData } from "@/types/inference";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
} from "recharts";

interface Props { data: NCKDData; }

const COLORS: Record<string, string> = {
  teacher: "#6366f1", dkd: "#8b5cf6", takd: "#f59e0b", baseline: "#10b981",
};
const LABELS: Record<string, string> = {
  teacher: "Teacher", dkd: "DKD", takd: "TAKD", baseline: "Baseline",
};

function RankingChart({ title, ranking, color }: {
  title: string; ranking: { class: string; prob: number }[]; color: string;
}) {
  const chartData = ranking.slice(0, 8).map((r) => ({
    name: r.class.length > 14 ? r.class.slice(0, 13) + "\u2026" : r.class,
    prob: +(r.prob * 100).toFixed(2),
  }));

  return (
    <div>
      <h4 className="text-sm font-semibold mb-2" style={{ color }}>{title}</h4>
      <div className="h-[200px]">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} layout="vertical" margin={{ left: 0, right: 10 }}>
            <XAxis type="number" hide />
            <YAxis type="category" dataKey="name" width={95} tick={{ fontSize: 10, fill: "#94a3b8" }} />
            <Tooltip
              contentStyle={{ background: "#1e1b4b", border: "1px solid #312e81", borderRadius: 8, fontSize: 11 }}
              formatter={(v) => [`${v}%`, "Prob"]}
            />
            <Bar dataKey="prob" radius={[0, 4, 4, 0]} maxBarSize={14}>
              {chartData.map((_, i) => (
                <Cell key={i} fill={i === 0 ? color : `${color}70`} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function MetricBadge({ label, value, unit, color }: {
  label: string; value: number; unit: string; color: string;
}) {
  return (
    <div className="bg-white/5 rounded-xl p-3 text-center">
      <div className="text-xs text-slate-500 mb-1">{label}</div>
      <div className="text-lg font-mono font-bold" style={{ color }}>
        {value.toFixed(4)}
      </div>
      <div className="text-[10px] text-slate-600">{unit}</div>
    </div>
  );
}

export default function NCKDPanel({ data }: Props) {
  // Find ranking keys
  const rankingKeys = Object.keys(data).filter(k => k.endsWith("_ranking"));
  const roles = rankingKeys.map(k => k.replace("_ranking", ""));

  // Collect KL and correlation pairs
  const klPairs: { label: string; value: number; roleKey: string }[] = [];
  const corrPairs: { label: string; value: number; roleKey: string }[] = [];
  for (let i = 0; i < roles.length; i++) {
    for (let j = i + 1; j < roles.length; j++) {
      const a = roles[i], b = roles[j];
      const klKey = `kl_${a}_${b}`;
      const corrKey = `rank_correlation_${a}_${b}`;
      if (typeof data[klKey] === "number") {
        klPairs.push({ label: `${LABELS[a] || a}\u2192${LABELS[b] || b}`, value: data[klKey] as number, roleKey: b });
      }
      if (typeof data[corrKey] === "number") {
        corrPairs.push({ label: `${LABELS[a] || a}\u2194${LABELS[b] || b}`, value: data[corrKey] as number, roleKey: b });
      }
    }
  }

  const gridCols = roles.length <= 3 ? "md:grid-cols-3" : roles.length === 4 ? "md:grid-cols-4" : "md:grid-cols-2";

  return (
    <div className="glass-card p-8">
      <h3 className="text-lg font-bold mb-1">Non-Target Class Knowledge Distillation</h3>
      <p className="text-sm text-slate-500 mb-6">
        Does each model preserve the teacher&apos;s ranking among non-target classes?
      </p>

      <div className={`grid grid-cols-1 ${gridCols} gap-4 mb-8`}>
        {rankingKeys.map((k) => {
          const role = k.replace("_ranking", "");
          return (
            <RankingChart
              key={k}
              title={`${LABELS[role] || role} Ranking`}
              ranking={data[k] as { class: string; prob: number }[]}
              color={COLORS[role] || "#94a3b8"}
            />
          );
        })}
      </div>

      <h4 className="text-sm font-semibold text-slate-300 mb-3">Ranking Preservation Metrics</h4>
      <div className="grid grid-cols-2 md:grid-cols-3 gap-3 mb-4">
        {klPairs.map((p) => (
          <MetricBadge key={`kl_${p.label}`} label={`KL (${p.label})`} value={p.value} unit="nats" color={COLORS[p.roleKey] || "#94a3b8"} />
        ))}
        {corrPairs.map((p) => (
          <MetricBadge key={`corr_${p.label}`} label={`\u03C4 (${p.label})`} value={p.value} unit="correlation" color={COLORS[p.roleKey] || "#94a3b8"} />
        ))}
      </div>

      <div className="mt-6 p-4 bg-violet-500/5 rounded-xl text-xs text-slate-400 leading-relaxed">
        <strong className="text-violet-400">NCKD Insight:</strong> The non-target class distribution
        encodes semantic similarity. FDKD preserves this structure better through staged transfer,
        as shown by lower KL divergence and higher rank correlation.
      </div>
    </div>
  );
}

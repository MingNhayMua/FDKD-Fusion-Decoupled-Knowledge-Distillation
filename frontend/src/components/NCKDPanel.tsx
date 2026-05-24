"use client";

import { motion } from "framer-motion";
import type { NCKDData } from "@/types/inference";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
} from "recharts";

interface Props { data: NCKDData; }

const COLORS = { teacher: "#6366f1", assistant: "#8b5cf6", student: "#10b981" };

function RankingChart({ title, ranking, color }: {
  title: string; ranking: { class: string; prob: number }[]; color: string;
}) {
  const chartData = ranking.slice(0, 8).map((r) => ({
    name: r.class.length > 14 ? r.class.slice(0, 13) + "…" : r.class,
    prob: +(r.prob * 100).toFixed(2),
  }));

  return (
    <div>
      <h4 className="text-sm font-semibold mb-2" style={{ color }}>{title}</h4>
      <div className="h-[240px]">
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
    <div className="bg-white/5 rounded-xl p-4 text-center">
      <div className="text-xs text-slate-500 mb-1">{label}</div>
      <div className="text-xl font-mono font-bold" style={{ color }}>
        {value.toFixed(4)}
      </div>
      <div className="text-[10px] text-slate-600">{unit}</div>
    </div>
  );
}

export default function NCKDPanel({ data }: Props) {
  return (
    <div className="glass-card p-8">
      <h3 className="text-lg font-bold mb-1">Non-Target Class Knowledge Distillation</h3>
      <p className="text-sm text-slate-500 mb-6">
        Does the student preserve the teacher&apos;s ranking among non-target classes?
      </p>

      {/* Side-by-side Rankings */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        <RankingChart title="Teacher Ranking" ranking={data.teacher_ranking} color={COLORS.teacher} />
        <RankingChart title="Assistant Ranking" ranking={data.assistant_ranking} color={COLORS.assistant} />
        <RankingChart title="Student Ranking" ranking={data.student_ranking} color={COLORS.student} />
      </div>

      {/* Metrics */}
      <h4 className="text-sm font-semibold text-slate-300 mb-3">Ranking Preservation Metrics</h4>
      <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
        <MetricBadge label="KL Div (T→A)" value={data.kl_ta} unit="nats" color="#8b5cf6" />
        <MetricBadge label="KL Div (A→S)" value={data.kl_as} unit="nats" color="#10b981" />
        <MetricBadge label="KL Div (T→S)" value={data.kl_ts} unit="nats" color="#06b6d4" />
        <MetricBadge label="Kendall τ (T↔A)" value={data.rank_correlation_ta} unit="correlation" color="#8b5cf6" />
        <MetricBadge label="Kendall τ (A↔S)" value={data.rank_correlation_as} unit="correlation" color="#10b981" />
        <MetricBadge label="Kendall τ (T↔S)" value={data.rank_correlation_ts} unit="correlation" color="#06b6d4" />
      </div>

      <div className="mt-6 p-4 bg-violet-500/5 rounded-xl text-xs text-slate-400 leading-relaxed">
        <strong className="text-violet-400">NCKD Insight:</strong> The non-target class distribution
        encodes semantic similarity (e.g., &quot;tiger cat&quot; and &quot;Persian cat&quot; having nearby probabilities).
        FDKD preserves this structure better through staged transfer, as shown by lower KL divergence
        and higher rank correlation.
      </div>
    </div>
  );
}

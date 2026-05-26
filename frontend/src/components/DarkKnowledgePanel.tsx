"use client";

import { motion } from "framer-motion";
import type { DarkKnowledgeData } from "@/types/inference";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend,
} from "recharts";

interface Props { data: DarkKnowledgeData; }

const COLORS: Record<string, string> = {
  teacher: "#6366f1", dkd: "#8b5cf6", takd: "#f59e0b", baseline: "#10b981",
};
const LABELS: Record<string, string> = {
  teacher: "Teacher", dkd: "DKD", takd: "TAKD", baseline: "Baseline",
};

export default function DarkKnowledgePanel({ data }: Props) {
  // Find top_non_target keys
  const nonTargetKeys = Object.keys(data).filter(k => k.endsWith("_top_non_target"));
  const roles = nonTargetKeys.map(k => k.replace("_top_non_target", ""));

  // Use first model's (teacher) top non-target as reference
  const refKey = nonTargetKeys.find(k => k.startsWith("teacher_")) || nonTargetKeys[0];
  const refData = (data[refKey] as { class: string; prob: number; class_id: number }[]) || [];

  const chartData = refData.slice(0, 10).map((ref) => {
    const row: Record<string, string | number> = {
      name: ref.class.length > 14 ? ref.class.slice(0, 13) + "\u2026" : ref.class,
    };
    for (let i = 0; i < nonTargetKeys.length; i++) {
      const key = nonTargetKeys[i];
      const role = roles[i];
      const modelData = data[key] as { class: string; prob: number; class_id: number }[];
      const match = modelData?.find((m) => m.class_id === ref.class_id);
      row[LABELS[role] || role] = match ? +(match.prob * 100).toFixed(2) : 0;
    }
    return row;
  });

  return (
    <div className="glass-card p-8">
      <h3 className="text-lg font-bold mb-1">Dark Knowledge Transfer</h3>
      <p className="text-sm text-slate-500 mb-6">
        Semantic structure encoded in non-target probabilities
      </p>

      <motion.div
        className="bg-gradient-to-r from-indigo-500/10 via-violet-500/10 to-emerald-500/10
                   rounded-xl p-5 mb-8 border border-white/5"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
      >
        <p className="text-sm text-slate-300 leading-relaxed">
          {data.explanation as string}
        </p>
      </motion.div>

      <div className="h-[350px]">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} layout="vertical" margin={{ left: 10, right: 30 }}>
            <XAxis
              type="number" tick={{ fontSize: 11, fill: "#94a3b8" }}
              tickFormatter={(v) => `${v}%`}
            />
            <YAxis type="category" dataKey="name" width={110}
              tick={{ fontSize: 11, fill: "#94a3b8" }} />
            <Tooltip
              contentStyle={{ background: "#1e1b4b", border: "1px solid #312e81", borderRadius: 8, fontSize: 12 }}
              formatter={(v) => [`${v}%`]}
            />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            {nonTargetKeys.map((key, i) => (
              <Bar
                key={key}
                dataKey={LABELS[roles[i]] || roles[i]}
                fill={COLORS[roles[i]] || "#94a3b8"}
                radius={[0, 3, 3, 0]}
                maxBarSize={14}
              />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-6 p-4 bg-emerald-500/5 rounded-xl text-xs text-slate-400 leading-relaxed">
        <strong className="text-emerald-400">Key Insight:</strong> When the teacher assigns similar
        probabilities to &quot;tiger cat&quot; and &quot;tabby cat&quot; for a &quot;Persian cat&quot; image, it reveals that
        these classes are semantically related. The student learns this relational structure &mdash;
        this is the &quot;dark knowledge&quot; that makes KD more effective than hard labels alone.
      </div>
    </div>
  );
}

"use client";

import { motion } from "framer-motion";
import type { DarkKnowledgeData } from "@/types/inference";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, Legend,
} from "recharts";

interface Props { data: DarkKnowledgeData; }

export default function DarkKnowledgePanel({ data }: Props) {
  // Build comparison data
  const chartData = data.teacher_top_non_target.slice(0, 10).map((t) => {
    const sMatch = data.student_top_non_target.find((s) => s.class_id === t.class_id);
    return {
      name: t.class.length > 14 ? t.class.slice(0, 13) + "…" : t.class,
      Teacher: +(t.prob * 100).toFixed(2),
      Student: sMatch ? +(sMatch.prob * 100).toFixed(2) : 0,
    };
  });

  return (
    <div className="glass-card p-8">
      <h3 className="text-lg font-bold mb-1">Dark Knowledge Transfer</h3>
      <p className="text-sm text-slate-500 mb-6">
        Semantic structure encoded in non-target probabilities
      </p>

      {/* Explanation Card */}
      <motion.div
        className="bg-gradient-to-r from-indigo-500/10 via-violet-500/10 to-emerald-500/10
                   rounded-xl p-5 mb-8 border border-white/5"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
      >
        <p className="text-sm text-slate-300 leading-relaxed">
          {data.explanation}
        </p>
      </motion.div>

      {/* Comparison Chart */}
      <div className="h-[350px]">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} layout="vertical" margin={{ left: 10, right: 30 }}>
            <XAxis
              type="number"
              tick={{ fontSize: 11, fill: "#94a3b8" }}
              tickFormatter={(v) => `${v}%`}
            />
            <YAxis type="category" dataKey="name" width={110} tick={{ fontSize: 11, fill: "#94a3b8" }} />
            <Tooltip
              contentStyle={{ background: "#1e1b4b", border: "1px solid #312e81", borderRadius: 8, fontSize: 12 }}
              formatter={(v) => [`${v}%`]}
            />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            <Bar dataKey="Teacher" fill="#6366f1" radius={[0, 3, 3, 0]} maxBarSize={14} />
            <Bar dataKey="Student" fill="#10b981" radius={[0, 3, 3, 0]} maxBarSize={14} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-6 p-4 bg-emerald-500/5 rounded-xl text-xs text-slate-400 leading-relaxed">
        <strong className="text-emerald-400">Key Insight:</strong> When the teacher assigns similar
        probabilities to &quot;tiger cat&quot; and &quot;tabby cat&quot; for a &quot;Persian cat&quot; image, it reveals that
        these classes are semantically related. The student learns this relational structure —
        this is the &quot;dark knowledge&quot; that makes KD more effective than hard labels alone.
      </div>
    </div>
  );
}

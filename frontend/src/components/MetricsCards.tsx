"use client";

import { motion } from "framer-motion";
import { useStore } from "@/hooks/useStore";
import { TrendingDown, GitCompare, Activity } from "lucide-react";

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

export default function MetricsCards() {
  const { inferenceResult } = useStore();
  if (!inferenceResult?.metrics) return null;

  const m = inferenceResult.metrics;

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
          items={[
            { label: "T → A", value: m.kl_teacher_assistant, color: "#8b5cf6" },
            { label: "A → S", value: m.kl_assistant_student, color: "#10b981" },
            { label: "T → S (direct)", value: m.kl_teacher_student, color: "#06b6d4" },
          ]}
          insight="Lower KL divergence indicates better distribution matching. FDKD reduces the gap by staging transfer through the assistant."
          delay={0}
        />

        <MetricCard
          icon={<GitCompare className="w-5 h-5 text-blue-400" />}
          title="Cosine Similarity"
          items={[
            { label: "T ↔ A", value: m.cosine_teacher_assistant, color: "#8b5cf6", format: "pct" },
            { label: "A ↔ S", value: m.cosine_assistant_student, color: "#10b981", format: "pct" },
            { label: "T ↔ S", value: m.cosine_teacher_student, color: "#06b6d4", format: "pct" },
          ]}
          insight="Higher cosine similarity means the student's probability distribution is better aligned with the teacher's."
          delay={0.15}
        />

        <MetricCard
          icon={<Activity className="w-5 h-5 text-amber-400" />}
          title="Distribution Entropy"
          items={[
            { label: "Teacher", value: m.entropy_teacher, color: "#6366f1" },
            { label: "Assistant", value: m.entropy_assistant, color: "#8b5cf6" },
            { label: "Student", value: m.entropy_student, color: "#10b981" },
          ]}
          insight="Lower entropy = sharper, more confident predictions. The student's entropy should be close to the teacher's after successful distillation."
          delay={0.3}
        />
      </div>
    </section>
  );
}

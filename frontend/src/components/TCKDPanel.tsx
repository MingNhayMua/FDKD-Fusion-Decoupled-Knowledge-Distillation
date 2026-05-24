"use client";

import { motion } from "framer-motion";
import type { TCKDData } from "@/types/inference";

interface Props { data: TCKDData; }

function ConfidenceGauge({ label, value, color, delay }: {
  label: string; value: number; color: string; delay: number;
}) {
  const pct = value * 100;
  const circumference = 2 * Math.PI * 45;
  const offset = circumference - (pct / 100) * circumference;

  return (
    <div className="flex flex-col items-center">
      <div className="relative w-28 h-28">
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
          <span className="text-lg font-mono font-bold" style={{ color }}>
            {pct.toFixed(1)}%
          </span>
        </div>
      </div>
      <span className="text-sm font-medium mt-2" style={{ color }}>{label}</span>
    </div>
  );
}

function AlignmentBar({ label, value, color }: {
  label: string; value: number; color: string;
}) {
  return (
    <div className="mb-3">
      <div className="flex justify-between text-xs mb-1">
        <span className="text-slate-400">{label}</span>
        <span className="font-mono" style={{ color }}>{(value * 100).toFixed(1)}%</span>
      </div>
      <div className="h-2 bg-white/5 rounded-full overflow-hidden">
        <motion.div
          className="h-full rounded-full"
          style={{ background: color }}
          initial={{ width: 0 }}
          animate={{ width: `${value * 100}%` }}
          transition={{ duration: 1 }}
        />
      </div>
    </div>
  );
}

export default function TCKDPanel({ data }: Props) {
  return (
    <div className="glass-card p-8">
      <h3 className="text-lg font-bold mb-1">Target Class Knowledge Distillation</h3>
      <p className="text-sm text-slate-500 mb-6">
        How confident is each model about the target class &quot;<span className="text-indigo-400">{data.target_class}</span>&quot;?
      </p>

      {/* Confidence Gauges */}
      <div className="flex justify-center gap-12 mb-8">
        <ConfidenceGauge label="Teacher" value={data.teacher_confidence} color="#6366f1" delay={0} />
        <div className="flex items-center text-slate-600 text-2xl">→</div>
        <ConfidenceGauge label="Assistant" value={data.assistant_confidence} color="#8b5cf6" delay={0.2} />
        <div className="flex items-center text-slate-600 text-2xl">→</div>
        <ConfidenceGauge label="Student" value={data.student_confidence} color="#10b981" delay={0.4} />
      </div>

      {/* Alignment Bars */}
      <div className="max-w-md mx-auto">
        <h4 className="text-sm font-semibold text-slate-300 mb-3">Confidence Alignment</h4>
        <AlignmentBar label="Teacher → Assistant" value={data.ta_alignment} color="#8b5cf6" />
        <AlignmentBar label="Assistant → Student" value={data.as_alignment} color="#10b981" />
        <AlignmentBar label="Teacher → Student (direct)" value={data.ts_alignment} color="#06b6d4" />
      </div>

      <div className="mt-6 p-4 bg-indigo-500/5 rounded-xl text-xs text-slate-400 leading-relaxed">
        <strong className="text-indigo-400">TCKD Insight:</strong> Progressive transfer through the assistant
        preserves the teacher&apos;s confidence more effectively than direct T→S transfer,
        reducing the capacity gap at each stage.
      </div>
    </div>
  );
}

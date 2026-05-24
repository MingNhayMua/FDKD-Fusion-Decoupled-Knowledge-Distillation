"use client";

import { motion } from "framer-motion";
import { useStore } from "@/hooks/useStore";
import TCKDPanel from "./TCKDPanel";
import NCKDPanel from "./NCKDPanel";
import DarkKnowledgePanel from "./DarkKnowledgePanel";

const TABS = [
  { id: "tckd" as const, label: "TCKD", desc: "Target Class" },
  { id: "nckd" as const, label: "NCKD", desc: "Non-Target" },
  { id: "dark" as const, label: "Dark Knowledge", desc: "Semantics" },
];

export default function DKDAnalysis() {
  const { inferenceResult, activeTab, setActiveTab } = useStore();
  if (!inferenceResult?.dkd) return null;

  return (
    <section className="section-container">
      <h2 className="text-2xl font-bold text-center mb-2">
        Decoupled Knowledge Distillation Analysis
      </h2>
      <p className="text-sm text-slate-500 text-center mb-8">
        Decomposing what the student learns: target-class confidence vs. non-target structure
      </p>

      {/* Tab Buttons */}
      <div className="flex justify-center gap-2 mb-8">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`relative px-5 py-2.5 rounded-xl text-sm font-medium transition-all
              ${activeTab === tab.id
                ? "bg-indigo-600/20 text-indigo-300 border border-indigo-500/30"
                : "bg-white/5 text-slate-400 border border-transparent hover:bg-white/10"
              }`}
          >
            <span className="font-bold">{tab.label}</span>
            <span className="block text-[10px] text-slate-500">{tab.desc}</span>
            {activeTab === tab.id && (
              <motion.div
                layoutId="activeTabIndicator"
                className="absolute inset-0 rounded-xl border border-indigo-500/40"
                transition={{ type: "spring", stiffness: 300, damping: 30 }}
              />
            )}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      <motion.div
        key={activeTab}
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
      >
        {activeTab === "tckd" && <TCKDPanel data={inferenceResult.dkd.tckd} />}
        {activeTab === "nckd" && <NCKDPanel data={inferenceResult.dkd.nckd} />}
        {activeTab === "dark" && <DarkKnowledgePanel data={inferenceResult.dkd.dark_knowledge} />}
      </motion.div>
    </section>
  );
}

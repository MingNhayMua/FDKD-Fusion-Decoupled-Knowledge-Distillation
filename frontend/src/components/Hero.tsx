"use client";

import { motion } from "framer-motion";

export default function Hero() {
  const nodes = [
    { label: "Teacher", sub: "Swin-B", color: "#6366f1", x: 0 },
    { label: "Assistant", sub: "ResNet-152", color: "#8b5cf6", x: 1 },
    { label: "Student", sub: "ResNet-18", color: "#10b981", x: 2 },
  ];

  return (
    <section className="relative overflow-hidden py-20 px-6">
      {/* Background gradient */}
      <div className="absolute inset-0 bg-gradient-to-b from-indigo-950/30 via-transparent to-transparent" />
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[800px] h-[400px] bg-indigo-500/10 rounded-full blur-[120px]" />

      <div className="relative max-w-5xl mx-auto text-center">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
        >
          <p className="text-sm font-mono text-indigo-400 tracking-widest mb-4 uppercase">
            Interactive Research Demo
          </p>
          <h1 className="text-4xl md:text-6xl font-bold mb-4 leading-tight">
            <span className="gradient-text">Fusion Decoupled</span>
            <br />
            <span className="text-white">Knowledge Distillation</span>
          </h1>
          <p className="text-lg text-slate-400 max-w-2xl mx-auto mb-12">
            Visualizing progressive knowledge transfer via{" "}
            <span className="text-indigo-400">Teacher</span> →{" "}
            <span className="text-violet-400">Assistant</span> →{" "}
            <span className="text-emerald-400">Student</span> decoupling
          </p>
        </motion.div>

        {/* Pipeline Animation */}
        <motion.div
          className="flex items-center justify-center gap-4 md:gap-8"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.3 }}
        >
          {nodes.map((node, i) => (
            <div key={node.label} className="flex items-center gap-4 md:gap-8">
              <motion.div
                className="glass-card p-4 md:p-6 text-center min-w-[120px] md:min-w-[160px]"
                style={{ borderColor: node.color, borderWidth: 1 }}
                whileHover={{ scale: 1.05, boxShadow: `0 0 30px ${node.color}40` }}
                animate={{
                  boxShadow: [
                    `0 0 10px ${node.color}20`,
                    `0 0 25px ${node.color}40`,
                    `0 0 10px ${node.color}20`,
                  ],
                }}
                transition={{ duration: 2, repeat: Infinity, delay: i * 0.5 }}
              >
                <div className="text-sm font-mono mb-1" style={{ color: node.color }}>
                  {node.label}
                </div>
                <div className="text-xs text-slate-500">{node.sub}</div>
              </motion.div>

              {/* Arrow between nodes */}
              {i < nodes.length - 1 && (
                <div className="relative w-12 md:w-20 h-[2px]" style={{ background: `linear-gradient(90deg, ${nodes[i].color}, ${nodes[i+1].color})` }}>
                  <motion.div
                    className="absolute top-1/2 -translate-y-1/2 w-3 h-3 rounded-full"
                    style={{ background: nodes[i+1].color }}
                    animate={{ x: [0, 48, 80], opacity: [0, 1, 0] }}
                    transition={{ duration: 1.5, repeat: Infinity, delay: i * 0.6 }}
                  />
                  <div
                    className="absolute right-0 top-1/2 -translate-y-1/2 w-0 h-0"
                    style={{
                      borderTop: "5px solid transparent",
                      borderBottom: "5px solid transparent",
                      borderLeft: `8px solid ${nodes[i+1].color}`,
                    }}
                  />
                </div>
              )}
            </div>
          ))}
        </motion.div>

        <motion.p
          className="text-xs text-slate-600 mt-8 font-mono"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.2 }}
        >
          Tran et al. — Fusion Decoupled Knowledge Distillation (PRLetters)
        </motion.p>
      </div>
    </section>
  );
}

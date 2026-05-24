"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Wifi, WifiOff, Loader2 } from "lucide-react";
import { useStore } from "@/hooks/useStore";
import { checkHealth } from "@/services/api";

export default function ConnectionInput() {
  const { apiUrl, setApiUrl, isConnected, setConnected } = useStore();
  const [input, setInput] = useState(apiUrl);
  const [checking, setChecking] = useState(false);

  const handleConnect = async () => {
    if (!input.trim()) return;
    setChecking(true);
    const url = input.trim().replace(/\/$/, "");
    setApiUrl(url);

    const ok = await checkHealth(url);
    setConnected(ok);
    setChecking(false);
  };

  return (
    <section className="section-container py-8">
      <motion.div
        className="glass-card p-6 max-w-2xl mx-auto"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <div className="flex items-center gap-3 mb-3">
          {isConnected ? (
            <Wifi className="w-5 h-5 text-emerald-400" />
          ) : (
            <WifiOff className="w-5 h-5 text-slate-500" />
          )}
          <h2 className="text-sm font-semibold text-slate-300">
            Backend Connection
          </h2>
          {isConnected && (
            <span className="text-xs bg-emerald-500/20 text-emerald-400 px-2 py-0.5 rounded-full">
              Connected
            </span>
          )}
        </div>

        <div className="flex gap-3">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleConnect()}
            placeholder="Paste your ngrok URL (e.g. https://xxxx.ngrok-free.app)"
            className="flex-1 bg-white/5 border border-white/10 rounded-lg px-4 py-2.5 text-sm
                       text-white placeholder-slate-500 outline-none focus:border-indigo-500/50
                       transition-colors font-mono"
          />
          <button
            onClick={handleConnect}
            disabled={checking || !input.trim()}
            className="px-5 py-2.5 bg-indigo-600 hover:bg-indigo-500 disabled:bg-slate-700
                       rounded-lg text-sm font-medium transition-colors flex items-center gap-2"
          >
            {checking ? <Loader2 className="w-4 h-4 animate-spin" /> : null}
            Connect
          </button>
        </div>

        {!isConnected && apiUrl && !checking && (
          <p className="text-xs text-red-400 mt-2">
            Failed to connect. Make sure your Colab server is running.
          </p>
        )}
      </motion.div>
    </section>
  );
}

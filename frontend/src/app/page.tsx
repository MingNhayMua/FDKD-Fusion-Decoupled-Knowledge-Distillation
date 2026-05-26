"use client";

import { useStore } from "@/hooks/useStore";
import Hero from "@/components/Hero";
import ConnectionInput from "@/components/ConnectionInput";
import ImageUpload from "@/components/ImageUpload";
import ModelComparison from "@/components/ModelComparison";
import DistributionSection from "@/components/DistributionSection";
import DKDAnalysis from "@/components/DKDAnalysis";
import MetricsCards from "@/components/MetricsCards";
import GradCAMSection from "@/components/GradCAMSection";

export default function Home() {
  const { inferenceResult } = useStore();

  return (
    <main className="min-h-screen">
      <Hero />
      <ConnectionInput />
      <ImageUpload />

      {inferenceResult && (
        <>
          <ModelComparison />
          <GradCAMSection />
          <DistributionSection />
          <DKDAnalysis />
          <MetricsCards />
        </>
      )}
    </main>
  );
}

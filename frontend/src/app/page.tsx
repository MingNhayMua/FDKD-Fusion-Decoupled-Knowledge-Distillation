"use client";

import { useStore } from "@/hooks/useStore";
import Hero from "@/components/Hero";
import ConnectionInput from "@/components/ConnectionInput";
import ImageUpload from "@/components/ImageUpload";
import ModelComparison from "@/components/ModelComparison";
import DistributionSection from "@/components/DistributionSection";
import DKDAnalysis from "@/components/DKDAnalysis";
import GradCAMViewer from "@/components/GradCAMViewer";
import MetricsCards from "@/components/MetricsCards";

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
          <DistributionSection />
          <DKDAnalysis />
          <GradCAMViewer />
          <MetricsCards />
        </>
      )}
    </main>
  );
}

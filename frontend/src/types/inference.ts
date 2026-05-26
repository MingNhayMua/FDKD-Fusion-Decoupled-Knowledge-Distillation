export interface TopKPrediction {
  class_id: number;
  class: string;
  prob: number;
}

export interface ModelPrediction {
  name: string;
  params: string;
  gflops: string;
  topk: TopKPrediction[];
  confidence: number;
  predicted_class: string;
  predicted_class_id: number;
  entropy: number;
  full_probs: number[];
  error?: string;
}

export interface TCKDData {
  target_class: string;
  target_class_id: number;
  confidences: Record<string, number>;
  [key: string]: unknown; // alignment keys like teacher_dkd_alignment
}

export interface NCKDRanking {
  class: string;
  class_id: number;
  prob: number;
}

export interface NCKDData {
  [key: string]: unknown; // {role}_ranking, kl_{a}_{b}, rank_correlation_{a}_{b}
}

export interface DarkKnowledgeData {
  [key: string]: unknown; // {role}_top_non_target, explanation
}

export interface DKDData {
  tckd: TCKDData;
  nckd: NCKDData;
  dark_knowledge: DarkKnowledgeData;
}

export interface MetricsData {
  [key: string]: number; // entropy_{role}, kl_{a}_{b}, cosine_{a}_{b}
}

export interface InferenceResponse {
  image_id: string;
  image_base64: string;
  temperature: number;
  models: Record<string, ModelPrediction>;
  dkd: DKDData;
  metrics: MetricsData;
}

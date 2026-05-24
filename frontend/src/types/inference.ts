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
  teacher_confidence: number;
  assistant_confidence: number;
  student_confidence: number;
  ta_alignment: number;
  as_alignment: number;
  ts_alignment: number;
}

export interface NCKDRanking {
  class: string;
  class_id: number;
  prob: number;
}

export interface NCKDData {
  teacher_ranking: NCKDRanking[];
  assistant_ranking: NCKDRanking[];
  student_ranking: NCKDRanking[];
  kl_ta: number;
  kl_as: number;
  kl_ts: number;
  rank_correlation_ta: number;
  rank_correlation_as: number;
  rank_correlation_ts: number;
}

export interface DarkKnowledgeData {
  teacher_top_non_target: NCKDRanking[];
  student_top_non_target: NCKDRanking[];
  explanation: string;
}

export interface DKDData {
  tckd: TCKDData;
  nckd: NCKDData;
  dark_knowledge: DarkKnowledgeData;
}

export interface MetricsData {
  kl_teacher_student: number;
  kl_teacher_assistant: number;
  kl_assistant_student: number;
  cosine_teacher_student: number;
  cosine_teacher_assistant: number;
  cosine_assistant_student: number;
  entropy_teacher: number;
  entropy_assistant: number;
  entropy_student: number;
}

export interface GradCAMData {
  teacher: string | null;
  assistant: string | null;
  student: string | null;
}

export interface InferenceResponse {
  image_id: string;
  image_base64: string;
  temperature: number;
  teacher: ModelPrediction;
  assistant: ModelPrediction;
  student: ModelPrediction;
  dkd: DKDData;
  metrics: MetricsData;
  gradcam?: GradCAMData;
}

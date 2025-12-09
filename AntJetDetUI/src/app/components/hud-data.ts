// ============================================================
// MOCK DATA — Jet Aircraft Detection & Intelligence System
// ============================================================

export interface Detection {
  label: string;
  confidence: number;
  bbox: [number, number, number, number]; // x%, y%, w%, h% (normalized 0-1)
  targetId: string;
  azimuth?: number;
  elevation?: number;
  distance_km?: number;
}

export interface PRPoint {
  recall: number;
  precision: number;
}

export interface ModelData {
  id: string;
  name: string;
  shortName: string;
  color: string;
  inferenceTime: number; // ms
  fps: number;
  gpuUsage: number; // %
  vramUsage: number; // GB
  vramTotal: number; // GB
  detections: Detection[];
  mAP: number;
  prCurve: PRPoint[];
  ioU: number;
  consensusScore: number;
  visualizedImage?: string;
  heatmapImage?: string;
}

export const MODEL_DATA: Record<string, ModelData> = {
  "cascade-rcnn": {
    id: "cascade-rcnn",
    name: "Cascade R-CNN R50",
    shortName: "CAS-R50",
    color: "#00FF41",
    inferenceTime: 0,
    fps: 0,
    gpuUsage: 0,
    vramUsage: 0,
    vramTotal: 6,
    detections: [],
    mAP: 0,
    prCurve: [],
    ioU: 0,
    consensusScore: 0,
  },
};

export const MODEL_OPTIONS = [
  { id: "cascade-rcnn", label: "Cascade R-CNN R50" },
];

export const ENSEMBLE_DATA = {
  wbfBbox: [0.12, 0.17, 0.695, 0.65],
  finalLabel: "F-16 Fighting Falcon",
  finalConfidence: 0.947,
  avgInference: 65,
  modelAgreement: 0.94,
  wbfIoU: 0.861,
  comparisonTable: [
    {
      model: "Cascade R-CNN",
      mAP: 89.4,
      fps: 7.8,
      ioU: 0.847,
      vram: "4.2 GB",
    },
    { model: "YOLOv11", mAP: 86.2, fps: 43.5, ioU: 0.812, vram: "2.1 GB" },
    {
      model: "Model 3 Custom",
      mAP: 91.8,
      fps: 22.2,
      ioU: 0.873,
      vram: "3.4 GB",
    },
    {
      model: "WBF Ensemble",
      mAP: 93.1,
      fps: "—",
      ioU: 0.861,
      vram: "9.7 GB",
    },
  ],
};

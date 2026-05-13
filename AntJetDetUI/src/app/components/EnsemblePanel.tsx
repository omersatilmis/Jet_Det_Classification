import React, { useState } from "react";
import { GitMerge, ChevronDown, ChevronUp, AlertTriangle, CheckCircle } from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import { motion, AnimatePresence } from "motion/react";
import { ModelData } from "./hud-data";

interface EnsemblePanelProps {
  activeModels: ModelData[];
  isAnalyzed: boolean;
  ensembleResult?: any;
}

export function EnsemblePanel({ activeModels, isAnalyzed, ensembleResult }: EnsemblePanelProps) {
  const [expanded, setExpanded] = useState(true);

  const hasMultiple = activeModels.length > 1;

  // Calculate averages from backend metrics when available
  const analyzedModels = activeModels.filter(m => m.inferenceTime > 0);
  const fallbackAvgInference = analyzedModels.length > 0
    ? (analyzedModels.reduce((acc, m) => acc + m.inferenceTime, 0) / analyzedModels.length)
    : 0;
  const fallbackAvgFps = analyzedModels.length > 0
    ? (analyzedModels.reduce((acc, m) => acc + m.fps, 0) / analyzedModels.length)
    : 0;
  const fallbackAvgGpu = analyzedModels.length > 0
    ? (analyzedModels.reduce((acc, m) => acc + (m.gpuUsage || 0), 0) / analyzedModels.length)
    : 0;
  const fallbackAvgVram = analyzedModels.length > 0
    ? (analyzedModels.reduce((acc, m) => acc + (m.vramUsage || 0), 0) / analyzedModels.length)
    : 0;

  const ensembleMetrics = ensembleResult?.metrics;
  const avgInference = (ensembleMetrics?.avg_inference_time_ms ?? fallbackAvgInference).toFixed(1);
  const avgFps = (ensembleMetrics?.avg_fps ?? fallbackAvgFps).toFixed(1);
  const avgGpu = (ensembleMetrics?.avg_gpu_usage ?? fallbackAvgGpu).toFixed(1);
  const avgVram = (ensembleMetrics?.avg_vram_usage_mb != null
    ? (ensembleMetrics.avg_vram_usage_mb / 1024.0)
    : fallbackAvgVram).toFixed(1);

  // Prepare comparison data for chart and table
  const comparisonData = [
    ...activeModels.map(m => ({
      model: m.shortName,
      mAP: m.mAP && m.mAP > 0 ? m.mAP : null,
      ioU: m.ioU && m.ioU > 0 ? m.ioU : null,
      fps: m.fps || 0,
      color: m.color
    })),
    ...(isAnalyzed && hasMultiple ? [{
      model: "WBF Ensemble",
      mAP: ensembleMetrics?.avg_map ?? null,
      ioU: ensembleMetrics?.avg_iou ?? null,
      fps: parseFloat(avgFps),
      color: "#00E5FF"
    }] : [])
  ];

  // Get primary ensemble detection
  const primaryEnsemble = ensembleResult?.detections?.[0];

  return (
    <div
      style={{
        background: "#040b14",
        borderTop: "1px solid #0d2030",
        fontFamily: "'Share Tech Mono', monospace",
      }}
    >
      <button
        className="w-full flex items-center justify-between px-6 py-3"
        style={{ borderBottom: expanded ? "1px solid #0d2030" : "none" }}
        onClick={() => setExpanded((v) => !v)}
      >
        <div className="flex items-center gap-3">
          <GitMerge size={16} style={{ color: "#00E5FF" }} />
          <div>
            <span
              style={{
                fontFamily: "'Orbitron', sans-serif",
                color: "#00E5FF",
                fontSize: "12px",
                letterSpacing: "0.2em",
                fontWeight: 600,
              }}
            >
              ENSEMBLE SYNTHESIS — HİBRİT KARAR VE ORTALAMALAMA
            </span>
            {isAnalyzed && (
              <span
                className="ml-4"
                style={{ color: "#3a6a5a", fontSize: "9px" }}
              >
                WBF · {activeModels.length} MODEL · AVG LATENCY: {avgInference}ms
              </span>
            )}
          </div>
        </div>
        <div className="flex items-center gap-4">
          {isAnalyzed && hasMultiple && (
            <div
              className="flex items-center gap-2 px-3 py-1 rounded"
              style={{
                background: "rgba(0,255,65,0.08)",
                border: "1px solid rgba(0,255,65,0.25)",
              }}
            >
              <CheckCircle size={12} style={{ color: "#00FF41" }} />
              <span style={{ color: "#00FF41", fontSize: "9px", letterSpacing: "0.15em" }}>
                CONSENSUS ACHIEVED ({(ensembleResult?.metrics?.consensus_score * 100 || 0).toFixed(0)}%)
              </span>
            </div>
          )}
          {expanded ? (
            <ChevronUp size={14} style={{ color: "#3a5a7a" }} />
          ) : (
            <ChevronDown size={14} style={{ color: "#3a5a7a" }} />
          )}
        </div>
      </button>

      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3 }}
            style={{ overflow: "hidden" }}
          >
            <div className="p-6">
              {!isAnalyzed ? (
                <div
                  className="text-center py-4"
                  style={{ color: "#1a3a4a", fontSize: "10px", letterSpacing: "0.3em" }}
                >
                  — GÖRSEL ANALİZİ BEKLENIYOR — ENSEMBLE VERİSİ MEVCUT DEĞİL —
                </div>
              ) : (
                <div className="grid grid-cols-1 gap-6" style={{ gridTemplateColumns: "1fr 1fr 1fr" }}>
                  <div className="space-y-3">
                    <SectionLabel color="#00E5FF">
                      WBF ORTALAMA TESPİT ÇIKTISI
                    </SectionLabel>
                    <div
                      className="p-4 rounded space-y-2"
                      style={{
                        background: "#020810",
                        border: "1px solid #00E5FF30",
                      }}
                    >
                      <div className="flex items-center gap-2 mb-3">
                        <motion.div
                          className="w-3 h-3 rounded-full"
                          style={{
                            background: "#00FF41",
                            boxShadow: "0 0 8px #00FF41",
                          }}
                          animate={{ opacity: [0.5, 1, 0.5] }}
                          transition={{ duration: 1.2, repeat: Infinity }}
                        />
                        <span
                          style={{
                            color: "#00FF41",
                            fontSize: "12px",
                            letterSpacing: "0.1em",
                          }}
                        >
                          {primaryEnsemble?.class_name?.toUpperCase() || "JET-DET-CONSENSUS"}
                        </span>
                      </div>
                      <DataRow
                        label="P_conf (WBF)"
                        value={`${(primaryEnsemble?.confidence * 100 || 0).toFixed(2)}%`}
                        color="#00FF41"
                        highlight
                      />
                      <DataRow
                        label="WBF BBox (x,y,w,h)"
                        value={primaryEnsemble ? `(${primaryEnsemble.box.x.toFixed(2)}, ${primaryEnsemble.box.y.toFixed(2)}, ${primaryEnsemble.box.width.toFixed(2)}, ${primaryEnsemble.box.height.toFixed(2)})` : "N/A"}
                        color="#00E5FF"
                      />
                      <DataRow
                        label="Model Anlaşması"
                        value={`${((primaryEnsemble as any)?.agreement || 0) * 100}%`}
                        color="#00FF41"
                      />
                      <DataRow
                        label="Ort. Çıkarım Süresi"
                        value={`${avgInference} ms`}
                        color="#FF8C00"
                      />
                      <DataRow
                        label="Sistem VRAM (Avg)"
                        value={`${avgVram} GB`}
                        color="#FF3366"
                      />
                    </div>
                  </div>

                  <div className="space-y-3">
                    <SectionLabel color="#00E5FF">
                      KARAR DESTEK GRAFİĞİ — mAP KARŞILAŞTIRMA
                    </SectionLabel>
                    <ResponsiveContainer width="100%" height={180}>
                      <BarChart
                        data={comparisonData}
                        margin={{ top: 4, right: 4, left: -16, bottom: 4 }}
                        barCategoryGap="30%"
                      >
                        <XAxis
                          dataKey="model"
                          tick={{
                            fill: "#2a4a6a",
                            fontSize: 7,
                            fontFamily: "Share Tech Mono",
                          }}
                          tickLine={false}
                          axisLine={{ stroke: "#0d2030" }}
                        />
                        <YAxis
                          domain={[0, 100]}
                          tick={{
                            fill: "#2a4a6a",
                            fontSize: 7,
                            fontFamily: "Share Tech Mono",
                          }}
                          tickLine={false}
                          axisLine={{ stroke: "#0d2030" }}
                        />
                        <Tooltip
                          contentStyle={{
                            background: "#050c14",
                            border: "1px solid #00E5FF30",
                            fontSize: "9px",
                            fontFamily: "Share Tech Mono",
                            color: "#00E5FF",
                          }}
                          formatter={(v: any) => [v > 0 ? `${v}%` : "N/A", "mAP"]}
                        />
                        <Bar dataKey="mAP" radius={[2, 2, 0, 0]}>
                          {comparisonData.map((entry, i) => (
                            <Cell
                              key={i}
                              fill={entry.color}
                              opacity={entry.model === "WBF Ensemble" ? 1 : 0.7}
                            />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>

                  <div className="space-y-3">
                    <SectionLabel color="#00E5FF">
                      HATA PAYI KARŞILAŞTIRMA TABLOSU
                    </SectionLabel>
                    <div
                      className="rounded overflow-hidden"
                      style={{ border: "1px solid #0d2030" }}
                    >
                      <div
                        className="grid text-center"
                        style={{
                          gridTemplateColumns: "2fr 1fr 1fr 1fr",
                          background: "#020810",
                          borderBottom: "1px solid #0d2030",
                          padding: "6px 8px",
                          fontSize: "8px",
                          letterSpacing: "0.15em",
                          color: "#3a5a7a",
                        }}
                      >
                        <div className="text-left">MODEL</div>
                        <div>mAP%</div>
                        <div>AP50</div>
                        <div>FPS</div>
                      </div>

                      {comparisonData.map((row, i) => {
                        const isEnsemble = row.model === "WBF Ensemble";
                        return (
                          <div
                            key={i}
                            className="grid text-center"
                            style={{
                              gridTemplateColumns: "2fr 1fr 1fr 1fr",
                              padding: "5px 8px",
                              fontSize: "9px",
                              background: isEnsemble
                                ? "rgba(0,229,255,0.06)"
                                : "transparent",
                              borderBottom:
                                i < comparisonData.length - 1
                                  ? "1px solid #080f1a"
                                  : "none",
                              fontFamily: "'Share Tech Mono', monospace",
                            }}
                          >
                            <div
                              className="text-left flex items-center gap-1"
                              style={{ color: row.color }}
                            >
                              <div
                                className="w-1.5 h-1.5 rounded-full"
                                style={{ background: row.color, flexShrink: 0 }}
                              />
                              <span>{row.model}</span>
                            </div>
                            <div style={{ color: "#e0f4ff" }}>{row.mAP ? row.mAP.toFixed(1) : "—"}</div>
                            <div style={{ color: "#e0f4ff" }}>{row.ioU ? row.ioU.toFixed(2) : "—"}</div>
                            <div
                              style={{
                                color:
                                  row.fps > 30 ? "#00FF41" : row.fps > 10 ? "#FF8C00" : "#e0f4ff",
                              }}
                            >
                              {row.fps.toFixed(1)}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function SectionLabel({ children, color }: { children: React.ReactNode; color: string }) {
  return (
    <div
      className="flex items-center gap-2 pb-1"
      style={{ borderBottom: `1px solid ${color}20` }}
    >
      <div className="w-1 h-3 rounded-sm" style={{ background: color }} />
      <span
        style={{ color: "#3a5a7a", fontSize: "9px", letterSpacing: "0.25em" }}
      >
        {children}
      </span>
    </div>
  );
}

function DataRow({
  label,
  value,
  color,
  highlight,
}: {
  label: string;
  value: string;
  color: string;
  highlight?: boolean;
}) {
  return (
    <div className="flex items-center justify-between">
      <span
        style={{ color: "#2a4a6a", fontSize: "9px", letterSpacing: "0.1em" }}
      >
        {label}
      </span>
      <span
        style={{
          color,
          fontSize: "10px",
          letterSpacing: "0.1em",
          textShadow: highlight ? `0 0 8px ${color}` : "none",
        }}
      >
        {value}
      </span>
    </div>
  );
}
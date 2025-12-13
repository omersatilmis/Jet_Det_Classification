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
import { ENSEMBLE_DATA, ModelData } from "./hud-data";

interface EnsemblePanelProps {
  activeModels: ModelData[];
  isAnalyzed: boolean;
}

export function EnsemblePanel({ activeModels, isAnalyzed }: EnsemblePanelProps) {
  const [expanded, setExpanded] = useState(true);

  const hasMultiple = activeModels.length > 1;

  const radarData = [
    {
      subject: "mAP",
      "Cascade R-CNN": 89.4,
      YOLOv11: 86.2,
      "Custom Model": 91.8,
      WBF: 93.1,
    },
    {
      subject: "IoU",
      "Cascade R-CNN": 84.7,
      YOLOv11: 81.2,
      "Custom Model": 87.3,
      WBF: 86.1,
    },
    {
      subject: "Speed",
      "Cascade R-CNN": 35,
      YOLOv11: 95,
      "Custom Model": 60,
      WBF: 50,
    },
    {
      subject: "P_conf",
      "Cascade R-CNN": 95.3,
      YOLOv11: 91.7,
      "Custom Model": 97.1,
      WBF: 94.7,
    },
    {
      subject: "Consensus",
      "Cascade R-CNN": 91,
      YOLOv11: 87,
      "Custom Model": 94,
      WBF: 100,
    },
  ];

  return (
    <div
      style={{
        background: "#040b14",
        borderTop: "1px solid #0d2030",
        fontFamily: "'Share Tech Mono', monospace",
      }}
    >
      {/* Section header (always visible) */}
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
                WBF · {activeModels.length} MODEL · mAP: 93.1%
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
                CONSENSUS ACHIEVED
              </span>
            </div>
          )}
          {isAnalyzed && !hasMultiple && (
            <div
              className="flex items-center gap-2 px-3 py-1 rounded"
              style={{
                background: "rgba(255,140,0,0.08)",
                border: "1px solid rgba(255,140,0,0.25)",
              }}
            >
              <AlertTriangle size={12} style={{ color: "#FF8C00" }} />
              <span style={{ color: "#FF8C00", fontSize: "9px", letterSpacing: "0.15em" }}>
                ENSEMBLE İÇİN EN AZ 2 MODEL GEREKLİ
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

      {/* Expanded content */}
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
                  — GÖRSEL ANALİZİ BEKLENIYOR — ENSEMBLe VERİSİ MEVCUT DEĞİL —
                </div>
              ) : (
                <div className="grid grid-cols-1 gap-6" style={{ gridTemplateColumns: "1fr 1fr 1fr" }}>
                  {/* WBF Result */}
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
                          {ENSEMBLE_DATA.finalLabel.toUpperCase()}
                        </span>
                      </div>
                      <DataRow
                        label="P_conf (WBF)"
                        value={`${(ENSEMBLE_DATA.finalConfidence * 100).toFixed(2)}%`}
                        color="#00FF41"
                        highlight
                      />
                      <DataRow
                        label="WBF BBox (x,y,w,h)"
                        value={`(${ENSEMBLE_DATA.wbfBbox
                          .map((v) => (v * 1000).toFixed(0))
                          .join(", ")})`}
                        color="#00E5FF"
                      />
                      <DataRow
                        label="Model Anlaşması"
                        value={`${(ENSEMBLE_DATA.modelAgreement * 100).toFixed(0)}%`}
                        color="#00FF41"
                      />
                      <DataRow
                        label="WBF IoU"
                        value={ENSEMBLE_DATA.wbfIoU.toFixed(3)}
                        color="#00E5FF"
                      />
                      <DataRow
                        label="Ort. Çıkarım Süresi"
                        value={`${ENSEMBLE_DATA.avgInference} ms`}
                        color="#FF8C00"
                      />
                    </div>
                  </div>

                  {/* Comparison bar chart */}
                  <div className="space-y-3">
                    <SectionLabel color="#00E5FF">
                      KARAR DESTEK GRAFİĞİ — mAP KARŞILAŞTIRMA
                    </SectionLabel>
                    <ResponsiveContainer width="100%" height={180}>
                      <BarChart
                        data={ENSEMBLE_DATA.comparisonTable}
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
                          domain={[75, 100]}
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
                          formatter={(v) => [`${v}%`, "mAP"]}
                        />
                        <Bar dataKey="mAP" radius={[2, 2, 0, 0]}>
                          {ENSEMBLE_DATA.comparisonTable.map((entry, i) => (
                            <Cell
                              key={i}
                              fill={
                                entry.model === "WBF Ensemble"
                                  ? "#00E5FF"
                                  : i === 0
                                  ? "#00FF41"
                                  : i === 1
                                  ? "#00E5FF"
                                  : "#FF8C00"
                              }
                              opacity={entry.model === "WBF Ensemble" ? 1 : 0.7}
                            />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>

                  {/* Comparison table */}
                  <div className="space-y-3">
                    <SectionLabel color="#00E5FF">
                      HATA PAYI KARŞILAŞTIRMA TABLOSU
                    </SectionLabel>
                    <div
                      className="rounded overflow-hidden"
                      style={{ border: "1px solid #0d2030" }}
                    >
                      {/* Table header */}
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
                        <div>IoU</div>
                        <div>FPS</div>
                      </div>

                      {ENSEMBLE_DATA.comparisonTable.map((row, i) => {
                        const isEnsemble = row.model === "WBF Ensemble";
                        const rowColor = isEnsemble
                          ? "#00E5FF"
                          : i === 0
                          ? "#00FF41"
                          : i === 1
                          ? "#00E5FF"
                          : "#FF8C00";
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
                                i < ENSEMBLE_DATA.comparisonTable.length - 1
                                  ? "1px solid #080f1a"
                                  : "none",
                              fontFamily: "'Share Tech Mono', monospace",
                            }}
                          >
                            <div
                              className="text-left flex items-center gap-1"
                              style={{ color: rowColor }}
                            >
                              <div
                                className="w-1.5 h-1.5 rounded-full"
                                style={{ background: rowColor, flexShrink: 0 }}
                              />
                              <span>{row.model}</span>
                            </div>
                            <div style={{ color: "#e0f4ff" }}>{row.mAP}</div>
                            <div style={{ color: "#e0f4ff" }}>{row.ioU}</div>
                            <div
                              style={{
                                color:
                                  typeof row.fps === "number" && row.fps > 30
                                    ? "#00FF41"
                                    : typeof row.fps === "number" && row.fps > 10
                                    ? "#FF8C00"
                                    : "#e0f4ff",
                              }}
                            >
                              {typeof row.fps === "number"
                                ? row.fps.toFixed(1)
                                : row.fps}
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
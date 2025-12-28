import React, { useState } from "react";
import {
  ChevronDown,
  ChevronRight,
  Cpu,
  Zap,
  BarChart2,
  Target,
  Network,
  BookOpen,
  Image as ImageIcon,
  RefreshCw,
  Play,
  X,
} from "lucide-react";
import { motion, AnimatePresence } from "motion/react";
import { ModelData } from "./hud-data";

import { HudTab } from "./inference/hud_layer/HudTab";
import { PerfTab } from "./inference/performance/PerfTab";
import { ObjectsTab } from "./inference/objects/ObjectsTab";
import { EnsembleTab } from "./inference/ensemble/EnsembleTab";
import { AcademicTab } from "./inference/academic/AcademicTab";
import { OutputTab } from "./inference/output/OutputTab";

const JET_THUMB =
  "https://images.unsplash.com/photo-1750526997059-7ad806d66411?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxGLTE2JTIwZmlnaHRlciUyMGpldCUyMG1pbGl0YXJ5JTIwYWlyY3JhZnQlMjBmbHlpbmd8ZW58MXx8fHwxNzcxNzY1NDgyfDA&ixlib=rb-4.1.0&q=80&w=200";

interface ModelCardProps {
  model: ModelData;
  isAnalyzed: boolean;
  defaultOpen?: boolean;
  isAnalyzing: boolean;
  canAnalyze: boolean;
  onAnalyze: () => void;
  imageUrl?: string | null;
  onRemove?: () => void;
}

type TabKey = "hud" | "perf" | "objects" | "ensemble" | "academic" | "output";

const TABS: { key: TabKey; label: string; icon: React.ReactNode }[] = [
  { key: "hud", label: "HUD LAYER", icon: <Target size={10} /> },
  { key: "perf", label: "PERFORMANCE", icon: <Cpu size={10} /> },
  { key: "objects", label: "OBJECTS", icon: <BarChart2 size={10} /> },
  { key: "ensemble", label: "ENSEMBLE", icon: <Network size={10} /> },
  { key: "academic", label: "AKADEMIK", icon: <BookOpen size={10} /> },
  { key: "output", label: "OUTPUT", icon: <ImageIcon size={10} /> },
];

export function ModelCard({
  model,
  isAnalyzed,
  defaultOpen,
  isAnalyzing,
  canAnalyze,
  onAnalyze,
  imageUrl,
  onRemove,
}: ModelCardProps) {
  const [isOpen, setIsOpen] = useState(defaultOpen ?? false);
  const [activeTab, setActiveTab] = useState<TabKey>("hud");

  return (
    <div
      className="rounded overflow-hidden"
      style={{
        border: `1px solid ${isOpen ? model.color + "40" : "#0d2030"}`,
        background: "#050c14",
        transition: "border-color 0.3s",
      }}
    >
      {/* Card header */}
      <div
        className="w-full flex items-center justify-between px-4 py-3 group cursor-pointer"
        style={{
          background: isOpen ? `${model.color}08` : "transparent",
          transition: "background 0.3s",
          fontFamily: "'Share Tech Mono', monospace",
        }}
        onClick={() => setIsOpen((v) => !v)}
      >
        <div className="flex items-center gap-3">
          {/* Status dot */}
          <div
            className="w-2 h-2 rounded-full"
            style={{
              background: isAnalyzing ? "#e0f4ff" : (isAnalyzed ? model.color : "#1a3a4a"),
              boxShadow: isAnalyzing ? "0 0 6px #e0f4ff" : (isAnalyzed ? `0 0 6px ${model.color}` : "none"),
              transition: "all 0.3s",
            }}
          />
          {/* Model name */}
          <div className="text-left">
            <div
              style={{
                color: model.color,
                fontSize: "12px",
                letterSpacing: "0.15em",
              }}
            >
              {model.name.toUpperCase()}
            </div>
            {isAnalyzed && !isAnalyzing && (
              <div
                style={{ color: "#3a6a5a", fontSize: "9px", letterSpacing: "0.1em" }}
              >
                mAP: {model.mAP}% · t_inf: {model.inferenceTime}ms · {model.fps} FPS
              </div>
            )}
            {isAnalyzing && (
              <div style={{ color: "#e0f4ff", fontSize: "9px", letterSpacing: "0.1em" }} className="animate-pulse">
                ANALİZ EDİLİYOR...
              </div>
            )}
          </div>
        </div>

        <div className="flex flex-col items-end gap-2">
          {/* Action Buttons Container */}
          <div className="flex items-center gap-2">
            {!isAnalyzing && canAnalyze && (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onAnalyze();
                }}
                className="flex items-center gap-1.5 px-3 py-1 rounded transition-colors"
                style={{
                  background: "rgba(0,255,65,0.1)",
                  border: "1px solid rgba(0,255,65,0.3)",
                  color: "#00FF41",
                  fontSize: "9px",
                  letterSpacing: "0.1em"
                }}
              >
                <Play size={10} />
                <span>BAŞLAT</span>
              </button>
            )}

            {onRemove && (
              <button
                className="opacity-50 hover:opacity-100 transition-opacity duration-200 p-1.5 rounded"
                style={{
                  background: "rgba(255,34,68,0.1)",
                  border: "1px solid rgba(255,34,68,0.3)",
                }}
                onClick={(e) => {
                  e.stopPropagation();
                  onRemove();
                }}
                title="Modeli kaldır"
              >
                <X size={10} style={{ color: "#FF2244" }} />
              </button>
            )}
          </div>

          <div className="flex items-center gap-3">
            {isAnalyzing && (
              <div style={{ color: "#e0f4ff" }}>
                <RefreshCw size={12} className="animate-spin" />
              </div>
            )}
            {isAnalyzed && !isAnalyzing && (
              <div
                className="px-2 py-0.5 rounded"
                style={{
                  background: `${model.color}15`,
                  border: `1px solid ${model.color}40`,
                  color: model.color,
                  fontSize: "9px",
                  letterSpacing: "0.15em",
                }}
              >
                LOCK-ON
              </div>
            )}
            <div style={{ color: "#3a5a7a" }}>
              {isOpen ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
            </div>
          </div>
        </div>
      </div>

      {/* Expandable content */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3, ease: "easeInOut" }}
            style={{ overflow: "hidden" }}
          >
            <div style={{ borderTop: `1px solid ${model.color}20` }}>
              {/* Tabs */}
              <div
                className="flex overflow-x-auto"
                style={{ borderBottom: "1px solid #0d2030" }}
              >
                {TABS.map((tab) => (
                  <button
                    key={tab.key}
                    className="flex items-center gap-1.5 px-3 py-2 whitespace-nowrap transition-all duration-200"
                    style={{
                      fontFamily: "'Share Tech Mono', monospace",
                      fontSize: "9px",
                      letterSpacing: "0.15em",
                      color:
                        activeTab === tab.key ? model.color : "#3a5a7a",
                      borderBottom:
                        activeTab === tab.key
                          ? `2px solid ${model.color}`
                          : "2px solid transparent",
                      background:
                        activeTab === tab.key
                          ? `${model.color}08`
                          : "transparent",
                    }}
                    onClick={() => setActiveTab(tab.key)}
                  >
                    {tab.icon}
                    {tab.label}
                  </button>
                ))}
              </div>

              {/* Tab content */}
              <div className="p-4">
                {activeTab === "hud" && (
                  <HudTab model={model} isAnalyzed={isAnalyzed} />
                )}
                {activeTab === "perf" && (
                  <PerfTab model={model} isAnalyzed={isAnalyzed} />
                )}
                {activeTab === "objects" && (
                  <ObjectsTab model={model} isAnalyzed={isAnalyzed} />
                )}
                {activeTab === "ensemble" && (
                  <EnsembleTab model={model} isAnalyzed={isAnalyzed} />
                )}
                {activeTab === "academic" && (
                  <AcademicTab model={model} isAnalyzed={isAnalyzed} />
                )}
                {activeTab === "output" && (
                  <OutputTab model={model} imageUrl={imageUrl} />
                )}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}


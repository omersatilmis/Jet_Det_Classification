import React, { useState, useRef, useCallback } from "react";
import { Upload, Crosshair, ZoomIn, Target, Lock } from "lucide-react";
import { motion, AnimatePresence } from "motion/react";
import { ModelData } from "./hud-data";

const JET_IMAGE =
  "https://images.unsplash.com/photo-1750526997059-7ad806d66411?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxGLTE2JTIwZmlnaHRlciUyMGpldCUyMG1pbGl0YXJ5JTIwYWlyY3JhZnQlMjBmbHlpbmd8ZW58MXx8fHwxNzcxNzY1NDgyfDA&ixlib=rb-4.1.0&q=80&w=1080";

interface LeftPanelProps {
  activeModels: ModelData[];
  imageUrl: string | null;
  isAnalyzing: boolean;
  onLoadImage: () => void;
  selectedFile?: File | null;
  progress?: number;
}

export function LeftPanel({
  activeModels,
  imageUrl,
  isAnalyzing,
  onLoadImage,
  selectedFile,
  progress,
}: LeftPanelProps) {
  const [isDragOver, setIsDragOver] = useState(false);
  const [scanLine, setScanLine] = useState(0);
  const scanRef = useRef<number | null>(null);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);
  const handleDragLeave = useCallback(() => setIsDragOver(false), []);
  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragOver(false);
      onLoadImage();
    },
    [onLoadImage]
  );

  return (
    <div
      className="flex flex-col h-full"
      style={{ fontFamily: "'Share Tech Mono', monospace" }}
    >
      {/* Panel header */}
      <div
        className="flex items-center justify-between px-4 py-2"
        style={{ borderBottom: "1px solid #0d2030" }}
      >
        <div className="flex items-center gap-2">
          <ZoomIn size={14} style={{ color: "#00E5FF" }} />
          <span style={{ color: "#00E5FF", fontSize: "11px", letterSpacing: "0.2em" }}>
            MEDIA DENETIM TERMİNALİ
          </span>
        </div>
        <div className="flex items-center gap-3">
          {activeModels.map((m) => (
            <div key={m.id} className="flex items-center gap-1">
              <div
                className="w-1.5 h-1.5 rounded-full"
                style={{ background: m.color, boxShadow: `0 0 4px ${m.color}` }}
              />
              <span style={{ color: m.color, fontSize: "9px", letterSpacing: "0.15em" }}>
                {m.shortName}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Main image/dropzone area */}
      <div className="flex-1 relative overflow-hidden p-3">
        <div
          className="w-full h-full relative rounded overflow-hidden cursor-pointer"
          style={{
            border: !!imageUrl
              ? `1px solid rgba(0,229,255,0.2)`
              : isDragOver
                ? `2px dashed #00FF41`
                : `2px dashed #1a3a4a`,
            background: "#050c14",
            minHeight: "340px",
          }}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={!imageUrl ? onLoadImage : undefined}
        >
          {!imageUrl ? (
            /* Empty state */
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-4">
              <motion.div
                animate={{ scale: [1, 1.08, 1], opacity: [0.6, 1, 0.6] }}
                transition={{ duration: 2.5, repeat: Infinity }}
                className="flex flex-col items-center gap-3"
              >
                {/* Aviation-style upload icon */}
                <div className="relative">
                  <div
                    className="w-20 h-20 rounded-full flex items-center justify-center"
                    style={{
                      border: "1px solid #1a3a4a",
                      background: "rgba(0,229,255,0.05)",
                    }}
                  >
                    <Target size={36} style={{ color: "#1a5a7a" }} />
                  </div>
                  <div
                    className="absolute inset-0 rounded-full animate-ping"
                    style={{
                      border: "1px solid rgba(0,229,255,0.2)",
                      animationDuration: "2s",
                    }}
                  />
                </div>
                <div style={{ color: "#1a5a7a", fontSize: "12px", letterSpacing: "0.3em" }}>
                  ANALİZ İÇİN GÖRSEL/VİDEO EKLEYİN
                </div>
                <div style={{ color: "#0d2a3a", fontSize: "10px", letterSpacing: "0.2em" }}>
                  DRAG &amp; DROP / CLICK TO UPLOAD
                </div>
              </motion.div>
              {/* Corner decorations */}
              <CornerBrackets color="#1a3a4a" />
            </div>
          ) : (
            /* Image loaded with HUD overlay */
            <>
              <img
                src={imageUrl}
                alt="Aircraft detection target"
                className="w-full h-full object-cover"
                style={{ filter: "brightness(0.85) saturate(0.9)" }}
              />

              {/* CRT scanline overlay */}
              <div
                className="absolute inset-0 pointer-events-none"
                style={{
                  background:
                    "repeating-linear-gradient(0deg, transparent, transparent 3px, rgba(0,0,0,0.08) 3px, rgba(0,0,0,0.08) 4px)",
                }}
              />

              {/* Vignette */}
              <div
                className="absolute inset-0 pointer-events-none"
                style={{
                  background:
                    "radial-gradient(ellipse at center, transparent 50%, rgba(0,0,0,0.6) 100%)",
                }}
              />

              {/* Scan animation when analyzing */}
              {isAnalyzing && (
                <motion.div
                  className="absolute inset-x-0 h-0.5 pointer-events-none"
                  style={{
                    background:
                      "linear-gradient(90deg, transparent, #00FF41, #00FF41, transparent)",
                    boxShadow: "0 0 12px #00FF41",
                    top: 0,
                  }}
                  animate={{ top: ["0%", "100%", "0%"] }}
                  transition={{ duration: 2.5, repeat: Infinity, ease: "linear" }}
                />
              )}

              {/* Bounding boxes for each active model */}
              {activeModels.map((model) =>
                model.detections.map((det, i) => (
                  <BoundingBox
                    key={`${model.id}-${i}`}
                    detection={det}
                    modelColor={model.color}
                    modelName={model.shortName}
                    isFirst={i === 0}
                    isAnalyzing={isAnalyzing}
                  />
                ))
              )}

              {/* HUD corner info */}
              <div className="absolute top-2 left-2" style={{ color: "#00FF41", fontSize: "9px" }}>
                <div>SYS: ARMED</div>
                <div>MODE: DETECTION</div>
                <div>RES: 1080×720</div>
              </div>

              <div className="absolute top-2 right-2 text-right" style={{ color: "#00E5FF", fontSize: "9px" }}>
                <div>FRAME: 001</div>
                <div>ZOOM: 1.0x</div>
                <div>SECTOR: ALPHA-7</div>
              </div>

              <div className="absolute bottom-2 left-2" style={{ color: "#00FF41", fontSize: "9px" }}>
                <div>LAT: 37.4219° N</div>
                <div>LON: 35.6687° E</div>
              </div>

              <div className="absolute bottom-2 right-2 text-right" style={{ color: "#FF8C00", fontSize: "9px" }}>
                <div>ALT: 8,400 FT</div>
                <div>SPD: 480 KT</div>
              </div>

              <CornerBrackets color="rgba(0,229,255,0.4)" />
            </>
          )}
        </div>
      </div>

      {/* Bottom action bar */}
      <div
        className="px-3 pb-3 pt-1"
        style={{ borderTop: "1px solid #0a1f2e" }}
      >
        <button
          onClick={onLoadImage}
          className="w-full flex items-center justify-center gap-3 py-3 rounded transition-all duration-300 relative overflow-hidden group"
          style={{
            background: imageUrl
              ? "rgba(0,229,255,0.05)"
              : "rgba(0,255,65,0.08)",
            border: `1px solid ${imageUrl ? "#0d3a4a" : "#0d3a1a"}`,
            color: imageUrl ? "#00E5FF" : "#00FF41",
            fontSize: "11px",
            letterSpacing: "0.25em",
          }}
        >
          <div
            className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-300"
            style={{
              background: imageUrl
                ? "rgba(0,229,255,0.06)"
                : "rgba(0,255,65,0.06)",
            }}
          />
          <Upload size={14} />
          <span>{imageUrl ? "YENİ MEDYA YÜKLE" : "DOSYA EKLE"}</span>
        </button>
      </div>
    </div>
  );
}

function BoundingBox({
  detection,
  modelColor,
  modelName,
  isFirst,
  isAnalyzing,
}: {
  detection: {
    label: string;
    confidence: number;
    bbox: [number, number, number, number];
    targetId: string;
  };
  modelColor: string;
  modelName: string;
  isFirst: boolean;
  isAnalyzing: boolean;
}) {
  const [x, y, w, h] = detection.bbox;
  return (
    <div
      className="absolute pointer-events-none"
      style={{
        left: `${x * 100}%`,
        top: `${y * 100}%`,
        width: `${w * 100}%`,
        height: `${h * 100}%`,
      }}
    >
      {/* Box border */}
      <div
        className="absolute inset-0"
        style={{
          border: `1px solid ${modelColor}`,
          boxShadow: `0 0 8px ${modelColor}40, inset 0 0 20px ${modelColor}08`,
        }}
      />

      {/* Corner brackets */}
      {isFirst && (
        <>
          <TargetCorner pos="tl" color={modelColor} size={12} />
          <TargetCorner pos="tr" color={modelColor} size={12} />
          <TargetCorner pos="bl" color={modelColor} size={12} />
          <TargetCorner pos="br" color={modelColor} size={12} />
        </>
      )}

      {/* Label */}
      <div
        className="absolute -top-5 left-0 flex items-center gap-1.5 px-1.5 py-0.5"
        style={{
          background: `${modelColor}15`,
          border: `1px solid ${modelColor}60`,
          backdropFilter: "blur(4px)",
          whiteSpace: "nowrap",
        }}
      >
        {isAnalyzing && isFirst && (
          <Lock size={8} style={{ color: modelColor }} />
        )}
        <span
          style={{
            color: modelColor,
            fontSize: "9px",
            letterSpacing: "0.1em",
            fontFamily: "'Share Tech Mono', monospace",
          }}
        >
          [{modelName}] {detection.label.toUpperCase()} — P
          <sub>conf</sub>={" "}
          {(detection.confidence * 100).toFixed(1)}%
        </span>
      </div>

      {/* Coordinates readout */}
      {isFirst && (
        <div
          className="absolute -bottom-5 left-0"
          style={{
            color: `${modelColor}90`,
            fontSize: "8px",
            letterSpacing: "0.08em",
            fontFamily: "'Share Tech Mono', monospace",
            whiteSpace: "nowrap",
          }}
        >
          (x={Math.round(x * 1000)}, y={Math.round(y * 1000)}, w=
          {Math.round(w * 1000)}, h={Math.round(h * 1000)})
        </div>
      )}

      {/* Target lock crosshair (center, only for primary detection) */}
      {isFirst && isAnalyzing && (
        <motion.div
          className="absolute"
          style={{
            top: "50%",
            left: "50%",
            transform: "translate(-50%,-50%)",
          }}
          animate={{ opacity: [0.4, 1, 0.4], scale: [0.95, 1.05, 0.95] }}
          transition={{ duration: 1.2, repeat: Infinity }}
        >
          <Crosshair size={20} style={{ color: modelColor }} />
        </motion.div>
      )}
    </div>
  );
}

function TargetCorner({
  pos,
  color,
  size,
}: {
  pos: "tl" | "tr" | "bl" | "br";
  color: string;
  size: number;
}) {
  const s = `${size}px`;
  const style: React.CSSProperties = {
    position: "absolute",
    width: s,
    height: s,
    borderColor: color,
    borderStyle: "solid",
    boxShadow: `0 0 6px ${color}`,
  };
  if (pos === "tl")
    return (
      <div
        style={{
          ...style,
          top: -1,
          left: -1,
          borderWidth: "2px 0 0 2px",
        }}
      />
    );
  if (pos === "tr")
    return (
      <div
        style={{
          ...style,
          top: -1,
          right: -1,
          borderWidth: "2px 2px 0 0",
        }}
      />
    );
  if (pos === "bl")
    return (
      <div
        style={{
          ...style,
          bottom: -1,
          left: -1,
          borderWidth: "0 0 2px 2px",
        }}
      />
    );
  return (
    <div
      style={{
        ...style,
        bottom: -1,
        right: -1,
        borderWidth: "0 2px 2px 0",
      }}
    />
  );
}

function CornerBrackets({ color }: { color: string }) {
  const len = "16px";
  const thick = "2px";
  const inset = "8px";
  return (
    <>
      {/* TL */}
      <div
        className="absolute pointer-events-none"
        style={{ top: inset, left: inset, width: len, height: len }}
      >
        <div
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            width: len,
            height: thick,
            background: color,
          }}
        />
        <div
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            width: thick,
            height: len,
            background: color,
          }}
        />
      </div>
      {/* TR */}
      <div
        className="absolute pointer-events-none"
        style={{ top: inset, right: inset, width: len, height: len }}
      >
        <div
          style={{
            position: "absolute",
            top: 0,
            right: 0,
            width: len,
            height: thick,
            background: color,
          }}
        />
        <div
          style={{
            position: "absolute",
            top: 0,
            right: 0,
            width: thick,
            height: len,
            background: color,
          }}
        />
      </div>
      {/* BL */}
      <div
        className="absolute pointer-events-none"
        style={{ bottom: inset, left: inset, width: len, height: len }}
      >
        <div
          style={{
            position: "absolute",
            bottom: 0,
            left: 0,
            width: len,
            height: thick,
            background: color,
          }}
        />
        <div
          style={{
            position: "absolute",
            bottom: 0,
            left: 0,
            width: thick,
            height: len,
            background: color,
          }}
        />
      </div>
      {/* BR */}
      <div
        className="absolute pointer-events-none"
        style={{ bottom: inset, right: inset, width: len, height: len }}
      >
        <div
          style={{
            position: "absolute",
            bottom: 0,
            right: 0,
            width: len,
            height: thick,
            background: color,
          }}
        />
        <div
          style={{
            position: "absolute",
            bottom: 0,
            right: 0,
            width: thick,
            height: len,
            background: color,
          }}
        />
      </div>
    </>
  );
}

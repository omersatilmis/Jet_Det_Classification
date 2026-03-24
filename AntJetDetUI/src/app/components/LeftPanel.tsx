import React, { useState, useRef, useCallback, useEffect } from "react";
import { Upload, Crosshair, ZoomIn, Target, Lock } from "lucide-react";
import { motion, AnimatePresence } from "motion/react";
import { ModelData } from "./hud-data";

const JET_IMAGE =
  "https://images.unsplash.com/photo-1750526997059-7ad806d66411?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxGLTE2JTIwZmlnaHRlciUyMGpldCUyMG1pbGl0YXJ5JTIwYWlyY3JhZnQlMjBmbHlpbmd8ZW58MXx8fHwxNzcxNzY1NDgyfDA&ixlib=rb-4.1.0&q=80&w=1080";

interface LeftPanelProps {
  activeModels: ModelData[];
  imageUrl: string | null;
  videoUrl?: string | null;
  mediaType?: "image" | "video" | null;
  isAnalyzing: boolean;
  onLoadMedia: (file?: File) => void;
  onVideoTimeUpdate?: (timeMs: number | null) => void;
  selectedFile?: File | null;
  progress?: number;
}

export function LeftPanel({
  activeModels,
  imageUrl,
  videoUrl,
  mediaType,
  isAnalyzing,
  onLoadMedia,
  onVideoTimeUpdate,
  selectedFile,
  progress,
}: LeftPanelProps) {
  const [isDragOver, setIsDragOver] = useState(false);
  const [scanLine, setScanLine] = useState(0);
  const scanRef = useRef<number | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const imageContainerRef = useRef<HTMLDivElement | null>(null);
  const [imageNatural, setImageNatural] = useState<{ w: number; h: number } | null>(null);
  const [containerSize, setContainerSize] = useState<{ w: number; h: number } | null>(null);

  useEffect(() => {
    const el = imageContainerRef.current;
    if (!el) return;

    const updateSize = () => {
      const rect = el.getBoundingClientRect();
      setContainerSize({ w: Math.max(1, rect.width), h: Math.max(1, rect.height) });
    };

    updateSize();

    if (typeof ResizeObserver !== "undefined") {
      const observer = new ResizeObserver(() => updateSize());
      observer.observe(el);
      return () => observer.disconnect();
    }

    window.addEventListener("resize", updateSize);
    return () => window.removeEventListener("resize", updateSize);
  }, [imageUrl]);

  useEffect(() => {
    if (mediaType !== "video" || !videoUrl) return;
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let rafId: number | null = null;

    const drawFrame = () => {
      if (!video.videoWidth || !video.videoHeight) {
        rafId = requestAnimationFrame(drawFrame);
        return;
      }

      const rect = video.getBoundingClientRect();
      const width = Math.max(1, Math.floor(rect.width));
      const height = Math.max(1, Math.floor(rect.height));

      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }

      ctx.clearRect(0, 0, width, height);

      const nowMs = video.currentTime * 1000.0;
      const videoW = video.videoWidth || 1;
      const videoH = video.videoHeight || 1;
      const scale = Math.min(width / videoW, height / videoH);
      const drawW = videoW * scale;
      const drawH = videoH * scale;
      const offsetX = (width - drawW) / 2;
      const offsetY = (height - drawH) / 2;

      activeModels.forEach((model) => {
        const frames = model.videoFrames || [];
        if (!frames.length) return;

        let nearest = frames[0];
        let minDiff = Math.abs(frames[0].timestamp_ms - nowMs);
        for (let i = 1; i < frames.length; i += 1) {
          const diff = Math.abs(frames[i].timestamp_ms - nowMs);
          if (diff < minDiff) {
            minDiff = diff;
            nearest = frames[i];
          }
        }

        nearest.detections.forEach((det) => {
          const [x, y, w, h] = det.bbox;
          const bx = offsetX + (x * drawW);
          const by = offsetY + (y * drawH);
          const bw = w * drawW;
          const bh = h * drawH;

          ctx.strokeStyle = model.color;
          ctx.lineWidth = 2;
          ctx.strokeRect(bx, by, bw, bh);

          const label = `${model.shortName} ${det.label} ${(det.confidence * 100).toFixed(1)}%`;
          ctx.fillStyle = model.color;
          ctx.font = "10px Share Tech Mono";
          const textWidth = ctx.measureText(label).width;
          ctx.fillRect(bx, Math.max(0, by - 14), textWidth + 6, 12);
          ctx.fillStyle = "#050c14";
          ctx.fillText(label, bx + 3, Math.max(10, by - 4));
        });
      });

      rafId = requestAnimationFrame(drawFrame);
    };

    rafId = requestAnimationFrame(drawFrame);

    return () => {
      if (rafId) cancelAnimationFrame(rafId);
    };
  }, [activeModels, mediaType, videoUrl]);

  useEffect(() => {
    if (!onVideoTimeUpdate) return;
    if (mediaType !== "video") {
      onVideoTimeUpdate(null);
      return;
    }
    const video = videoRef.current;
    if (!video) return;

    const handler = () => {
      onVideoTimeUpdate(video.currentTime * 1000.0);
    };

    video.addEventListener("timeupdate", handler);
    video.addEventListener("loadedmetadata", handler);
    return () => {
      video.removeEventListener("timeupdate", handler);
      video.removeEventListener("loadedmetadata", handler);
    };
  }, [mediaType, onVideoTimeUpdate, videoUrl]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);
  const handleDragLeave = useCallback(() => setIsDragOver(false), []);
  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragOver(false);
      const file = e.dataTransfer.files?.[0];
      if (file) {
        onLoadMedia(file);
      } else {
        onLoadMedia();
      }
    },
    [onLoadMedia]
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
          onClick={!imageUrl && !videoUrl ? () => onLoadMedia() : undefined}
        >
          {!imageUrl && !videoUrl ? (
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
              {mediaType === "video" ? (
                <>
                  <video
                    ref={videoRef}
                    src={videoUrl || undefined}
                    className="w-full h-full object-contain"
                    style={{ filter: "brightness(0.85) saturate(0.9)" }}
                    controls
                    muted
                  />
                  <canvas
                    ref={canvasRef}
                    className="absolute inset-0 pointer-events-none"
                  />
                </>
              ) : (
                <div ref={imageContainerRef} className="absolute inset-0">
                  <img
                    ref={imageRef}
                    src={imageUrl || undefined}
                    alt="Aircraft detection target"
                    className="w-full h-full object-contain"
                    style={{ filter: "brightness(0.85) saturate(0.9)" }}
                    onLoad={(e) => {
                      const img = e.currentTarget;
                      setImageNatural({ w: img.naturalWidth || 1, h: img.naturalHeight || 1 });
                    }}
                  />
                </div>
              )}

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
              {mediaType !== "video" && imageNatural && containerSize && activeModels.map((model) => {
                const scale = Math.min(containerSize.w / imageNatural.w, containerSize.h / imageNatural.h);
                const drawW = imageNatural.w * scale;
                const drawH = imageNatural.h * scale;
                const offsetX = (containerSize.w - drawW) / 2;
                const offsetY = (containerSize.h - drawH) / 2;

                return model.detections.map((det, i) => (
                  <BoundingBox
                    key={`${model.id}-${i}`}
                    detection={det}
                    modelColor={model.color}
                    modelName={model.shortName}
                    isFirst={i === 0}
                    isAnalyzing={isAnalyzing}
                    pixelBox={{
                      x: offsetX + det.bbox[0] * drawW,
                      y: offsetY + det.bbox[1] * drawH,
                      w: det.bbox[2] * drawW,
                      h: det.bbox[3] * drawH,
                    }}
                  />
                ));
              })}

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
          onClick={() => onLoadMedia()}
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
          <span>{imageUrl || videoUrl ? "YENİ MEDYA YÜKLE" : "DOSYA EKLE"}</span>
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
  pixelBox,
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
  pixelBox?: { x: number; y: number; w: number; h: number };
}) {
  const [x, y, w, h] = detection.bbox;
  const left = pixelBox ? `${pixelBox.x}px` : `${x * 100}%`;
  const top = pixelBox ? `${pixelBox.y}px` : `${y * 100}%`;
  const width = pixelBox ? `${pixelBox.w}px` : `${w * 100}%`;
  const height = pixelBox ? `${pixelBox.h}px` : `${h * 100}%`;
  return (
    <div
      className="absolute pointer-events-none"
      style={{
        left,
        top,
        width,
        height,
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

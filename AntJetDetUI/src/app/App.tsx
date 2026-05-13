import React, { useState, useEffect } from "react";
import { SystemHeader } from "./components/SystemHeader";
import { LeftPanel } from "./components/LeftPanel";
import { RightPanel } from "./components/RightPanel";
import { EnsemblePanel } from "./components/EnsemblePanel";
import { MODEL_DATA, ModelData } from "./components/hud-data";
import { motion, AnimatePresence } from "motion/react";
import { Play, Square, RefreshCw, AlertCircle } from "lucide-react";

class ErrorBoundary extends React.Component<{ children: React.ReactNode }, { hasError: boolean, error: Error | null, info: React.ErrorInfo | null }> {
  constructor(props: { children: React.ReactNode }) {
    super(props);
    this.state = { hasError: false, error: null, info: null };
  }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    this.setState({ info });
    console.error("ErrorBoundary caught an error:", error, info);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{ padding: 20, background: '#1a0000', color: '#ff6666', height: '100vh', fontFamily: 'monospace' }}>
          <h2>UI Çöktü (Error Boundary Caught Exception)</h2>
          <p>Lütfen bu hatayı asistana kopyalayın:</p>
          <pre style={{ background: '#000', padding: 10, overflow: 'auto' }}>
            {this.state.error?.toString()}
            <br />
            <br />
            {this.state.info?.componentStack}
          </pre>
          <button onClick={() => window.location.reload()} style={{ padding: '8px 16px', marginTop: 20, background: '#330000', border: '1px solid #ff0000', color: 'white' }}>Sayfayı Yenile</button>
        </div>
      );
    }
    return this.props.children;
  }
}

function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [mediaType, setMediaType] = useState<"image" | "video" | null>(null);
  const [analyzingModels, setAnalyzingModels] = useState<Set<string>>(new Set());
  const [isAnalyzed, setIsAnalyzed] = useState(false);
  const [activeModels, setActiveModels] = useState([
    MODEL_DATA["cascade-rcnn-r50-tiny"]
  ]);
  const [analysisProgress, setAnalysisProgress] = useState<Record<string, number>>({});
  const [logLines, setLogLines] = useState<string[]>([]);
  const [ensembleResult, setEnsembleResult] = useState<any>(null);
  const [currentVideoTimeMs, setCurrentVideoTimeMs] = useState<number | null>(null);

  useEffect(() => {
    return () => {
      if (imageUrl) URL.revokeObjectURL(imageUrl);
      if (videoUrl) URL.revokeObjectURL(videoUrl);
    };
  }, [imageUrl, videoUrl]);

  const handleLoadMedia = (file?: File) => {
    if (file) {
      if (imageUrl) URL.revokeObjectURL(imageUrl);
      if (videoUrl) URL.revokeObjectURL(videoUrl);

      const isVideo = file.type.startsWith("video/");
      const objectUrl = URL.createObjectURL(file);
      setSelectedFile(file);
      setMediaType(isVideo ? "video" : "image");
      setImageUrl(isVideo ? null : objectUrl);
      setVideoUrl(isVideo ? objectUrl : null);

      setIsAnalyzed(false);
      setAnalyzingModels(new Set());
      setAnalysisProgress({});
      setActiveModels(prev => prev.map(m => ({
        ...m,
        detections: [],
        inferenceTime: 0,
        fps: 0,
        vramUsage: 0,
        gpuUsage: 0,
        videoFrames: []
      })));
      setLogLines([`[SYS] Medya seçildi: ${file.name}`]);
      return;
    }

    const input = document.createElement("input");
    input.type = "file";
    input.accept = "image/*,video/*";
    input.onchange = (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (file) handleLoadMedia(file);
    };
    input.click();
  };

  const handleAnalyzeAll = async () => {
    if (!selectedFile || activeModels.length === 0) return;

    const modelIds = activeModels.map(m => m.id);
    setAnalyzingModels(new Set(modelIds));
    setLogLines((prev) => [...prev, `[SYS] Sıralı çoklu model analizi başlatılıyor (${modelIds.length} model)...`]);
    
    setIsAnalyzed(false);
    setEnsembleResult(null);

    try {
      const { computeEnsemble } = await import('../api');
      
      // Run each model sequentially
      const completedResults: any[] = [];
      
      for (const mid of modelIds) {
        const result = await handleAnalyzeSingle(mid, true);
        if (result && result.models && result.models[0]) {
          completedResults.push(result.models[0]);
        }
      }
      
      if (completedResults.length > 0) {
        setLogLines((prev) => [...prev, `[WBF] Modeller tamamlandı, hibrit karar hesaplanıyor...`]);
        const ensemble = await computeEnsemble(completedResults);
        setEnsembleResult(ensemble);
        setIsAnalyzed(true);
        setLogLines((prev) => [...prev, `[SYS] Analiz Tamamlandı. Hibrit karara varıldı.`]);
      }

    } catch (error) {
      console.error(`Multi-Analysis Failed:`, error);
      setLogLines((prev) => [...prev, `[ERROR] Çoklu analiz başarısız: ${error}`]);
    } finally {
      setAnalyzingModels(new Set());
    }
  };

  const handleAnalyzeSingle = async (modelId: string, silent: boolean = false) => {
    if (!selectedFile) return null;

    if (!silent) {
        setAnalyzingModels(prev => new Set(prev).add(modelId));
        setEnsembleResult(null); // Reset ensemble if individual model is re-run
    }
    setAnalysisProgress(prev => ({ ...prev, [modelId]: 10 }));
    setLogLines((prev) => [...prev, `[SYS] ${modelId} modeli başlatılıyor...`]);

    try {
      const { analyzeSingleImage, analyzeSingleVideo } = await import('../api');
      const result = mediaType === "video"
        ? await analyzeSingleVideo(modelId, selectedFile)
        : await analyzeSingleImage(modelId, selectedFile);
      
      setAnalysisProgress(prev => ({ ...prev, [modelId]: 100 }));

      setActiveModels(prevModels => prevModels.map((baseModel) => {
        if (baseModel.id !== modelId) return baseModel;
        const backendData = result.models[0];
        if (!backendData) return baseModel;

        return {
          ...baseModel,
          inferenceTime: backendData.metrics.inference_time_ms,
          fps: backendData.metrics.fps || baseModel.fps,
          gpuUsage: backendData.metrics.gpu_usage || baseModel.gpuUsage,
          vramUsage: backendData.metrics.vram_usage_mb
            ? backendData.metrics.vram_usage_mb / 1024
            : baseModel.vramUsage,
          mAP: backendData.metrics.map ?? baseModel.mAP,
          ioU: backendData.metrics.iou ?? baseModel.ioU,
          prCurve: backendData.pr_curve ? backendData.pr_curve : baseModel.prCurve,
          videoFrames: backendData.frame_detections
            ? backendData.frame_detections.map((frame: any) => ({
                timestamp_ms: frame.timestamp_ms,
                detections: frame.detections.map((d: any, i: number) => ({
                  label: d.class_name,
                  confidence: d.confidence,
                  bbox: [d.box.x, d.box.y, d.box.width, d.box.height] as [number, number, number, number],
                  targetId: `TGT-${i + 1}`,
                  azimuth: d.azimuth,
                  elevation: d.elevation,
                  distance_km: d.distance_km,
                })),
              }))
            : [],
          detections: backendData.detections.map((d: any, i: number) => ({
            label: d.class_name,
            confidence: d.confidence,
            bbox: [d.box.x, d.box.y, d.box.width, d.box.height] as [number, number, number, number],
            targetId: `TGT-${i + 1}`,
            azimuth: d.azimuth,
            elevation: d.elevation,
            distance_km: d.distance_km
          })) as any,
          visualizedImage: backendData.visualized_image,
          heatmapImage: backendData.heatmap_image
        };
      }));

      setLogLines((prev) => [...prev, `[SYS] ${modelId} Analizi Tamamlandı.`]);
      
      if (!silent) {
          setIsAnalyzed(true);
      }
      return result;
    } catch (error) {
      console.error(`Analysis Failed for ${modelId}:`, error);
      setLogLines((prev) => [...prev, `[ERROR] ${modelId} analizi başarısız: ${error}`]);
      return null;
    } finally {
      if (!silent) {
          setAnalyzingModels(prev => {
            const next = new Set(prev);
            next.delete(modelId);
            return next;
          });
      }
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    if (imageUrl) URL.revokeObjectURL(imageUrl);
    if (videoUrl) URL.revokeObjectURL(videoUrl);
    setImageUrl(null);
    setVideoUrl(null);
    setMediaType(null);
    setAnalyzingModels(new Set());
    setIsAnalyzed(false);
    setAnalysisProgress({});
    setLogLines([]);
    setEnsembleResult(null);
  };

  return (
    <ErrorBoundary>
      <div
        className="flex flex-col"
        style={{
          background: "#040a12",
          minHeight: "100vh",
          fontFamily: "'Share Tech Mono', monospace",
          color: "#e0f4ff",
          overflow: "hidden",
        }}
      >
        <SystemHeader />

        <AnimatePresence>
          {selectedFile && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: "auto", opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.25 }}
              style={{
                background: "#030810",
                borderBottom: "1px solid #0d2030",
                overflow: "hidden",
              }}
            >
              <div className="flex items-center gap-4 px-6 py-2">
                <button
                  onClick={handleAnalyzeAll}
                  disabled={analyzingModels.size > 0 || activeModels.length === 0}
                  className="flex items-center gap-2 px-6 py-2 rounded transition-all duration-300 disabled:opacity-30 disabled:cursor-not-allowed group"
                  style={{
                    background: analyzingModels.size > 0 
                      ? "rgba(0,255,65,0.05)" 
                      : "linear-gradient(135deg, rgba(0,255,65,0.1), rgba(0,229,255,0.1))",
                    border: "1px solid rgba(0,255,65,0.3)",
                    boxShadow: analyzingModels.size > 0 ? "none" : "0 0 15px rgba(0,255,65,0.1)",
                  }}
                >
                  {analyzingModels.size > 0 ? (
                    <>
                      <RefreshCw size={14} className="animate-spin text-[#00FF41]" />
                      <span className="text-[#00FF41] text-[10px] tracking-[0.2em] font-bold">ANALİZ SÜRÜYOR</span>
                    </>
                  ) : (
                    <>
                      <Play size={14} className="text-[#00FF41] group-hover:scale-110 transition-transform" />
                      <span className="text-[#00FF41] text-[10px] tracking-[0.2em] font-bold">TÜMÜNÜ ANALİZ ET</span>
                    </>
                  )}
                </button>

                <button
                  onClick={handleReset}
                  className="flex items-center gap-2 px-4 py-2 rounded transition-all duration-200"
                  style={{
                    background: "rgba(255,34,68,0.06)",
                    border: "1px solid rgba(255,34,68,0.2)",
                    color: "#FF2244",
                    fontSize: "10px",
                    letterSpacing: "0.2em",
                  }}
                >
                  <Square size={11} />
                  SIFIRLA
                </button>

                {(analyzingModels.size > 0 || isAnalyzed) && (
                  <div className="flex-1 flex items-center gap-3">
                    <div
                      className="flex-1 h-1.5 rounded-full overflow-hidden"
                      style={{ background: "#0a1a2a" }}
                    >
                      <motion.div
                        className="h-full rounded-full"
                        style={{
                          background: isAnalyzed && analyzingModels.size === 0
                            ? "linear-gradient(90deg, #00FF41, #00E5FF)"
                            : "linear-gradient(90deg, #00FF4160, #00FF41)",
                          boxShadow: "0 0 8px #00FF4160",
                        }}
                        animate={{ width: `${isAnalyzed && analyzingModels.size === 0 ? 100 : (Object.values(analysisProgress).reduce((a, b) => a + b, 0) / Math.max(Object.keys(analysisProgress).length, 1))}%` }}
                        transition={{ duration: 0.2 }}
                      />
                    </div>
                    <span
                      style={{
                        color: isAnalyzed && analyzingModels.size === 0 ? "#00FF41" : "#3a6a5a",
                        fontSize: "9px",
                        letterSpacing: "0.15em",
                        minWidth: "80px",
                      }}
                    >
                      {isAnalyzed && analyzingModels.size === 0
                        ? "TAMAMLANDI"
                        : `${Math.round(Object.values(analysisProgress).reduce((a, b) => a + b, 0) / Math.max(Object.keys(analysisProgress).length, 1))}% — SCANNING`}
                    </span>
                  </div>
                )}

                {activeModels.length === 0 && analyzingModels.size === 0 && (
                  <div
                    className="flex items-center gap-2"
                    style={{ color: "#FF8C00", fontSize: "9px", letterSpacing: "0.15em" }}
                  >
                    <AlertCircle size={12} />
                    ANALİZ İÇİN EN AZ 1 MODEL SEÇİN
                  </div>
                )}

                <div
                  className="flex-1 overflow-hidden"
                  style={{
                    maxWidth: "500px",
                    color: "#2a5a3a",
                    fontSize: "8px",
                    letterSpacing: "0.05em",
                  }}
                >
                  <AnimatePresence mode="popLayout">
                    {logLines.slice(-1).map((line, i) => (
                      <motion.div
                        key={line}
                        initial={{ opacity: 0, y: 4 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0 }}
                        transition={{ duration: 0.15 }}
                        style={{
                          color: line.startsWith("[SYS]")
                            ? "#00FF41"
                            : line.startsWith("[WBF]")
                              ? "#00E5FF"
                              : "#3a6a5a",
                        }}
                      >
                        {line}
                      </motion.div>
                    ))}
                  </AnimatePresence>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        <div
          className="flex-1 flex overflow-hidden"
          style={{ minHeight: 0 }}
        >
          <div
            className="flex flex-col"
            style={{
              width: "45%",
              minWidth: "360px",
              borderRight: "1px solid #0a1e2c",
              overflow: "hidden",
            }}
          >
            <LeftPanel
              activeModels={isAnalyzed ? activeModels : []}
              imageUrl={imageUrl}
              videoUrl={videoUrl}
              mediaType={mediaType}
              isAnalyzing={analyzingModels.size > 0}
              onLoadMedia={handleLoadMedia}
                onVideoTimeUpdate={setCurrentVideoTimeMs}
            />
          </div>

          <div
            className="flex flex-col flex-1"
            style={{ overflow: "hidden", minWidth: 0 }}
          >
            <RightPanel
              activeModels={activeModels}
              setActiveModels={setActiveModels}
              isAnalyzed={isAnalyzed}
              analyzingModels={analyzingModels}
              onAnalyze={handleAnalyzeSingle}
              canAnalyze={selectedFile !== null}
              imageUrl={mediaType === "image" ? imageUrl : null}
                currentVideoTimeMs={mediaType === "video" ? currentVideoTimeMs : null}
            />
          </div>
        </div>

        <EnsemblePanel activeModels={activeModels} isAnalyzed={isAnalyzed} ensembleResult={ensembleResult} />

        <div
          className="pointer-events-none fixed inset-0"
          style={{
            background:
              "repeating-linear-gradient(0deg, transparent, transparent 3px, rgba(0,0,0,0.04) 3px, rgba(0,0,0,0.04) 4px)",
            zIndex: 9999,
          }}
        />
      </div>
    </ErrorBoundary>
  );
}

export default App;

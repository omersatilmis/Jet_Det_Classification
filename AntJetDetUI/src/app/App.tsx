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
  const [analyzingModels, setAnalyzingModels] = useState<Set<string>>(new Set());
  const [isAnalyzed, setIsAnalyzed] = useState(false); // Can technically keep global to signal "at least one run"
  const [activeModels, setActiveModels] = useState([
    MODEL_DATA["cascade-rcnn"]
  ]);
  const [analysisProgress, setAnalysisProgress] = useState<Record<string, number>>({});
  const [logLines, setLogLines] = useState<string[]>([]);

  // Cleanup object URLs to prevent memory leaks
  useEffect(() => {
    return () => {
      if (imageUrl) URL.revokeObjectURL(imageUrl);
    };
  }, [imageUrl]);

  const handleLoadImage = () => {
    // Open a file picker
    const input = document.createElement("input");
    input.type = "file";
    input.accept = "image/*";
    input.onchange = (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (file) {
        if (imageUrl) URL.revokeObjectURL(imageUrl);
        setSelectedFile(file);
        setImageUrl(URL.createObjectURL(file));

        // Reset states
        setIsAnalyzed(false);
        setAnalyzingModels(new Set());
        setAnalysisProgress({});
        setActiveModels(prev => prev.map(m => ({
          ...m,
          detections: [],
          inferenceTime: 0,
          fps: 0,
          vramUsage: 0,
          gpuUsage: 0
        })));
        setLogLines([`[SYS] Görsel seçildi: ${file.name}`]);
      }
    };
    input.click();
  };

  const handleAnalyzeSingle = async (modelId: string) => {
    if (!selectedFile) return;

    setAnalyzingModels(prev => new Set(prev).add(modelId));
    setAnalysisProgress(prev => ({ ...prev, [modelId]: 10 }));
    setLogLines((prev) => [...prev, `[SYS] ${modelId} modeli başlatılıyor...`]);

    try {
      const { analyzeSingleImage } = await import('../api');

      // We simulate progress for the UI while waiting on the real fetch
      const progressInterval = setInterval(() => {
        setAnalysisProgress(prev => ({
          ...prev,
          [modelId]: Math.min((prev[modelId] || 10) + 15, 85)
        }));
      }, 200);

      const result = await analyzeSingleImage(modelId, selectedFile);
      clearInterval(progressInterval);
      setAnalysisProgress(prev => ({ ...prev, [modelId]: 100 }));

      // Map backend `result.models[0]` back to the UI `activeModels`
      const updatedModels = activeModels.map((baseModel) => {
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
          detections: backendData.detections.map((d: import('../api').Detection, i: number) => ({
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
      });

      setActiveModels(updatedModels);

      setLogLines((prev) => [
        ...prev,
        `[SYS] ${modelId} Analizi Tamamlandı. ${result.models[0]?.detections.length || 0} nesne bulundu.`
      ]);

      setIsAnalyzed(true);

    } catch (error) {
      console.error(`Analysis Failed for ${modelId}:`, error);
      setLogLines((prev) => [...prev, `[ERROR] ${modelId} analizi başarısız: ${error}`]);
      setAnalysisProgress(prev => ({ ...prev, [modelId]: 0 }));
    } finally {
      setAnalyzingModels(prev => {
        const next = new Set(prev);
        next.delete(modelId);
        return next;
      });
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    if (imageUrl) URL.revokeObjectURL(imageUrl);
    setImageUrl(null);

    setAnalyzingModels(new Set());
    setIsAnalyzed(false);
    setAnalysisProgress({});
    setLogLines([]);
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
        {/* Header */}
        <SystemHeader />

        {/* Analysis control bar */}
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
                {/* Removed global analyze button */}

                {/* Reset */}
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

                {/* Progress bar */}
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

                {/* Log stream */}
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

        {/* Main content */}
        <div
          className="flex-1 flex overflow-hidden"
          style={{ minHeight: 0 }}
        >
          {/* Left panel */}
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
              isAnalyzing={analyzingModels.size > 0}
              onLoadImage={handleLoadImage}
            />
          </div>

          {/* Right panel */}
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
              imageUrl={imageUrl}
            />
          </div>
        </div>

        {/* Ensemble bottom panel */}
        <EnsemblePanel activeModels={activeModels} isAnalyzed={isAnalyzed} />

        {/* Global scanlines overlay */}
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

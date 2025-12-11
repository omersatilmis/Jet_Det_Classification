import React, { useState, useEffect } from "react";
import { ChevronDown, ChevronUp, Plus, X, Layers, Trash2 } from "lucide-react";
import { AnimatePresence, motion } from "motion/react";
import { MODEL_OPTIONS, MODEL_DATA, ModelData } from "./hud-data";
import { ModelCard } from "./ModelCard";
import { fetchModels, deleteModel } from "../../api";

interface RightPanelProps {
  activeModels: ModelData[];
  setActiveModels: (models: ModelData[]) => void;
  isAnalyzed: boolean;
  analyzingModels: Set<string>;
  onAnalyze: (modelId: string) => void;
  canAnalyze: boolean;
  imageUrl: string | null;
}

export function RightPanel({
  activeModels,
  setActiveModels,
  isAnalyzed,
  analyzingModels,
  onAnalyze,
  canAnalyze,
  imageUrl,
}: RightPanelProps) {
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const [dynamicOptions, setDynamicOptions] = useState(MODEL_OPTIONS);
  const [dynamicModelData, setDynamicModelData] = useState<Record<string, ModelData>>(MODEL_DATA);

  // Custom Model Upload Modal State
  const [uploadModalOpen, setUploadModalOpen] = useState(false);
  const [uploadName, setUploadName] = useState("");
  const [uploadArchitecture, setUploadArchitecture] = useState("cascade_rcnn_r50_tiny");
  const [uploadColor, setUploadColor] = useState("#00FF41");
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);

  const loadBackendModels = async () => {
    try {
      const res = await fetchModels();
      const customModels = res.models;

      const newOptions = [...MODEL_OPTIONS];
      const newData = { ...MODEL_DATA };

      customModels.forEach((cm: any) => {
        if (!newOptions.find(o => o.id === cm.id)) {
          newOptions.push({ id: cm.id, label: cm.name });
          newData[cm.id] = {
            id: cm.id,
            name: cm.name,
            shortName: cm.shortName || cm.name.substring(0, 8),
            color: cm.color || "#FFD700",
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
          };
        }
      });
      setDynamicOptions(newOptions);
      setDynamicModelData(newData);
    } catch (e) {
      console.error("Failed to load backend models", e);
    }
  };

  useEffect(() => {
    loadBackendModels();
  }, []);

  const addModel = (id: string) => {
    if (!activeModels.find((m) => m.id === id)) {
      setActiveModels([...activeModels, dynamicModelData[id]]);
    }
    setDropdownOpen(false);
  };

  const removeModel = (id: string) => {
    setActiveModels(activeModels.filter((m) => m.id !== id));
  };

  const availableOptions = dynamicOptions.filter(
    (opt) => !activeModels.find((m) => m.id === opt.id)
  );

  return (
    <div
      className="flex flex-col h-full overflow-hidden"
      style={{ fontFamily: "'Share Tech Mono', monospace" }}
    >
      {/* Panel header */}
      <div
        className="flex items-center justify-between px-4 py-2"
        style={{ borderBottom: "1px solid #0d2030" }}
      >
        <div className="flex items-center gap-2">
          <Layers size={14} style={{ color: "#00FF41" }} />
          <span
            style={{
              color: "#00FF41",
              fontSize: "11px",
              letterSpacing: "0.2em",
            }}
          >
            MODEL YÖNETİM PANELI
          </span>
        </div>
        <div
          className="px-2 py-0.5 rounded"
          style={{
            background: "rgba(0,255,65,0.08)",
            border: "1px solid rgba(0,255,65,0.2)",
            color: "#00FF41",
            fontSize: "9px",
            letterSpacing: "0.15em",
          }}
        >
          {activeModels.length} / 3 AKTİF
        </div>
      </div>

      {/* Scrollable content */}
      <div className="flex-1 overflow-y-auto p-3 space-y-3">
        {/* Model Selector Dropdown */}
        <div className="relative">
          <button
            className="w-full flex items-center justify-between px-4 py-3 rounded transition-all duration-200"
            style={{
              background: dropdownOpen
                ? "rgba(0,229,255,0.08)"
                : "rgba(0,229,255,0.04)",
              border: `1px solid ${dropdownOpen ? "#00E5FF40" : "#0d2a3a"}`,
              color: "#00E5FF",
              fontSize: "11px",
              letterSpacing: "0.2em",
            }}
            onClick={() => setDropdownOpen((v) => !v)}
          >
            <div className="flex items-center gap-2">
              <Plus size={13} />
              <span>MODEL SEÇİM BİRİMİ</span>
            </div>
            {dropdownOpen ? (
              <ChevronUp size={14} />
            ) : (
              <ChevronDown size={14} />
            )}
          </button>

          <AnimatePresence>
            {dropdownOpen && (
              <motion.div
                initial={{ opacity: 0, y: -4, scaleY: 0.9 }}
                animate={{ opacity: 1, y: 0, scaleY: 1 }}
                exit={{ opacity: 0, y: -4, scaleY: 0.9 }}
                transition={{ duration: 0.15 }}
                className="absolute inset-x-0 z-10 rounded overflow-hidden"
                style={{
                  top: "calc(100% + 4px)",
                  background: "#060d18",
                  border: "1px solid #0d3a4a",
                  boxShadow: "0 8px 32px rgba(0,0,0,0.8)",
                }}
              >
                {dynamicOptions.map((opt) => {
                  const already = !!activeModels.find((m) => m.id === opt.id);
                  const modelData = dynamicModelData[opt.id];
                  return (
                    <button
                      key={opt.id}
                      disabled={already}
                      onClick={() => addModel(opt.id)}
                      className="w-full flex items-center justify-between px-4 py-3 transition-all duration-150"
                      style={{
                        color: already ? "#1a3a4a" : "#e0f4ff",
                        fontSize: "11px",
                        letterSpacing: "0.1em",
                        cursor: already ? "not-allowed" : "pointer",
                        borderBottom: "1px solid #0a1a28",
                      }}
                      onMouseEnter={(e) => {
                        if (!already)
                          (e.currentTarget as HTMLElement).style.background =
                            "rgba(0,229,255,0.06)";
                      }}
                      onMouseLeave={(e) => {
                        (e.currentTarget as HTMLElement).style.background =
                          "transparent";
                      }}
                    >
                      <div className="flex items-center gap-3">
                        <div
                          className="w-2 h-2 rounded-full"
                          style={{
                            background: already ? "#1a3a4a" : modelData.color,
                            boxShadow: already
                              ? "none"
                              : `0 0 5px ${modelData.color}`,
                          }}
                        />
                        <span>{opt.label}</span>
                      </div>

                      <div className="flex items-center gap-2">
                        {already ? (
                          <span
                            style={{ color: "#1a3a4a", fontSize: "9px", letterSpacing: "0.2em" }}
                          >
                            AKTİF
                          </span>
                        ) : (
                          <Plus size={11} style={{ color: "#3a5a7a" }} />
                        )}

                        {!opt.id.startsWith("cascade") && (
                          <button
                            onClick={async (e) => {
                              e.stopPropagation();
                              if (confirm(`"${opt.label}" modelini kalıcı olarak silmek istediğinize emin misiniz?`)) {
                                try {
                                  await deleteModel(opt.id);
                                  await loadBackendModels();
                                  if (already) {
                                    removeModel(opt.id);
                                  }
                                } catch (err) {
                                  alert("Model silinemedi: " + err);
                                }
                              }
                            }}
                            className="p-1 rounded opacity-50 hover:opacity-100 transition-opacity ml-2"
                            style={{ background: "rgba(255,34,68,0.1)", color: "#FF2244" }}
                            title="Modeli Sil"
                          >
                            <Trash2 size={11} />
                          </button>
                        )}
                      </div>
                    </button>
                  );
                })}
                {availableOptions.length === 0 && (
                  <div
                    className="px-4 py-3 text-center"
                    style={{ color: "#1a3a4a", fontSize: "10px", letterSpacing: "0.2em" }}
                  >
                    TÜM MODELLER AKTİF
                  </div>
                )}

                {/* Model Ekle Button */}
                <div
                  className="px-4 py-3 border-t"
                  style={{ borderColor: "#0a1a28" }}
                >
                  <button
                    className="w-full flex items-center justify-center gap-2 py-2 rounded transition-all duration-200"
                    style={{
                      background: "rgba(0,255,65,0.06)",
                      border: "1px dashed rgba(0,255,65,0.3)",
                      color: "#00FF41",
                      fontSize: "10px",
                      letterSpacing: "0.15em",
                    }}
                    onClick={() => {
                      setUploadModalOpen(true);
                      setDropdownOpen(false);
                    }}
                  >
                    <Plus size={12} />
                    <span>YENİ MODEL EKLE (.PTH)</span>
                  </button>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Custom Upload Modal */}
        <AnimatePresence>
          {uploadModalOpen && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 z-50 flex items-center justify-center"
              style={{ background: "rgba(0,0,0,0.8)", backdropFilter: "blur(4px)" }}
            >
              <motion.div
                initial={{ scale: 0.95, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                exit={{ scale: 0.95, opacity: 0 }}
                className="w-96 rounded border p-6 flex flex-col gap-4"
                style={{
                  background: "#060d18",
                  borderColor: "#0d3a4a",
                  boxShadow: "0 0 40px rgba(0,229,255,0.1)",
                }}
              >
                <div className="flex justify-between items-center mb-2">
                  <h3 style={{ color: "#00E5FF", fontSize: "14px", letterSpacing: "0.1em" }}>
                    YENİ MODEL SİSTEME YÜKLE
                  </h3>
                  <button onClick={() => setUploadModalOpen(false)} style={{ color: "#3a5a7a" }}>
                    <X size={16} />
                  </button>
                </div>

                <div className="flex flex-col gap-1">
                  <label style={{ color: "#8ab4f8", fontSize: "10px" }}>MODEL İSMİ</label>
                  <input
                    type="text"
                    value={uploadName}
                    onChange={(e) => setUploadName(e.target.value)}
                    placeholder="Örn: Benim YOLO Modelim"
                    className="px-3 py-2 rounded outline-none"
                    style={{ background: "#0a1a28", border: "1px solid #1a3a4a", color: "#e0f4ff", fontSize: "12px" }}
                  />
                </div>

                <div className="flex flex-col gap-1">
                  <label style={{ color: "#8ab4f8", fontSize: "10px" }}>MİMARİ ALTYAPI</label>
                  <select
                    value={uploadArchitecture}
                    onChange={(e) => setUploadArchitecture(e.target.value)}
                    className="px-3 py-2 rounded outline-none w-full"
                    style={{ background: "#0a1a28", border: "1px solid #1a3a4a", color: "#e0f4ff", fontSize: "12px" }}
                  >
                    <option value="cascade_rcnn_r50_tiny">Cascade R-CNN R50 (Varsayılan)</option>
                    <option value="cascade_rcnn_convnext_tiny">Cascade ConvNeXt-Tiny</option>
                    <option value="yolov8_s_jet">YOLOv8-Small</option>
                  </select>
                </div>

                <div className="flex flex-col gap-1">
                  <label style={{ color: "#8ab4f8", fontSize: "10px" }}>AĞIRLIK DOSYASI (.pth)</label>
                  <label
                    className="flex flex-col items-center justify-center p-4 border-2 border-dashed rounded cursor-pointer transition-colors"
                    style={{
                      borderColor: uploadFile ? "#00FF41" : "#1a3a4a",
                      background: uploadFile ? "rgba(0, 255, 65, 0.05)" : "#0a1a28",
                    }}
                  >
                    <div className="flex items-center gap-2">
                      <Plus size={16} color={uploadFile ? "#00FF41" : "#3a5a7a"} />
                      <span style={{ color: uploadFile ? "#00FF41" : "#8ab4f8", fontSize: "12px", fontWeight: "bold" }}>
                        {uploadFile ? uploadFile.name : "Tıkla ve .pth Dosyasını Seç"}
                      </span>
                    </div>
                    <input
                      type="file"
                      accept=".pth"
                      onChange={(e) => setUploadFile(e.target.files?.[0] || null)}
                      className="hidden"
                    />
                  </label>
                </div>

                <div className="flex flex-col gap-1">
                  <label style={{ color: "#8ab4f8", fontSize: "10px" }}>ARAYÜZ RENGİ</label>
                  <div className="flex gap-2">
                    {["#FFD700", "#FF3366", "#00E5FF", "#BB86FC", "#00FF41"].map(color => (
                      <div
                        key={color}
                        onClick={() => setUploadColor(color)}
                        className="w-6 h-6 rounded-full cursor-pointer transition-transform"
                        style={{
                          background: color,
                          border: uploadColor === color ? "2px solid #fff" : "2px solid transparent",
                          transform: uploadColor === color ? "scale(1.1)" : "scale(1)"
                        }}
                      />
                    ))}
                  </div>
                </div>

                <button
                  disabled={uploading || !uploadName || !uploadFile}
                  onClick={async () => {
                    if (!uploadName || !uploadFile) return;
                    setUploading(true);
                    try {
                      const { uploadModel } = await import('../../api');
                      await uploadModel(uploadName, uploadArchitecture, uploadColor, uploadFile);
                      alert(`${uploadName} başarıyla yüklendi!`);
                      setUploadModalOpen(false);
                      setUploadName("");
                      setUploadFile(null);
                      await loadBackendModels();
                    } catch (err) {
                      alert("Yükleme başarısız: " + err);
                    } finally {
                      setUploading(false);
                    }
                  }}
                  className="mt-4 w-full py-2 rounded font-bold transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                  style={{
                    background: uploadName && uploadFile ? "#00FF41" : "#1a3a4a",
                    color: uploadName && uploadFile ? "#000" : "#3a5a7a",
                    fontSize: "12px",
                    letterSpacing: "0.1em"
                  }}
                >
                  {uploading ? "YÜKLENİYOR..." : "SİSTEME KAYDET"}
                </button>
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Active model cards */}
        <AnimatePresence>
          {activeModels.length === 0 ? (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="text-center py-10"
              style={{ color: "#1a3a4a", fontSize: "10px", letterSpacing: "0.25em" }}
            >
              — HENÜZ MODEL SEÇİLMEDİ —
              <br />
              <span style={{ fontSize: "8px", color: "#0d2030" }}>
                YUKARIDAN MODEL EKLEYIN
              </span>
            </motion.div>
          ) : (
            activeModels.map((model, idx) => (
              <motion.div
                key={model.id}
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                transition={{ duration: 0.25, delay: idx * 0.05 }}
              >
                <div className="relative group">
                  <ModelCard
                    model={model}
                    isAnalyzed={isAnalyzed || model.detections.length > 0}
                    defaultOpen={idx === 0}
                    isAnalyzing={analyzingModels.has(model.id)}
                    canAnalyze={canAnalyze && !analyzingModels.has(model.id)}
                    onAnalyze={() => onAnalyze(model.id)}
                    imageUrl={imageUrl}
                    onRemove={() => removeModel(model.id)}
                  />
                </div>
              </motion.div>
            ))
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}

import React from "react";
import { motion } from "motion/react";
import { Target } from "lucide-react";
import { ModelData } from "../../hud-data";
import { SectionLabel, DataRow, NoDataMsg } from "../shared";

export function HudTab({ model, isAnalyzed }: { model: ModelData; isAnalyzed: boolean }) {
    const det = model.detections[0];
    return (
        <div
            className="space-y-3"
            style={{ fontFamily: "'Share Tech Mono', monospace" }}
        >
            <SectionLabel color={model.color}>HUD KATMANI — VISUAL OVERLAY</SectionLabel>
            {isAnalyzed ? (
                det ? (
                    <>
                        <div
                            className="p-3 rounded space-y-2"
                            style={{
                                background: "#020810",
                                border: `1px solid ${model.color}20`,
                            }}
                        >
                            <DataRow label="TARGET ID" value={det.targetId} color={model.color} />

                            <div className="grid grid-cols-2 gap-4 py-1">
                                <div className="space-y-1">
                                    <div style={{ color: "#2a4a6a", fontSize: "8px" }}>AZIMUTH</div>
                                    <div style={{ color: model.color, fontSize: "14px", fontWeight: "bold" }}>
                                        {det.azimuth ?? "0.00"}°
                                    </div>
                                </div>
                                <div className="space-y-1">
                                    <div style={{ color: "#2a4a6a", fontSize: "8px" }}>ELEVATION</div>
                                    <div style={{ color: model.color, fontSize: "14px", fontWeight: "bold" }}>
                                        {det.elevation ?? "0.00"}°
                                    </div>
                                </div>
                            </div>

                            <div className="flex items-center justify-between pt-1" style={{ borderTop: "1px solid #0d2030" }}>
                                <div style={{ color: "#2a4a6a", fontSize: "9px" }}>TARGET RANGE</div>
                                <div style={{ color: "#00E5FF", fontSize: "12px", letterSpacing: "0.1em" }}>
                                    {det.distance_km?.toFixed(2) ?? "---"} KM
                                </div>
                            </div>

                            <DataRow
                                label="P_conf (Güven Skoru)"
                                value={`${(det.confidence * 100).toFixed(2)}%`}
                                color={det.confidence > 0.9 ? "#00FF41" : det.confidence > 0.7 ? "#FF8C00" : "#FF2244"}
                                highlight
                            />

                            <div className="flex items-center justify-between">
                                <span style={{ color: "#2a4a6a", fontSize: "9px" }}>THREAT LEVEL</span>
                                <span style={{
                                    color: det.confidence > 0.8 ? "#FF2244" : "#FF8C00",
                                    fontSize: "10px",
                                    fontWeight: "bold",
                                    textShadow: "0 0 8px rgba(255,34,68,0.4)"
                                }}>
                                    {det.confidence > 0.8 ? "CRITICAL / HOSTILE" : "EVALUATING"}
                                </span>
                            </div>
                        </div>

                        <motion.div
                            className="flex items-center gap-3 p-3 rounded"
                            style={{
                                background: `${model.color}10`,
                                border: `1px solid ${model.color}30`,
                            }}
                            animate={{
                                borderColor: [`${model.color}30`, `${model.color}70`, `${model.color}30`],
                            }}
                            transition={{ duration: 1.5, repeat: Infinity }}
                        >
                            <Target size={16} style={{ color: model.color }} />
                            <div>
                                <div style={{ color: model.color, fontSize: "10px", letterSpacing: "0.2em" }}>
                                    TARGET LOCK — ACQUIRED
                                </div>
                                <div style={{ color: "#3a6a5a", fontSize: "9px" }}>
                                    Tracking Active · Quality: {det.confidence > 0.9 ? "OPTIMAL" : "STABLE"}
                                </div>
                            </div>
                        </motion.div>
                    </>
                ) : (
                    <div className="text-center py-4" style={{ color: "#3a5a7a", fontSize: "10px", letterSpacing: "0.2em" }}>
                        — TESPİT EDİLEN NESNE YOK —
                    </div>
                )
            ) : (
                <NoDataMsg />
            )}
        </div>
    );
}

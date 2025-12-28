import React from "react";
import { motion } from "motion/react";
import { ModelData } from "../../hud-data";
import { SectionLabel, NoDataMsg, JET_THUMB } from "../shared";

export function ObjectsTab({ model, isAnalyzed }: { model: ModelData; isAnalyzed: boolean }) {
    return (
        <div
            className="space-y-3"
            style={{ fontFamily: "'Share Tech Mono', monospace" }}
        >
            <SectionLabel color={model.color}>NESNE SINIFLANDIRMA</SectionLabel>
            {isAnalyzed ? (
                <>
                    {model.detections.map((det, i) => (
                        <div key={i}>
                            <div
                                className="flex items-center justify-between mb-1"
                                style={{ color: "#3a5a7a", fontSize: "9px" }}
                            >
                                <span style={{ color: i === 0 ? model.color : "#3a5a7a" }}>
                                    {det.label.toUpperCase()}
                                </span>
                                <span
                                    style={{
                                        color:
                                            det.confidence > 0.9
                                                ? "#00FF41"
                                                : det.confidence > 0.5
                                                    ? "#FF8C00"
                                                    : "#FF2244",
                                    }}
                                >
                                    {(det.confidence * 100).toFixed(1)}%
                                </span>
                            </div>
                            <div
                                className="h-1.5 rounded-full overflow-hidden"
                                style={{ background: "#0a1a2a" }}
                            >
                                <motion.div
                                    className="h-full rounded-full"
                                    style={{
                                        background:
                                            det.confidence > 0.9
                                                ? "#00FF41"
                                                : det.confidence > 0.5
                                                    ? "#FF8C00"
                                                    : "#FF2244",
                                    }}
                                    initial={{ width: 0 }}
                                    animate={{ width: `${det.confidence * 100}%` }}
                                    transition={{ duration: 0.8, delay: i * 0.1 }}
                                />
                            </div>
                        </div>
                    ))}

                    {/* Snapshot gallery */}
                    <div>
                        <div
                            style={{ color: "#3a5a7a", fontSize: "9px", letterSpacing: "0.2em", marginBottom: "8px" }}
                        >
                            SNAPSHOT GALERİSİ
                        </div>
                        <div className="flex gap-2">
                            {[1, 2, 3].map((i) => (
                                <div
                                    key={i}
                                    className="relative rounded overflow-hidden"
                                    style={{
                                        width: "60px",
                                        height: "42px",
                                        border: `1px solid ${model.color}30`,
                                    }}
                                >
                                    <img
                                        src={JET_THUMB}
                                        alt={`snapshot-${i}`}
                                        className="w-full h-full object-cover"
                                        style={{
                                            filter: `brightness(0.7) hue-rotate(${i * 20}deg)`,
                                        }}
                                    />
                                    <div
                                        className="absolute bottom-0 inset-x-0 text-center"
                                        style={{
                                            background: `${model.color}30`,
                                            fontSize: "7px",
                                            color: model.color,
                                            padding: "1px 0",
                                        }}
                                    >
                                        #{String(i).padStart(3, "0")}
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                </>
            ) : (
                <NoDataMsg />
            )}
        </div>
    );
}

import React from "react";
import { motion } from "motion/react";
import { ModelData } from "../../hud-data";
import { SectionLabel, MetricBox, NoDataMsg } from "../shared";

export function PerfTab({ model, isAnalyzed }: { model: ModelData; isAnalyzed: boolean }) {
    return (
        <div
            className="space-y-3"
            style={{ fontFamily: "'Share Tech Mono', monospace" }}
        >
            <SectionLabel color={model.color}>PERFORMANS METRİKLERİ</SectionLabel>
            {isAnalyzed ? (
                <>
                    <div className="grid grid-cols-2 gap-2">
                        <MetricBox
                            label="t_inf"
                            value={`${model.inferenceTime} ms`}
                            sub="Çıkarım Süresi"
                            color={model.color}
                        />
                        <MetricBox
                            label="FPS"
                            value={model.fps.toFixed(1)}
                            sub="Real-time"
                            color={model.color}
                        />
                        <MetricBox
                            label="GPU USAGE"
                            value={`${model.gpuUsage}%`}
                            sub="RTX 2060"
                            color={model.gpuUsage > 80 ? "#FF2244" : model.gpuUsage > 60 ? "#FF8C00" : "#00FF41"}
                        />
                        <MetricBox
                            label="VRAM"
                            value={`${model.vramUsage}/${model.vramTotal} GB`}
                            sub="Video Memory"
                            color={model.color}
                        />
                    </div>

                    {/* GPU bar */}
                    <div>
                        <div
                            className="flex justify-between mb-1"
                            style={{ color: "#3a5a7a", fontSize: "9px", letterSpacing: "0.15em" }}
                        >
                            <span>GPU LOAD — RTX 2060</span>
                            <span style={{ color: model.color }}>{model.gpuUsage}%</span>
                        </div>
                        <div
                            className="h-2 rounded-full overflow-hidden"
                            style={{ background: "#0a1a2a" }}
                        >
                            <motion.div
                                className="h-full rounded-full"
                                style={{
                                    background: `linear-gradient(90deg, ${model.color}80, ${model.color})`,
                                    boxShadow: `0 0 8px ${model.color}60`,
                                }}
                                initial={{ width: 0 }}
                                animate={{ width: `${model.gpuUsage}%` }}
                                transition={{ duration: 1, ease: "easeOut" }}
                            />
                        </div>
                    </div>

                    {/* VRAM bar */}
                    <div>
                        <div
                            className="flex justify-between mb-1"
                            style={{ color: "#3a5a7a", fontSize: "9px", letterSpacing: "0.15em" }}
                        >
                            <span>VRAM ALLOCATION</span>
                            <span style={{ color: model.color }}>
                                {model.vramUsage}/{model.vramTotal} GB
                            </span>
                        </div>
                        <div
                            className="h-2 rounded-full overflow-hidden"
                            style={{ background: "#0a1a2a" }}
                        >
                            <motion.div
                                className="h-full rounded-full"
                                style={{
                                    background: `linear-gradient(90deg, #00E5FF60, #00E5FF)`,
                                    boxShadow: `0 0 8px #00E5FF60`,
                                }}
                                initial={{ width: 0 }}
                                animate={{
                                    width: `${(model.vramUsage / model.vramTotal) * 100}%`,
                                }}
                                transition={{ duration: 1, ease: "easeOut" }}
                            />
                        </div>
                    </div>
                </>
            ) : (
                <NoDataMsg />
            )}
        </div>
    );
}

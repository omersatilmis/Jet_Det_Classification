import React from "react";
import { motion } from "motion/react";
import { ModelData } from "../../hud-data";
import { SectionLabel, NoDataMsg } from "../shared";

export function OutputTab({
    model,
    imageUrl,
}: {
    model: ModelData;
    imageUrl?: string | null;
}) {
    const displayUrl = model.visualizedImage || imageUrl;

    return (
        <div
            className="space-y-3"
            style={{ fontFamily: "'Share Tech Mono', monospace" }}
        >
            <SectionLabel color={model.color}>GÖRSEL ÇIKTISI</SectionLabel>

            {!displayUrl ? (
                <NoDataMsg />
            ) : (
                <div
                    className="relative w-full rounded overflow-hidden"
                    style={{
                        border: `1px solid ${model.color}40`,
                        aspectRatio: "16/9",
                        background: "#020810",
                    }}
                >
                    <img
                        src={displayUrl}
                        alt="Model Output"
                        className="w-full h-full object-contain opacity-100" // Opacity 100 for clear visualization
                    />

                    {/* Only draw CSS bounding boxes if we don't have the pre-rendered OpenCV image from backend */}
                    {!model.visualizedImage && model.detections.map((det, idx) => {
                        const [x, y, w, h] = det.bbox;
                        return (
                            <motion.div
                                key={idx}
                                className="absolute"
                                initial={{ opacity: 0, scale: 1.1 }}
                                animate={{ opacity: 1, scale: 1 }}
                                transition={{ duration: 0.3, delay: idx * 0.1 }}
                                style={{
                                    left: `${x * 100}%`,
                                    top: `${y * 100}%`,
                                    width: `${w * 100}%`,
                                    height: `${h * 100}%`,
                                    border: `1.5px solid ${model.color}`,
                                    boxShadow: `inset 0 0 10px ${model.color}40, 0 0 10px ${model.color}40`,
                                    backgroundColor: `${model.color}10`,
                                }}
                            >
                                {/* Crosshairs on corners */}
                                <div
                                    className="absolute -top-1 -left-1 w-2 h-2"
                                    style={{
                                        borderTop: `2px solid ${model.color}`,
                                        borderLeft: `2px solid ${model.color}`,
                                    }}
                                />
                                <div
                                    className="absolute -top-1 -right-1 w-2 h-2"
                                    style={{
                                        borderTop: `2px solid ${model.color}`,
                                        borderRight: `2px solid ${model.color}`,
                                    }}
                                />
                                <div
                                    className="absolute -bottom-1 -left-1 w-2 h-2"
                                    style={{
                                        borderBottom: `2px solid ${model.color}`,
                                        borderLeft: `2px solid ${model.color}`,
                                    }}
                                />
                                <div
                                    className="absolute -bottom-1 -right-1 w-2 h-2"
                                    style={{
                                        borderBottom: `2px solid ${model.color}`,
                                        borderRight: `2px solid ${model.color}`,
                                    }}
                                />

                                {/* Label */}
                                <div
                                    className="absolute -top-4 left-0 px-1 whitespace-nowrap"
                                    style={{
                                        background: `${model.color}90`,
                                        color: "#050c14",
                                        fontSize: "7px",
                                        fontWeight: "bold",
                                        letterSpacing: "0.1em",
                                    }}
                                >
                                    {det.label} {Math.round(det.confidence * 100)}%
                                </div>
                            </motion.div>
                        );
                    })}

                    {/* Grid overlay inside image */}
                    <div
                        className="absolute inset-0 pointer-events-none"
                        style={{
                            backgroundImage: `
                linear-gradient(${model.color}10 1px, transparent 1px),
                linear-gradient(90deg, ${model.color}10 1px, transparent 1px)
              `,
                            backgroundSize: "20px 20px",
                        }}
                    />
                </div>
            )}
        </div>
    );
}

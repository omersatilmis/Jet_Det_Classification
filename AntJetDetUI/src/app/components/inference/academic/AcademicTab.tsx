import React from "react";
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    Tooltip,
    ResponsiveContainer,
    Area,
    AreaChart,
} from "recharts";
import { ModelData } from "../../hud-data";
import { SectionLabel, NoDataMsg } from "../shared";

export function AcademicTab({ model, isAnalyzed }: { model: ModelData; isAnalyzed: boolean }) {
    const hasPr = (model.prCurve && model.prCurve.length > 0) || false;
    const map5095 = model.mAP || 0;
    const ap50 = model.ioU || 0;

    return (
        <div
            className="space-y-3"
            style={{ fontFamily: "'Share Tech Mono', monospace" }}
        >
            <SectionLabel color={model.color}>AKADEMİK KANIT ALANI</SectionLabel>
            {isAnalyzed ? (
                <>
                    <div
                        className="flex items-center justify-between p-3 rounded"
                        style={{ background: "#020810", border: `1px solid ${model.color}20` }}
                    >
                        <div>
                            <div style={{ color: "#3a5a7a", fontSize: "9px", letterSpacing: "0.2em" }}>
                                mAP@0.5 SKORU
                            </div>
                            <div
                                style={{
                                    color: model.color,
                                    fontSize: "22px",
                                    fontFamily: "'Orbitron', sans-serif",
                                    fontWeight: 700,
                                }}
                            >
                                <>
                                    <div
                                        className="grid grid-cols-2 gap-3 p-3 rounded"
                                        style={{ background: "#020810", border: `1px solid ${model.color}20` }}
                                    >
                                        <div>
                                            <div style={{ color: "#3a5a7a", fontSize: "9px", letterSpacing: "0.2em" }}>
                                                mAP@0.5:0.95
                                            </div>
                                            <div
                                                style={{
                                                    color: model.color,
                                                    fontSize: "22px",
                                                    fontFamily: "'Orbitron', sans-serif",
                                                    fontWeight: 700,
                                                }}
                                            >
                                                {map5095 ? `${map5095.toFixed(1)}%` : "—"}
                                            </div>
                                            <div style={{ color: "#3a6a5a", fontSize: "9px" }}>
                                                COCO mAP (AP50-95)
                                            </div>
                                        </div>
                                        <div>
                                            <div style={{ color: "#3a5a7a", fontSize: "9px", letterSpacing: "0.2em" }}>
                                                AP50 (IoU=0.50)
                                            </div>
                                            <div
                                                style={{
                                                    color: model.color,
                                                    fontSize: "22px",
                                                    fontFamily: "'Orbitron', sans-serif",
                                                    fontWeight: 700,
                                                }}
                                            >
                                                {ap50 ? `${ap50.toFixed(1)}%` : "—"}
                                            </div>
                                            <div style={{ color: "#3a6a5a", fontSize: "9px" }}>
                                                AP@0.50
                                            </div>
                                        </div>
                                    </div>
                            <div
                                className="relative rounded overflow-hidden"
                                style={{
                                    border: `1px solid ${model.color}40`,
                                    background: "#000",
                                    aspectRatio: "16/9"
                                }}
                            >
                                <img
                                    src={model.heatmapImage}
                                    alt="XAI Heatmap"
                                    className="w-full h-full object-contain"
                                />
                            </div>
                        </div>
                    )}

                    {/* PR Curve */}
                    <div>
                        <div
                            style={{
                                color: "#3a5a7a",
                                fontSize: "9px",
                                letterSpacing: "0.2em",
                                marginBottom: "8px",
                            }}
                        >
                            PRECISION-RECALL EĞRİSİ
                        </div>
                        <ResponsiveContainer width="100%" height={120}>
                            <AreaChart
                                data={model.prCurve || []}
                                margin={{ top: 2, right: 4, left: -20, bottom: 2 }}
                            >
                                <defs>
                                    <linearGradient id={`pr-${model.id}`} x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor={model.color} stopOpacity={0.3} />
                                        <stop offset="95%" stopColor={model.color} stopOpacity={0.02} />
                                    </linearGradient>
                                </defs>
                                <XAxis
                                    dataKey="recall"
                                    tick={{ fill: "#2a4a6a", fontSize: 8, fontFamily: "Share Tech Mono" }}
                                                <div className="mt-4">
                                                    <div
                                                        style={{
                                                            color: "#3a5a7a",
                                                            fontSize: "9px",
                                                            letterSpacing: "0.2em",
                                                            marginBottom: "8px",
                                                        }}
                                                    >
                                                        XAI / GRAD-CAM ISI HARİTASI
                                                    </div>
                                                    <NoDataMsg />
                                                </div>
                                    tickFormatter={(v) => v.toFixed(1)}
                                />
                                <YAxis
                                    tick={{ fill: "#2a4a6a", fontSize: 8, fontFamily: "Share Tech Mono" }}
                                    tickFormatter={(v) => v.toFixed(1)}
                                    domain={[0, 1]}
                                />
                                <Tooltip
                                    contentStyle={{
                                        background: "#050c14",
                                        border: `1px solid ${model.color}40`,
                                        fontSize: "9px",
                                        fontFamily: "Share Tech Mono",
                                        color: model.color,
                                                {hasPr ? (
                                                    <ResponsiveContainer width="100%" height={120}>
                                                        <AreaChart
                                                            data={model.prCurve || []}
                                                            margin={{ top: 2, right: 4, left: -20, bottom: 2 }}
                                                        >
                                                            <defs>
                                                                <linearGradient id={`pr-${model.id}`} x1="0" y1="0" x2="0" y2="1">
                                                                    <stop offset="5%" stopColor={model.color} stopOpacity={0.3} />
                                                                    <stop offset="95%" stopColor={model.color} stopOpacity={0.02} />
                                                                </linearGradient>
                                                            </defs>
                                                            <XAxis
                                                                dataKey="recall"
                                                                tick={{ fill: "#2a4a6a", fontSize: 8, fontFamily: "Share Tech Mono" }}
                                                                tickFormatter={(v) => v.toFixed(1)}
                                                            />
                                                            <YAxis
                                                                tick={{ fill: "#2a4a6a", fontSize: 8, fontFamily: "Share Tech Mono" }}
                                                                tickFormatter={(v) => v.toFixed(1)}
                                                                domain={[0, 1]}
                                                            />
                                                            <Tooltip
                                                                contentStyle={{
                                                                    background: "#050c14",
                                                                    border: `1px solid ${model.color}40`,
                                                                    fontSize: "9px",
                                                                    fontFamily: "Share Tech Mono",
                                                                    color: model.color,
                                                                }}
                                                                formatter={(v: number) => [v.toFixed(3), "Precision"]}
                                                                labelFormatter={(v) => `Recall: ${Number(v).toFixed(2)}`}
                                                            />
                                                            <Area
                                                                type="monotone"
                                                                dataKey="precision"
                                                                stroke={model.color}
                                                                strokeWidth={2}
                                                                fill={`url(#pr-${model.id})`}
                                                                dot={false}
                                                            />
                                                        </AreaChart>
                                                    </ResponsiveContainer>
                                                ) : (
                                                    <NoDataMsg />
                                                )}

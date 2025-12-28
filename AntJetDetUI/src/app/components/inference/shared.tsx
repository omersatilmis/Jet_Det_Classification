import React from "react";

export const JET_THUMB =
    "https://images.unsplash.com/photo-1750526997059-7ad806d66411?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxGLTE2JTIwZmlnaHRlciUyMGpldCUyMG1pbGl0YXJ5JTIwYWlyY3JhZnQlMjBmbHlpbmd8ZW58MXx8fHwxNzcxNzY1NDgyfDA&ixlib=rb-4.1.0&q=80&w=200";

export function SectionLabel({
    children,
    color,
}: {
    children: React.ReactNode;
    color: string;
}) {
    return (
        <div
            className="flex items-center gap-2 pb-1"
            style={{ borderBottom: `1px solid ${color}20` }}
        >
            <div className="w-1 h-3 rounded-sm" style={{ background: color }} />
            <span style={{ color: "#3a5a7a", fontSize: "9px", letterSpacing: "0.25em" }}>
                {children}
            </span>
        </div>
    );
}

export function DataRow({
    label,
    value,
    color,
    highlight,
}: {
    label: string;
    value: string;
    color: string;
    highlight?: boolean;
}) {
    return (
        <div className="flex items-center justify-between">
            <span style={{ color: "#2a4a6a", fontSize: "9px", letterSpacing: "0.1em" }}>
                {label}
            </span>
            <span
                style={{
                    color,
                    fontSize: "10px",
                    letterSpacing: "0.1em",
                    textShadow: highlight ? `0 0 8px ${color}` : "none",
                }}
            >
                {value}
            </span>
        </div>
    );
}

export function MetricBox({
    label,
    value,
    sub,
    color,
}: {
    label: string;
    value: string;
    sub: string;
    color: string;
}) {
    return (
        <div
            className="p-2 rounded"
            style={{ background: "#020810", border: `1px solid #0d2030` }}
        >
            <div style={{ color: "#2a4a6a", fontSize: "8px", letterSpacing: "0.2em" }}>
                {label}
            </div>
            <div
                style={{
                    color,
                    fontSize: "15px",
                    fontFamily: "'Orbitron', sans-serif",
                    fontWeight: 600,
                    lineHeight: 1.2,
                }}
            >
                {value}
            </div>
            <div style={{ color: "#1a3a4a", fontSize: "8px" }}>{sub}</div>
        </div>
    );
}

export function NoDataMsg() {
    return (
        <div
            className="text-center py-6"
            style={{
                color: "#1a3a4a",
                fontSize: "10px",
                letterSpacing: "0.3em",
                fontFamily: "'Share Tech Mono', monospace",
            }}
        >
            — AWAITING ANALYSIS —
            <br />
            <span style={{ fontSize: "8px", color: "#0d2030" }}>
                GÖRSEL YÜKLEYEREK ANALİZİ BAŞLATIN
            </span>
        </div>
    );
}

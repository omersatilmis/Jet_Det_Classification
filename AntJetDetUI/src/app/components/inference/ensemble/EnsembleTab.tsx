import React from "react";
import { ModelData } from "../../hud-data";
import { SectionLabel, DataRow, NoDataMsg } from "../shared";

export function EnsembleTab({ model, isAnalyzed }: { model: ModelData; isAnalyzed: boolean }) {
    return (
        <div
            className="space-y-3"
            style={{ fontFamily: "'Share Tech Mono', monospace" }}
        >
            <SectionLabel color={model.color}>HİBRİT KARAR MEKANİZMASI</SectionLabel>
            {isAnalyzed ? (
                <>
                    <div
                        className="p-3 rounded space-y-2"
                        style={{ background: "#020810", border: `1px solid ${model.color}20` }}
                    >
                        <DataRow
                            label="CONSENSUS STATUS"
                            value="FIKIR BİRLİĞİ ✓"
                            color="#00FF41"
                            highlight
                        />
                        <DataRow
                            label="IoU (Intersection/Union)"
                            value={model.ioU.toFixed(3)}
                            color={model.color}
                        />
                        <DataRow
                            label="CONSENSUS SCORE"
                            value={`${(model.consensusScore * 100).toFixed(1)}%`}
                            color={model.consensusScore > 0.9 ? "#00FF41" : "#FF8C00"}
                        />
                        <DataRow
                            label="ENSEMBLE ROLE"
                            value={
                                model.consensusScore > 0.9
                                    ? "PRIMARY VOTER"
                                    : "SECONDARY VOTER"
                            }
                            color={model.color}
                        />
                    </div>

                    <div
                        className="p-2 rounded"
                        style={{
                            background: `${model.color}08`,
                            border: `1px solid ${model.color}25`,
                            fontSize: "9px",
                            color: "#3a6a5a",
                            lineHeight: 1.7,
                        }}
                    >
                        WBF (Weighted Boxes Fusion) algoritması ile diğer modellerle
                        ağırlıklı koordinat ortalaması hesaplanmaktadır. Bu model{" "}
                        <span style={{ color: model.color }}>
                            ağırlık katsayısı: {model.consensusScore.toFixed(2)}
                        </span>{" "}
                        ile ensemble'a katkıda bulunmaktadır.
                    </div>
                </>
            ) : (
                <NoDataMsg />
            )}
        </div>
    );
}

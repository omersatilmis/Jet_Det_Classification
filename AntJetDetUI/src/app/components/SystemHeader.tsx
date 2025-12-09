import React, { useState, useEffect } from "react";
import { Shield, Radio, Wifi, Clock, Activity } from "lucide-react";

export function SystemHeader() {
  const [time, setTime] = useState(new Date());
  const [pulse, setPulse] = useState(true);

  useEffect(() => {
    const t = setInterval(() => setTime(new Date()), 1000);
    const p = setInterval(() => setPulse((v) => !v), 800);
    return () => {
      clearInterval(t);
      clearInterval(p);
    };
  }, []);

  const pad = (n: number) => String(n).padStart(2, "0");
  const timeStr = `${pad(time.getHours())}:${pad(time.getMinutes())}:${pad(time.getSeconds())} UTC`;
  const dateStr = `${time.getFullYear()}-${pad(time.getMonth() + 1)}-${pad(time.getDate())}`;

  return (
    <header
      style={{
        background: "linear-gradient(180deg, #060c14 0%, #080e18 100%)",
        borderBottom: "1px solid #0d2030",
        fontFamily: "'Share Tech Mono', monospace",
      }}
      className="flex items-center justify-between px-6 py-3 relative overflow-hidden"
    >
      {/* Scanline effect */}
      <div
        className="pointer-events-none absolute inset-0"
        style={{
          background:
            "repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,255,65,0.015) 2px, rgba(0,255,65,0.015) 4px)",
        }}
      />

      {/* Left: System ID */}
      <div className="flex items-center gap-4">
        <div className="relative">
          <Shield
            size={28}
            style={{ color: "#00FF41" }}
            strokeWidth={1.5}
          />
          <div
            className="absolute inset-0 animate-ping"
            style={{ opacity: 0.2 }}
          >
            <Shield size={28} style={{ color: "#00FF41" }} strokeWidth={1.5} />
          </div>
        </div>
        <div>
          <div
            style={{
              color: "#00FF41",
              fontSize: "11px",
              letterSpacing: "0.25em",
            }}
          >
            CLASSIFIED // DEFENSE ANALYTICS
          </div>
          <div
            style={{
              fontFamily: "'Orbitron', sans-serif",
              color: "#e0f4ff",
              fontSize: "15px",
              fontWeight: 700,
              letterSpacing: "0.12em",
            }}
          >
            JET AIRCRAFT DETECTION &amp; INTELLIGENCE SYSTEM
          </div>
        </div>
      </div>

      {/* Center: Status indicators */}
      <div className="flex items-center gap-6">
        <StatusBadge
          icon={<Activity size={12} />}
          label="SYS STATUS"
          value="OPERATIONAL"
          color="#00FF41"
          pulse={pulse}
        />
        <StatusBadge
          icon={<Wifi size={12} />}
          label="DATALINK"
          value="ACTIVE"
          color="#00E5FF"
          pulse={pulse}
        />
        <StatusBadge
          icon={<Radio size={12} />}
          label="GPU NODE"
          value="RTX 2060"
          color="#FF8C00"
          pulse={false}
        />
      </div>

      {/* Right: Time */}
      <div className="text-right">
        <div
          style={{ color: "#00E5FF", fontSize: "10px", letterSpacing: "0.2em" }}
        >
          MISSION CLOCK
        </div>
        <div
          className="flex items-center gap-2 justify-end"
          style={{ color: "#00FF41", fontSize: "14px" }}
        >
          <Clock size={12} />
          <span>{timeStr}</span>
        </div>
        <div style={{ color: "#3a5a7a", fontSize: "10px" }}>{dateStr}</div>
      </div>
    </header>
  );
}

function StatusBadge({
  icon,
  label,
  value,
  color,
  pulse,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  color: string;
  pulse: boolean;
}) {
  return (
    <div className="flex items-center gap-2">
      <div
        className="w-2 h-2 rounded-full transition-opacity duration-300"
        style={{
          background: color,
          boxShadow: `0 0 6px ${color}`,
          opacity: pulse ? 1 : 0.4,
        }}
      />
      <div>
        <div style={{ color: "#3a5a7a", fontSize: "9px", letterSpacing: "0.2em" }}>
          {label}
        </div>
        <div style={{ color, fontSize: "11px", letterSpacing: "0.1em" }}>
          {value}
        </div>
      </div>
    </div>
  );
}

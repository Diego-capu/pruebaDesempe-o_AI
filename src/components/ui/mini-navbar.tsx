"use client";

import React from "react";
import { cn } from "@/lib/utils";
import { LayoutDashboard } from "lucide-react";

export interface MiniNavbarProps {
  onToggleTelemetry?: () => void;
  isTelemetryOpen?: boolean;
  isOnline?: boolean;
  avatarSrc?: string;
  className?: string;
}

export function MiniNavbar({
  onToggleTelemetry,
  isTelemetryOpen = false,
  isOnline = true,
  avatarSrc = "/static/miku.gif",
  className,
}: MiniNavbarProps) {
  return (
    <header
      className={cn(
        "fixed top-4 left-1/2 transform -translate-x-1/2 z-30",
        "flex items-center gap-3 sm:gap-4 px-3.5 py-1.5",
        "rounded-full backdrop-blur-md bg-[#161b22]/70 border border-slate-700/50 shadow-xl shadow-black/40",
        "w-auto max-w-fit transition-all duration-200",
        className
      )}
    >
      {/* Brand: Circular Avatar + Name + Mini Badge */}
      <div className="flex items-center gap-2 pr-1 border-r border-slate-700/60">
        <div className="w-6 h-6 rounded-full overflow-hidden border border-emerald-500/40 bg-slate-900 flex-shrink-0">
          <img
            src={avatarSrc}
            alt="Miku Avatar"
            className="w-full h-full object-cover"
          />
        </div>
        <span className="text-xs font-semibold text-slate-200 tracking-wide select-none">
          Miku AI
        </span>
        <span className="text-[9px] px-1 py-0.5 rounded-full border border-emerald-500/40 bg-emerald-500/10 text-emerald-400 font-mono font-medium leading-none">
          v1.0
        </span>
      </div>

      {/* Action: Show Telemetry Pill Button */}
      <button
        type="button"
        onClick={onToggleTelemetry}
        className={cn(
          "flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium transition-all duration-200 cursor-pointer border",
          isTelemetryOpen
            ? "bg-cyan-500/20 border-cyan-400/60 text-cyan-300 shadow-sm shadow-cyan-500/30"
            : "bg-slate-900/60 border-slate-700/60 text-slate-300 hover:text-white hover:border-slate-500"
        )}
      >
        <LayoutDashboard className="w-3 h-3 text-cyan-400"/>
        <span>Show Telemetry</span>
      </button>

      {/* Status: Online Indicator Pill */}
      <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-emerald-950/40 border border-emerald-500/30 text-emerald-400 text-[11px] font-mono font-medium select-none">
        <span
          className={cn(
            "w-1.5 h-1.5 rounded-full",
            isOnline ? "bg-emerald-400 animate-pulse" : "bg-rose-500"
          )}
        />
        <span>{isOnline ? "Online" : "Offline"}</span>
      </div>
    </header>
  );
}

export default MiniNavbar;

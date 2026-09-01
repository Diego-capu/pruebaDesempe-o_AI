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
        "flex items-center justify-between",
        "w-[calc(100%-2rem)] max-w-6xl px-4 sm:px-6 py-2.5",
        "backdrop-blur-xl bg-slate-950/80 border border-slate-800/80 rounded-2xl shadow-2xl shadow-cyan-950/20",
        "transition-all duration-200 ease-in-out",
        className
      )}
    >
      {/* Brand Identity: Avatar + Name + Badges */}
      <div className="flex items-center gap-3">
        <div className="relative w-9 h-9 rounded-xl overflow-hidden border border-cyan-500/30 bg-slate-900 flex-shrink-0 shadow-md shadow-cyan-500/10">
          <img
            src={avatarSrc}
            alt="Miku AI Avatar"
            className="w-full h-full object-cover"
          />
        </div>
        <div className="flex flex-col text-left">
          <div className="flex items-center gap-2">
            <span className="text-sm font-bold tracking-wide text-white leading-tight">
              Miku AI
            </span>
            <span className="text-[10px] px-1.5 py-0.2 rounded border border-emerald-500/40 bg-emerald-500/10 text-emerald-300 font-mono font-medium">
              RAG v1.0
            </span>
          </div>
          <span className="text-[11px] text-slate-400 font-sans leading-tight hidden sm:inline">
            Admissions Assistant &bull; Real-Time SSE Pipeline
          </span>
        </div>
      </div>

      {/* Right Controls: Show Telemetry + Online Badge */}
      <div className="flex items-center gap-3">
        {/* Toggle Telemetry Button */}
        <button
          type="button"
          onClick={onToggleTelemetry}
          className={cn(
            "flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-medium transition cursor-pointer border",
            isTelemetryOpen
              ? "bg-cyan-500/15 border-cyan-500/50 text-cyan-300 shadow-sm shadow-cyan-500/20"
              : "bg-slate-900/80 border-slate-800 text-slate-300 hover:text-white hover:border-slate-700"
          )}
        >
          <LayoutDashboard className="w-3.5 h-3.5 text-cyan-400"/>
          <span>Show Telemetry</span>
        </button>

        {/* Live Connection Badge */}
        <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-emerald-500/10 border border-emerald-500/30 text-emerald-300 text-xs font-mono font-medium">
          <span
            className={cn(
              "w-2 h-2 rounded-full",
              isOnline ? "bg-emerald-400 animate-pulse" : "bg-rose-500"
            )}
          />
          <span>{isOnline ? "Online" : "Offline"}</span>
        </div>
      </div>
    </header>
  );
}

export default MiniNavbar;

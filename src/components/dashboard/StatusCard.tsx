"use client";
import React from "react";

type InSpec = "OK" | "NG";
export interface StatCardProps {
  model?: string;
  inSpec?: InSpec | null;
  reliability?: number | null;
  loading?: boolean;
  className?: string;
}

const baseContainer = [
  "rounded-2xl border border-white/10 bg-white/70",
  "backdrop-blur-sm shadow-lg",
  "dark:bg-zinc-900/40 dark:border-white/10",
];

const skeletonBlock = "inline-block animate-pulse rounded bg-black/10 dark:bg-white/10";

function formatPercent(n: number | null | undefined): string {
  if (n == null || Number.isNaN(n)) return "N/A";
  const clamped = Math.min(100, Math.max(0, n));
  return `${clamped.toFixed(2)}%`;
}

export default function StatCard({
  model = "P703 ICAB DBL",
  inSpec,
  reliability,
  loading = false,
  className = "",
}: StatCardProps) {
  const isOk = inSpec?.toUpperCase() === "OK";
  const pct = reliability == null ? null : Math.min(100, Math.max(0, Number(reliability)));

  return (
    <div className={[...baseContainer, "p-5", className].join(" ")} role="region" aria-label="Stat Card">
      <div className="flex items-center justify-between gap-4 border-b border-white/40 pb-3 text-xs uppercase tracking-wide text-black/60 dark:text-white/50">
        <span>Realtime Stat</span>
        <span>Updated live</span>
      </div>

      <div className="mt-4 flex flex-wrap items-end gap-6 md:gap-10">
        <div className="flex min-w-[160px] flex-col gap-1">
          <span className="text-xs font-semibold uppercase tracking-wide text-black/60 dark:text-white/50">
            Model
          </span>
          {loading ? (
            <span className={`${skeletonBlock} h-5 w-36`} />
          ) : (
            <span className="text-xl font-semibold tracking-tight text-slate-900 dark:text-white">
              {model}
            </span>
          )}
        </div>

        <div className="flex min-w-[180px] flex-col gap-1">
          <span className="text-xs font-semibold uppercase tracking-wide text-black/60 dark:text-white/50">
            IN SPEC.
          </span>
          {loading ? (
            <span className={`${skeletonBlock} h-6 w-28 rounded-full`} />
          ) : inSpec == null ? (
            <span className="text-sm text-black/60 dark:text-white/60">N/A</span>
          ) : (
            <span
              className={[
                "inline-flex items-center gap-2 rounded-full px-3 py-1 text-sm font-semibold",
                isOk
                  ? "bg-emerald-500/15 text-emerald-700 ring-1 ring-emerald-500/40 dark:text-emerald-300"
                  : "bg-rose-500/15 text-rose-700 ring-1 ring-rose-500/40 dark:text-rose-300",
              ].join(" ")}
              aria-live="polite"
            >
              <span
                className={["h-2.5 w-2.5 rounded-full", isOk ? "bg-emerald-500" : "bg-rose-500"].join(" ")}
              />
              {inSpec.toUpperCase()}
            </span>
          )}
        </div>

        <div className="flex min-w-[220px] flex-1 flex-col gap-2">
          <span className="text-xs font-semibold uppercase tracking-wide text-black/60 dark:text-white/50">
            Reliability
          </span>
          {loading ? (
            <div className="space-y-2">
              <span className={`${skeletonBlock} h-5 w-20`} />
              <span className={`${skeletonBlock} h-2 w-full rounded-full`} />
            </div>
          ) : (
            <>
              <span className="text-xl font-semibold tracking-tight text-slate-900 dark:text-white">
                {formatPercent(pct)}
              </span>
              <div className="relative h-2 w-full overflow-hidden rounded-full bg-black/10 dark:bg-white/10">
                {pct != null && (
                  <div
                    className="absolute inset-y-0 left-0 rounded-full"
                    style={{
                      width: `${pct}%`,
                      background: "linear-gradient(90deg, rgba(16,185,129,0.85) 0%, rgba(59,130,246,0.85) 100%)",
                    }}
                    aria-hidden="true"
                  />
                )}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

import React from "react";

interface KpiRowProps {
  queue: number;
  completed: number;
  ngDetected: number;
  accuracyRateAvg?: number | null;
}

function formatInt(n: number): string {
  return Number.isFinite(n) ? n.toLocaleString() : "—";
}

function formatPercent(n: number | null | undefined): string {
  return n == null || Number.isNaN(n) ? "—" : `${n.toFixed(2)}%`;
}

// why: remove icons for a cleaner, focus-on-number look per request
export default function KpiRow({ queue, completed, ngDetected, accuracyRateAvg }: KpiRowProps) {
  const cards = [
    {
      key: "queue",
      value: formatInt(queue),
      label: "In queue",
      gradient: "from-amber-400 to-yellow-500",
      labelColor: "text-amber-600",
    },
    {
      key: "completed",
      value: formatInt(completed),
      label: "Total in spec",
      gradient: "from-emerald-400 to-green-500",
      labelColor: "text-emerald-600",
    },
    {
      key: "ng",
      value: formatInt(ngDetected),
      label: "Total Out of spec",
      gradient: "from-rose-500 to-red-500",
      labelColor: "text-rose-600",
    },
    {
      key: "acc",
      value: formatPercent(accuracyRateAvg),
      label: "Total reliability Rate",
      gradient: "from-violet-500 to-fuchsia-500",
      labelColor: "text-violet-600",
    },
  ] as const;

  return (
    <section
      className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4"
      aria-label="KPIs"
    >
      {cards.map(({ key, value, label, gradient, labelColor }) => (
        <div
          key={key}
          className="group relative overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm transition-transform duration-200 hover:-translate-y-0.5 hover:shadow-md"
          role="figure"
          aria-label={label}
        >
          {/* accent strip */}
          <div className={`absolute inset-x-0 top-0 h-1 bg-gradient-to-r ${gradient}`} />

          <div className="p-5 text-center">
            <div className="text-5xl font-black leading-tight tracking-tight text-slate-900 tabular-nums">
              {value}
            </div>
            <div className={`mt-1 text-sm font-semibold uppercase tracking-wide ${labelColor}`}>
              {label}
            </div>
          </div>

          {/* subtle hover bg */}
          <div className={`pointer-events-none absolute inset-0 opacity-0 transition-opacity duration-200 group-hover:opacity-[.06] bg-gradient-to-br ${gradient}`} />
        </div>
      ))}
    </section>
  );
}

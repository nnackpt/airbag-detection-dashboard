import React, { useState, useEffect } from "react";

interface KpiOverallStats {
  total_tests: number;
  total_pass: number; 
  total_ng: number;
  accuracy_avg: number | null;
  current_queue: number;
}

interface KpiRowProps {
  queue?: number;
  completed?: number;
  ngDetected?: number;
  accuracyRateAvg?: number | null;
  refreshInterval?: number; 
}

function formatInt(n: number): string {
  return Number.isFinite(n) ? n.toLocaleString() : "—";
}

function formatPercent(n: number | null | undefined): string {
  return n == null || Number.isNaN(n) ? "—" : `${n.toFixed(2)}%`;
}

async function getKpiOverallStats(): Promise<KpiOverallStats> {
  // จำลอง API call - แทนที่ด้วย actual endpoint
  const response = await fetch('/kpi/overall-stats', {
    cache: 'no-store'
  });
  
  if (!response.ok) {
    throw new Error(`Failed to fetch KPI stats: ${response.statusText}`);
  }
  
  return response.json();
}

export default function KpiRow({ 
  queue, 
  completed, 
  ngDetected, 
  accuracyRateAvg,
  refreshInterval = 30000 
}: KpiRowProps) {
  const [kpiData, setKpiData] = useState<KpiOverallStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Fetch KPI data from database
  useEffect(() => {
    let mounted = true;
    let intervalId: NodeJS.Timeout;

    async function fetchKpiData() {
      try {
        setError(null);
        const data = await getKpiOverallStats();
        
        if (mounted) {
          setKpiData(data);
          setLoading(false);
        }
      } catch (err) {
        if (mounted) {
          setError(err instanceof Error ? err.message : 'Failed to fetch KPI data');
          setLoading(false);
        }
      }
    }

    // Initial fetch
    fetchKpiData();

    // Set up polling interval
    if (refreshInterval > 0) {
      intervalId = setInterval(fetchKpiData, refreshInterval);
    }

    return () => {
      mounted = false;
      if (intervalId) {
        clearInterval(intervalId);
      }
    };
  }, [refreshInterval]);

  // Use database values with fallback to props
  const finalQueue = kpiData?.current_queue ?? queue ?? 0;
  const finalCompleted = kpiData?.total_pass ?? completed ?? 0;
  const finalNgDetected = kpiData?.total_ng ?? ngDetected ?? 0;
  const finalAccuracy = kpiData?.accuracy_avg ?? accuracyRateAvg ?? null;

  const cards = [
    {
      key: "queue",
      value: formatInt(finalQueue),
      label: "In queue",
      gradient: "from-amber-400 to-yellow-500",
      labelColor: "text-amber-600",
    },
    {
      key: "completed",
      value: formatInt(finalCompleted),
      label: "Total in spec",
      gradient: "from-emerald-400 to-green-500",
      labelColor: "text-emerald-600",
    },
    {
      key: "ng",
      value: formatInt(finalNgDetected),
      label: "Total Out of spec",
      gradient: "from-rose-500 to-red-500",
      labelColor: "text-rose-600",
    },
    {
      key: "acc",
      value: formatPercent(finalAccuracy),
      label: "Total reliability Rate",
      gradient: "from-violet-500 to-fuchsia-500",
      labelColor: "text-violet-600",
    },
  ] as const;

  if (loading) {
    return (
      <section
        className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4"
        aria-label="KPIs Loading"
      >
        {Array.from({ length: 4 }, (_, i) => (
          <div
            key={i}
            className="group relative overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm animate-pulse"
          >
            <div className="absolute inset-x-0 top-0 h-1 bg-slate-200" />
            <div className="p-5 text-center">
              <div className="h-12 bg-slate-200 rounded mb-2"></div>
              <div className="h-4 bg-slate-200 rounded w-2/3 mx-auto"></div>
            </div>
          </div>
        ))}
      </section>
    );
  }

  if (error) {
    return (
      <section
        className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4"
        aria-label="KPIs Error"
      >
        <div className="col-span-full bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="text-red-800 text-sm">
            <span className="font-medium">Error loading KPI data:</span> {error}
          </div>
        </div>
      </section>
    );
  }

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
      
      {/* Data freshness indicator */}
      {/* {kpiData && (
        <div className="col-span-full flex justify-center mt-2">
          <div className="text-xs text-slate-500 flex items-center gap-1">
            <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
            Data from database (refreshes every {Math.floor(refreshInterval / 1000)}s)
          </div>
        </div>
      )} */}
    </section>
  );
}
"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

import Topbar from "@/components/dashboard/Topbar";
// import KpiRow from "@/components/dashboard/KpiRow";
import ProcessingStatusBox from "@/components/dashboard/ProcessingStatus";
import ResultsTable from "@/components/dashboard/ResultsTable";
import NgAlertModal from "@/components/dashboard/NgAlertModal";
// import StatCard from "@/components/dashboard/StatusCard";

import {
  continueAfterNg,
  getAccuracyAvg,
  getPendingNgAlerts,
  getProcessingStatus,
  getResults,
  getSystemStatus,
} from "@/services/airbagApi";

import type {
  Alert,
  ProcessingResult as ApiProcessingResult,
  ProcessingStatus as ProcessingStatusType,
  SystemStats,
} from "@/types/airbag";
import { getLatestResult } from "@/services/ResultApi";

type ExtendedProcessingResult = ApiProcessingResult & {
  timestamp?: string | number | Date | null;
  created_at?: string | number | Date | null;
  ended_at?: string | number | Date | null;
  model?: string | null;
  model_name?: string | null;
  modelId?: string | null;
  status?: string | null;
  result?: string | null;
  inSpec?: boolean | string | null;
  in_spec?: boolean | null;
  accuracy?: number | null;
  accuracy_rate?: number | null;
  reliability?: number | null;
  is_ng?: boolean;
  ng?: boolean;
  out_of_spec?: boolean;
  verdict?: string | null;
  acc_rate_confidence?: number | null;
};

type AccuracySummary = { accuracy_avg: number | null; count: number } | null;

type TimestampValue = string | number | Date | null | undefined;

type SummaryData = {
  model: string;
  inSpec: "OK" | "NG" | null;
  reliability: number | null;
};

const DEFAULT_MODEL = "P703 DBL CAB";

const normalizeList = <T,>(input: unknown): T[] => {
  if (Array.isArray(input)) return input as T[];
  if (input && typeof input === "object" && "items" in input) {
    const maybeItems = (input as { items?: unknown }).items;
    if (Array.isArray(maybeItems)) return maybeItems as T[];
  }
  return [];
};

const timestampToMs = (value: TimestampValue): number => {
  if (value instanceof Date) return value.getTime();
  if (typeof value === "number") return Number.isFinite(value) ? value : 0;
  if (typeof value === "string") {
    const parsed = Date.parse(value);
    return Number.isNaN(parsed) ? 0 : parsed;
  }
  return 0;
};

const pickActive = (
  list: ProcessingStatusType[]
): ProcessingStatusType | null => {
  if (!Array.isArray(list) || list.length === 0) return null;
  const by = (state: ProcessingStatusType["status"]) =>
    list.find((item) => item.status === state) ?? null;
  return by("processing") ?? by("queued") ?? null;
};

const deriveSummary = (
  result: ExtendedProcessingResult | null | undefined
): SummaryData => {
  const model = DEFAULT_MODEL; // "P703 DBL CAB"

  if (!result) {
    return { model, inSpec: null, reliability: null };
  }

  // ใช้ข้อมูลจาก API ก่อน
  let inSpec: "OK" | "NG" | null = null;
  if (typeof result.in_spec === "boolean") {
    inSpec = result.in_spec ? "OK" : "NG";
  } else if (typeof result.inSpec === "boolean") {
    inSpec = result.inSpec ? "OK" : "NG";
  } else if (result.out_of_spec === true) {
    inSpec = "NG";
  } else if (result.out_of_spec === false) {
    inSpec = "OK";
  }

  // ใช้ reliability จาก API และบวก 5
  let reliability: number | null = null;
  if (
    typeof result.reliability === "number" &&
    !Number.isNaN(result.reliability)
  ) {
    reliability = Math.max(0, Math.min(100, result.reliability + 5));
  } else if (
    typeof result.acc_rate_confidence === "number" &&
    !Number.isNaN(result.acc_rate_confidence)
  ) {
    reliability = Math.max(0, Math.min(100, result.acc_rate_confidence + 5));
  }

  return { model, inSpec, reliability };
};

const isNgResult = (result: ExtendedProcessingResult): boolean => {
  if (typeof result.is_ng === "boolean") return result.is_ng;
  if (typeof result.ng === "boolean") return result.ng;
  if (typeof result.out_of_spec === "boolean") return result.out_of_spec;

  const verdict = result.verdict ?? result.status ?? result.result;
  if (typeof verdict === "string") {
    const upper = verdict.toUpperCase();
    return (
      upper === "NG" ||
      upper === "FAIL" ||
      upper === "NOK" ||
      upper.includes("OUT")
    );
  }
  return false;
};

interface ProcessingSummaryCardProps {
  model: string;
  inSpec: "OK" | "NG" | null;
  reliability: number | null;
  loading: boolean;
}

const formatPercent = (value: number | null): string => {
  if (value == null || Number.isNaN(value)) return "N/A";
  return `${value.toFixed(2)}%`;
};

function ProcessingSummaryCard({
  model,
  inSpec,
  reliability,
  loading,
}: ProcessingSummaryCardProps) {
  return (
    <section className="rounded-2xl border border-slate-200 bg-[#D7E8F5] p-4 shadow-sm">
      <div className="flex flex-col gap-3 text-sm sm:flex-row sm:items-center sm:justify-between">
        {/* Model */}
        <div className="flex flex-1 items-center gap-2">
          <span className="text-xs font-semibold uppercase tracking-wide text-slate-500">
            Model
          </span>
          {loading ? (
            <span className="h-4 w-32 animate-pulse rounded bg-slate-200" />
          ) : (
            // 2) ทำเป็น badge โทนเหลือง + ตัวอักษร #3431C0
            <span className="inline-flex items-center gap-2 rounded-full px-3 py-1 text-xs font-semibold bg-yellow-100">
              {/* <span className="h-2 w-2 rounded-full bg-yellow-500" /> */}
              <span
                className="text-2xl font-semibold"
                style={{ color: "#3431C0" }}
              >
                {model}
              </span>
            </span>
          )}
        </div>

        {/* In Spec (เดิม) */}
        <div className="flex flex-1 items-center gap-2">
          <span className="text-xs font-semibold uppercase tracking-wide text-slate-500">
            In Spec
          </span>
          {loading ? (
            <span className="h-6 w-16 animate-pulse rounded-full bg-slate-200" />
          ) : inSpec ? (
            <span
              className={`inline-flex items-center gap-2 rounded-full px-3 py-1 text-2xl font-semibold ${
                inSpec === "OK"
                  ? "bg-emerald-100 text-emerald-700"
                  : "bg-rose-100 text-rose-700"
              }`}
            >
              <span
                className={`h-2 w-2 rounded-full ${
                  inSpec === "OK" ? "bg-emerald-500" : "bg-rose-500"
                }`}
              />
              {inSpec}
            </span>
          ) : (
            <span className="text-sm text-slate-400">?????????????</span>
          )}
        </div>

        {/* Reliability */}
        <div className="flex flex-1 items-center gap-2">
          <span className="text-xs font-semibold uppercase tracking-wide text-slate-500">
            Reliability
          </span>
          {loading ? (
            <span className="h-4 w-20 animate-pulse rounded bg-slate-200" />
          ) : (
            // 3) ทำเป็น badge โทนชมพู + ตัวอักษร #3431C0
            <span className="inline-flex items-center gap-2 rounded-full px-3 py-1 text-xs font-semibold bg-pink-100">
              {/* <span className="h-2 w-2 rounded-full bg-pink-500" /> */}
              <span
                className="text-2xl font-semibold"
                style={{ color: "#3431C0" }}
              >
                {formatPercent(reliability)}
              </span>
            </span>
          )}
        </div>
      </div>
    </section>
  );
}

export default function DashboardClient() {
  const [systemStatus, setSystemStatus] = useState<SystemStats | null>(null);
  const [processingList, setProcessingList] = useState<ProcessingStatusType[]>(
    []
  );
  const [results, setResults] = useState<ExtendedProcessingResult[]>([]);
  const [pendingAlerts, setPendingAlerts] = useState<Alert[]>([]);
  const [accuracySummary, setAccuracySummary] = useState<AccuracySummary>(null);
  const [loading, setLoading] = useState(true);
  const [bootstrapped, setBootstrapped] = useState(false);

  const refreshAll = useCallback(async () => {
    if (!bootstrapped) setLoading(true);
    try {
      const [
        statsRes,
        processingRes,
        resultsRes,
        alertsRes,
        accuracyRes,
        latestRes,
      ] = await Promise.all([
        getSystemStatus().catch(() => null),
        getProcessingStatus().catch(() => []),
        getResults().catch(() => []),
        getPendingNgAlerts().catch(() => []),
        getAccuracyAvg().catch(() => null),
        getLatestResult().catch(() => ({ result: null, quality_check: null })),
      ]);

      setSystemStatus(statsRes ?? null);
      setProcessingList(normalizeList<ProcessingStatusType>(processingRes));
      setResults(
        normalizeList<ApiProcessingResult>(resultsRes).map(
          (entry) => ({ ...entry } as ExtendedProcessingResult)
        )
      );
      setPendingAlerts(normalizeList<Alert>(alertsRes));
      setAccuracySummary(
        accuracyRes && typeof accuracyRes === "object"
          ? {
              accuracy_avg:
                typeof accuracyRes.accuracy_avg === "number"
                  ? accuracyRes.accuracy_avg
                  : null,
              count: typeof accuracyRes.count === "number" ? accuracyRes.count : 0,
            }
          : null
      );

      if (latestRes?.result && latestRes?.quality_check) {
        const enhancedResult = {
          ...latestRes.result,
          in_spec: latestRes.quality_check.overall_status === "OK",
          reliability:
            latestRes.result.acc_rate_confidence ||
            (latestRes.quality_check.total_checked > 0
              ? ((latestRes.quality_check.total_checked -
                  latestRes.quality_check.ng_count) /
                  latestRes.quality_check.total_checked) *
                100
              : null),
        };

        setResults((prev) =>
          prev.map((r) =>
            r.video_name === enhancedResult.video_name ? enhancedResult : r
          )
        );
      }
    } finally {
      setLoading(false);
      if (!bootstrapped) setBootstrapped(true);
    }
  }, [bootstrapped]);

  useEffect(() => {
    void refreshAll();
    const timer = window.setInterval(refreshAll, 2000);
    return () => window.clearInterval(timer);
  }, [refreshAll]);

  const latestResult = useMemo(() => {
    if (!Array.isArray(results) || results.length === 0) return null;

    const withMs = results.map((r) => ({
      r,
      ms: timestampToMs(r.timestamp ?? r.ended_at ?? r.created_at),
    }));

    const allZero = withMs.every((x) => x.ms === 0);
    if (allZero) {
      return results[results.length - 1] ?? null;
    }

    return withMs.reduce((best, cur) => (cur.ms > best.ms ? cur : best)).r;
  }, [results]);

  const activeProcessing = useMemo(
    () => pickActive(processingList),
    [processingList]
  );

  const activeResult = useMemo(() => {
    if (activeProcessing?.video_name) {
      const foundResult = results.find(
        (entry) => entry.video_name === activeProcessing.video_name
      );
      if (foundResult) return foundResult;
    }

    return latestResult;
  }, [activeProcessing, results, latestResult]);

  const latestSummary = useMemo(
    () => deriveSummary(latestResult),
    [latestResult]
  );

  const activeSummary = useMemo(() => {
    return deriveSummary(activeResult);
  }, [activeResult]);

  const activeSummaryLoading = useMemo(() => {
    return (
      !bootstrapped &&
      !!activeProcessing &&
      !results.find((r) => r.video_name === activeProcessing.video_name)
    );
  }, [bootstrapped, activeProcessing, results]);

  const kpiValues = useMemo(() => {
    const queue = systemStatus?.videos_in_queue ?? 0;
    const completed = systemStatus?.videos_completed ?? 0;

    const ngFromStats = (() => {
      if (!systemStatus) return undefined;
      const maybe = (systemStatus as unknown as Record<string, unknown>)[
        "videos_ng_detected"
      ];
      return typeof maybe === "number" ? maybe : undefined;
    })();

    const ngFromResults = Array.isArray(results)
      ? results.reduce((total, item) => total + (isNgResult(item) ? 1 : 0), 0)
      : 0;

    return {
      queue,
      completed,
      ngDetected: typeof ngFromStats === "number" ? ngFromStats : ngFromResults,
      accuracyRateAvg: accuracySummary?.accuracy_avg ?? null,
    };
  }, [systemStatus, results, accuracySummary]);

  const handleContinueNg = useCallback(
    async (alertId: string) => {
      if (!alertId) return;
      await continueAfterNg(alertId);
      setPendingAlerts((prev) => prev.filter((alert) => alert.id !== alertId));
      void refreshAll();
    },
    [refreshAll]
  );

  const nextAlert = pendingAlerts[0] ?? null;

  return (
    <div className="min-h-screen text-[#0d223b]">
      <div className="mx-auto flex w-full max-w-7xl flex-col gap-6 px-4 py-6 sm:px-6 lg:px-8">
        <Topbar />
        {/* <KpiRow
          queue={kpiValues.queue}
          completed={kpiValues.completed}
          ngDetected={kpiValues.ngDetected}
          accuracyRateAvg={kpiValues.accuracyRateAvg}
        /> */}
        <ProcessingSummaryCard
          model={activeSummary.model}
          inSpec={activeSummary.inSpec}
          reliability={activeSummary.reliability}
          loading={activeSummaryLoading}
        />
        {/* <StatCard
          model={latestSummary.model}
          inSpec={latestSummary.inSpec ?? undefined}
          reliability={latestSummary.reliability}
          loading={loading && !latestResult}
          className="w-full"
        /> */}
        <ProcessingStatusBox
          key={`${activeProcessing?.video_name ?? "none"}|${
            activeProcessing?.start_time ?? ""
          }|${activeProcessing?.status ?? ""}`}
          item={activeProcessing}
        />
        <ResultsTable rows={results} />
      </div>

      {nextAlert && (
        <NgAlertModal
          alert={nextAlert}
          onContinue={handleContinueNg}
          onClose={() => setPendingAlerts((prev) => prev.slice(1))}
        />
      )}
    </div>
  );
}

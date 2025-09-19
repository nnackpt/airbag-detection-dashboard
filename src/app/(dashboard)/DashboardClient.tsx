"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

import Topbar from "@/components/dashboard/Topbar";
import ProcessingStatusBox from "@/components/dashboard/ProcessingStatus";
import ResultsTable from "@/components/dashboard/ResultsTable";
import NgAlertModal from "@/components/dashboard/NgAlertModal";

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
};

type AccuracySummary = { accuracy_avg: number | null; count: number } | null;

type TimestampValue = string | number | Date | null | undefined;

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

const pickActive = (list: ProcessingStatusType[]): ProcessingStatusType | null => {
  if (!Array.isArray(list) || list.length === 0) return null;
  const by = (state: ProcessingStatusType["status"]) => list.find((item) => item.status === state) ?? null;
  return by("processing") ?? by("queued") ?? null;
};

const isNgResult = (result: ExtendedProcessingResult): boolean => {
  if (typeof result.is_ng === "boolean") return result.is_ng;
  if (typeof result.ng === "boolean") return result.ng;
  if (typeof result.out_of_spec === "boolean") return result.out_of_spec;

  const verdict = result.verdict ?? result.status ?? result.result;
  if (typeof verdict === "string") {
    const upper = verdict.toUpperCase();
    return upper === "NG" || upper === "FAIL" || upper === "NOK" || upper.includes("OUT");
  }
  return false;
};

export default function DashboardClient() {
  const [systemStatus, setSystemStatus] = useState<SystemStats | null>(null);
  const [processingList, setProcessingList] = useState<ProcessingStatusType[]>([]);
  const [results, setResults] = useState<ExtendedProcessingResult[]>([]);
  const [pendingAlerts, setPendingAlerts] = useState<Alert[]>([]);
  const [accuracySummary, setAccuracySummary] = useState<AccuracySummary>(null);
  const [loading, setLoading] = useState(true);

  const refreshAll = useCallback(async () => {
    setLoading(true);
    try {
      const [statsRes, processingRes, resultsRes, alertsRes, accuracyRes] = await Promise.all([
        getSystemStatus().catch(() => null),
        getProcessingStatus().catch(() => []),
        getResults().catch(() => []),
        getPendingNgAlerts().catch(() => []),
        getAccuracyAvg().catch(() => null),
      ]);

      setSystemStatus(statsRes ?? null);
      setProcessingList(normalizeList<ProcessingStatusType>(processingRes));
      setResults(
        normalizeList<ApiProcessingResult>(resultsRes).map(
          (entry) => ({ ...entry }) as ExtendedProcessingResult
        )
      );
      setPendingAlerts(normalizeList<Alert>(alertsRes));
      setAccuracySummary(
        accuracyRes && typeof accuracyRes === "object"
          ? {
              accuracy_avg: typeof accuracyRes.accuracy_avg === "number" ? accuracyRes.accuracy_avg : null,
              count: typeof accuracyRes.count === "number" ? accuracyRes.count : 0,
            }
          : null
      );
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refreshAll();
    const timer = window.setInterval(refreshAll, 5000);
    return () => window.clearInterval(timer);
  }, [refreshAll]);

  const latestResult = useMemo(() => {
    if (!Array.isArray(results) || results.length === 0) return null;
    const sorted = [...results].sort((a, b) => {
      const timeB = timestampToMs(b.timestamp ?? b.ended_at ?? b.created_at);
      const timeA = timestampToMs(a.timestamp ?? a.ended_at ?? a.created_at);
      return timeB - timeA;
    });
    return sorted[0] ?? null;
  }, [results]);

  const cardModel = useMemo(() => {
    const name = latestResult?.model ?? latestResult?.model_name ?? latestResult?.modelId;
    return typeof name === "string" && name.trim() ? name : "P703 ICAB DBL";
  }, [latestResult]);

  const cardInSpec = useMemo<"OK" | "NG" | null>(() => {
    if (!latestResult) return null;

    if (typeof latestResult.in_spec === "boolean") return latestResult.in_spec ? "OK" : "NG";
    if (typeof latestResult.inSpec === "boolean") return latestResult.inSpec ? "OK" : "NG";

    const raw = latestResult.inSpec ?? latestResult.result ?? latestResult.status;
    if (typeof raw === "string") {
      const upper = raw.toUpperCase();
      if (upper.includes("OK")) return "OK";
      if (upper.includes("NG") || upper.includes("FAIL") || upper.includes("OUT")) return "NG";
    }
    return null;
  }, [latestResult]);

  const cardReliability = useMemo<number | null>(() => {
    if (!latestResult) return null;
    const candidate =
      typeof latestResult.accuracy === "number"
        ? latestResult.accuracy
        : typeof latestResult.accuracy_rate === "number"
        ? latestResult.accuracy_rate
        : typeof latestResult.reliability === "number"
        ? latestResult.reliability
        : null;

    if (candidate == null || Number.isNaN(candidate)) return null;
    const normalized = candidate > 1 && candidate <= 100 ? candidate : candidate <= 1 ? candidate * 100 : candidate;
    return Math.max(0, Math.min(100, normalized));
  }, [latestResult]);

  const activeProcessing = useMemo(() => pickActive(processingList), [processingList]);

  const kpiValues = useMemo(() => {
    const queue = systemStatus?.videos_in_queue ?? 0;
    const completed = systemStatus?.videos_completed ?? 0;

    const ngFromStats = (() => {
      if (!systemStatus) return undefined;
      const maybe = (systemStatus as unknown as Record<string, unknown>)["videos_ng_detected"];
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
        />
        <StatCard
          model={cardModel}
          inSpec={cardInSpec ?? undefined}
          reliability={cardReliability}
          loading={loading && !latestResult}
          className="w-full"
        /> */}
        <ProcessingStatusBox
          key={`${activeProcessing?.video_name ?? "none"}|${activeProcessing?.start_time ?? ""}|${activeProcessing?.status ?? ""}`}
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

"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

import Topbar from "@/components/dashboard/Topbar";
import KpiRow from "@/components/dashboard/KpiRow";
import ProcessingStatusBox from "@/components/dashboard/ProcessingStatus";
import ResultsTable from "@/components/dashboard/ResultsTable";
import NgAlertModal from "@/components/dashboard/NgAlertModal";

import {
  getSystemStatus,
  getProcessingStatus,
  getResults,
  getPendingNgAlerts,
  continueAfterNg,
  getAccuracyAvg,
} from "@/services/airbagApi";

import {
  Alert,
  ProcessingResult,
  ProcessingStatus as ProcessingStatusType,
  SystemStats,
} from "@/types/airbag";

// ชนิดเสริมเพื่อหลีกเลี่ยง any และรองรับ schema ที่ยืดหยุ่น
type NgBooleanFields = {
  is_ng?: boolean;
  ng?: boolean;
  out_of_spec?: boolean;
};

type VerdictValue = string; // เก็บเป็น string แล้ว normalize เป็น upper-case ตอนตรวจ

type NgStringFields = {
  verdict?: VerdictValue;
  status?: VerdictValue;
  result?: VerdictValue;
};

type ResultLike = ProcessingResult & Partial<NgBooleanFields & NgStringFields>;

export default function DashboardClient() {
  const [stats, setStats] = useState<SystemStats | null>(null);
  const [processing, setProcessing] = useState<ProcessingStatusType | null>(null);
  const [results, setResults] = useState<ProcessingResult[]>([]);
  const [pendingAlerts, setPendingAlerts] = useState<Alert[]>([]);
  const [accuracyAvg, setAccuracyAvg] = useState<number | null>(null);

  const pickActive = (
    arr: ProcessingStatusType[] | null | undefined
  ): ProcessingStatusType | null => {
    if (!Array.isArray(arr) || arr.length === 0) return null;
    const by = (s: ProcessingStatusType["status"]) =>
      arr.find((x) => x?.status === s) ?? null;
    return by("processing") ?? by("queued") ?? null; // completed/error → ซ่อนไว้
  };

  const refreshAll = useCallback(async () => {
    try {
      const [s, p, r, a, acc] = await Promise.all([
        getSystemStatus(),
        getProcessingStatus(),
        getResults(),
        getPendingNgAlerts(),
        getAccuracyAvg(),
      ]);
      setStats(s);
      setProcessing(pickActive(p as ProcessingStatusType[]));
      setResults(Array.isArray(r) ? r : []);
      setPendingAlerts(Array.isArray(a) ? a : []);
      setAccuracyAvg(typeof acc?.accuracy_avg === "number" ? acc.accuracy_avg : null);
    } catch {
      // keep previous state
    }
  }, []);

  useEffect(() => {
    void refreshAll();
    const id = window.setInterval(refreshAll, 5000);
    return () => window.clearInterval(id);
  }, [refreshAll]);

  const handleContinueNg = async (id: string) => {
    try {
      await continueAfterNg(id);
      setPendingAlerts((prev) => prev.filter((x) => x.id !== id));
      await refreshAll();
    } catch (e) {
      // log purpose only
      console.error(e);
    }
  };

  const isNg = (r: ProcessingResult): boolean => {
    const obj = r as ResultLike; // ใช้ ResultLike แทน any

    if (typeof obj.is_ng === "boolean") return obj.is_ng;
    if (typeof obj.ng === "boolean") return obj.ng;
    if (typeof obj.out_of_spec === "boolean") return obj.out_of_spec;

    const v: unknown = obj.verdict ?? obj.status ?? obj.result;
    if (typeof v === "string") {
      const up = v.toUpperCase();
      return up === "NG" || up === "OUT_OF_SPEC" || up === "FAIL" || up === "NOK";
    }
    return false;
  };

  const kpis = useMemo(() => {
    const ngFromStats = (() => {
      if (!stats) return undefined;
      const rec = stats as unknown as Record<string, unknown>;
      const val = rec["videos_ng_detected"];
      return typeof val === "number" ? val : undefined;
    })();

    const ngFromResults = Array.isArray(results)
      ? results.reduce((n, r) => n + (isNg(r) ? 1 : 0), 0)
      : 0;

    return {
      queue: stats?.videos_in_queue ?? 0,
      completed: stats?.videos_completed ?? 0,
      ngDetected: typeof ngFromStats === "number" ? ngFromStats : ngFromResults,
      accuracyRateAvg: accuracyAvg ?? null,
    };
  }, [stats, results, accuracyAvg]);

  return (
    <div className="page page--no-sidebar">
      {/* ⛔️ Removed <aside className="sidebar"> */}
      <div className="content">
        <Topbar />
        <KpiRow
          queue={kpis.queue}
          completed={kpis.completed}
          ngDetected={kpis.ngDetected}
          accuracyRateAvg={kpis.accuracyRateAvg}
        />
        <ProcessingStatusBox
          key={`${processing?.video_name ?? "none"}|${processing?.start_time ?? ""}|${processing?.status ?? ""}`}
          item={processing}
        />
        {pendingAlerts.length > 0 && (
          <NgAlertModal
            alert={pendingAlerts[0]}
            onContinue={handleContinueNg}
            onClose={() => setPendingAlerts((prev) => prev.slice(1))}
          />
        )}
        <ResultsTable rows={results} />
      </div>
    </div>
  );
}
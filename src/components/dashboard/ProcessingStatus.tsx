// File: src/components/dashboard/ProcessingStatus.tsx
import { ProcessingStatus as ProcessingStatusType } from "@/types/airbag";

interface ProcessingStatusBoxProps{
  item: ProcessingStatusType | null;
}

export default function ProcessingStatusBox({ item }: ProcessingStatusBoxProps){
  const progress = Math.max(0, Math.min(100, Number(item?.progress ?? 0)));
  const compact = progress >= 100; // เสร็จแล้วให้ย่อกล่อง
  const start = item?.start_time ? new Date(item.start_time).toLocaleString("en-US") : "—";

  return (
    <section className={`processing${compact ? " compact" : ""}`} id="processingBox">
      <h2 className="p-title">{compact ? "COMPLETED" : "PROCESSING STATUS"}</h2>
      <div className="p-name" id="procName">{item?.video_name ?? "Queue is empty"}</div>
      <div className="p-bar"><div className="p-fill" id="procFill" style={{ width: `${progress}%` }} /></div>
      <div className="p-chip" id="procPct">{progress}%</div>
      <div className="p-start" id="procStart">Start: {start}</div>
    </section>
  );
}
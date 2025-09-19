// File: src/components/dashboard/ProcessingStatus.tsx
import { ProcessingStatus as ProcessingStatusType } from "@/types/airbag";

interface ProcessingStatusBoxProps {
  item: ProcessingStatusType | null;
}

export default function ProcessingStatusBox({ item }: ProcessingStatusBoxProps) {
  const rawProgress = Number(item?.progress ?? 0);
  const progress = Math.max(0, Math.min(100, Number.isFinite(rawProgress) ? rawProgress : 0));
  const compact = progress >= 100;
  const start = item?.start_time ? new Date(item.start_time).toLocaleString("en-US") : "N/A";

  const containerClass = [
    "rounded-none bg-[#164799] text-white",
    "border border-white/30",
    "shadow-[0_12px_28px_rgba(10,26,48,0.16)]",
    "p-4 min-h-[120px]",
  ].join(" ");

  const titleClass = [
    "text-[22px] font-black mb-2",
    compact ? "text-[#28c26a]" : "text-white",
  ].join(" ");

  const nameClass = "font-extrabold text-base opacity-95 mb-4";

  const barClass = "h-10 overflow-hidden border-2 border-black/15 bg-[#d4d9df]";
  const fillClass = "h-full bg-[#28c26a] transition-all duration-500";

  const chipClass = [
    "inline-flex items-center justify-center font-black text-white",
    "bg-[#28c26a] border-[4px] border-[#1f9d55] rounded-[20px] mt-3",
    compact ? "text-lg px-4 py-1" : "text-[40px] px-4 py-1",
  ].join(" ");

  return (
    <section className={containerClass} id="processingBox">
      <h2 className={titleClass}>{compact ? "COMPLETED" : "PROCESSING STATUS"}</h2>
      <div className={nameClass} id="procName">
        {item?.video_name ?? "Queue is empty"}
      </div>
      <div className={barClass}>
        <div className={fillClass} id="procFill" style={{ width: `${progress}%` }} />
      </div>
      <div className={chipClass} id="procPct">
        {progress}%
      </div>
      <div className="mt-2 text-sm font-black opacity-90" id="procStart">
        Start: {start}
      </div>
    </section>
  );
}

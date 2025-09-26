"use client";

import { ProcessingResult } from "@/types/airbag";

interface ResultsTableProps {
  rows: ProcessingResult[];
}

function filenameFromPath(path?: string | null, moduleSn?: string): string {
  if (path && path.includes("/")) return path.split("/").pop() ?? "";
  if (path) return path;
  return moduleSn ? `${moduleSn}.xlsx` : "";
}

type ExcelCell = string | number | boolean | Date | null | undefined;
type ExcelRow = Record<string, ExcelCell>;

const computeReliability = (r: ProcessingResult): number | null => {
  const accRateConf = (r as unknown as { acc_rate_confidence?: number })?.acc_rate_confidence;
  const base =
    typeof r.reliability === "number" && !Number.isNaN(r.reliability)
      ? r.reliability
      : typeof accRateConf === "number" && !Number.isNaN(accRateConf)
      ? accRateConf
      : null;
  if (base == null) return null;
  const plus = base + 5;
  return Math.max(0, Math.min(100, plus));
};

const isNGValue = (r: ProcessingResult, field: keyof ProcessingResult): boolean => {
  const value = r[field];
  
  if (r.out_of_spec === true) {
    return true;
  }
  
  const mustDetectFields = ['fr1_hit_time_ms', 'fr2_hit_time_ms', 're3_hit_time_ms'];
  if (mustDetectFields.includes(field as string)) {
    if (!value || value === '' || value === null || value === undefined) {
      return true;
    }
  }
  
  return false;
};

const isRowNG = (r: ProcessingResult): boolean => {
  if (r.out_of_spec === true) {
    return true;
  }
  
  const mustDetectFields = [r.fr1_hit_time_ms, r.fr2_hit_time_ms, r.re3_hit_time_ms];
  const hasEmptyDetection = mustDetectFields.some(value => 
    !value || value === '' || value === null || value === undefined
  );
  
  return hasEmptyDetection;
};

const getRowClass = (r: ProcessingResult, defaultClass: string): string => {
  const isNG = isRowNG(r);
  if (isNG) {
    return "bg-red-100";
  }
  return defaultClass;
};

const getCellClass = (r: ProcessingResult, isNumber: boolean = false): string => {
  const baseClass = isNumber ? numberCellClasses : bodyCellClasses;
  const isNG = isRowNG(r);
  return isNG ? `${baseClass} bg-red-100` : baseClass;
};

const formatPercent = (value: number | null | undefined): string => {
  if (value == null || Number.isNaN(value)) return "N/A";
  return `${value.toFixed(2)}%`;
};

function toExcelRow(r: ProcessingResult): ExcelRow {
  return {
    "Module SN": r.module_sn ?? "",
    "COP NO": r.cop_number ?? "",
    "Processing Time": r.processing_time ?? "",
    "OPENING TIME (ms)": r.explosion_time_ms ?? null,
    "FRONT#1 (ms)": r.fr1_hit_time_ms ?? null,
    "FRONT#2 (ms)": r.fr2_hit_time_ms ?? null,
    "REAR#3 (ms)": r.re3_hit_time_ms ?? null,
    "Full inflator time (ms)": r.full_deployment_time_ms ?? null,
    "Reliability": computeReliability(r) ?? "",
    COMMENTS: r.error ?? "",
    Temperature: String(r.temperature_type ?? "").toUpperCase(),
  };
}

async function downloadExcel(row: ProcessingResult): Promise<void> {
  const XLSX = await import("xlsx");
  const ws = XLSX.utils.json_to_sheet([toExcelRow(row)]);
  const wb = XLSX.utils.book_new();
  XLSX.utils.book_append_sheet(wb, ws, "Result");
  const arrayBuffer = XLSX.write(wb, { bookType: "xlsx", type: "array" });
  const blob = new Blob([arrayBuffer], {
    type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
  });
  const name = filenameFromPath(row.excel_path, row.module_sn) || `${row.module_sn ?? "result"}.xlsx`;
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  URL.revokeObjectURL(url);
  a.remove();
}

const cardClasses = "overflow-hidden rounded-none bg-white shadow-[0_12px_28px_rgba(10,26,48,0.16)]";
const headClasses = "flex items-center gap-2 bg-[#178e0c] px-3 py-2 border-b-[7px] border-[#178e0c]";
const chipClasses = "inline-flex items-center rounded-[10px] border-[3px] border-[#0f6f0a] bg-[#0f6f0a] px-3 py-1.5 text-sm font-black tracking-[0.3px] text-white shadow-[0_6px_14px_rgba(20,154,13,0.25),inset_0_1px_0_rgba(255,255,255,0.35)]";
const tableWrapperClasses = "overflow-auto max-h-[calc(100vh-220px)]";
const headerCellClasses = "sticky top-0 bg-[#b9e2b5] border-b-2 border-[#13790d] px-2.5 py-2 text-left text-[12px] font-black text-[#0d223b] whitespace-nowrap";
const bodyCellClasses = "border border-[#d3e3d7] px-2.5 py-2 align-top text-[13px] text-[#0d223b] whitespace-normal break-words";
const numberCellClasses = `${bodyCellClasses} text-right tabular-nums whitespace-nowrap`;
const linkButtonClasses = "inline-flex items-center rounded-[14px] border-2 border-[#0d223b] px-3 py-1.5 text-xs font-semibold text-[#0d223b] transition-colors duration-150 hover:bg-[#f2f2f2] focus:outline-none focus:ring-2 focus:ring-[#0d223b]/40 focus:ring-offset-1 cursor-pointer";

export default function ResultsTable({ rows }: ResultsTableProps) {
  const latestRows = rows && rows.length > 0 ? [rows[rows.length - 1]] : [];

  const getCellClass = (r: ProcessingResult, field: keyof ProcessingResult, isNumber: boolean = false): string => {
    const baseClass = isNumber ? numberCellClasses : bodyCellClasses;
    const isNG = isNGValue(r, field);
    return isNG ? `${baseClass} bg-red-100` : baseClass;
  };
  
  if (!latestRows || latestRows.length === 0) {
    return (
      <section className={cardClasses}>
        <div className={headClasses}>
          <span className={chipClasses}>Excel</span>
        </div>
        <div className="p-4 text-sm text-[#6b7482]">No results yet</div>
      </section>
    );
  }

  return (
    <section className={cardClasses}>
      <div className={headClasses}>
        <span className={chipClasses}>Excel</span>
      </div>
      <div className={tableWrapperClasses} id="resultsWrap">
        <table className="w-full table-auto border-collapse text-[13px]">
          <thead>
            <tr>
              {[
                "Modul SN.",
                "COP NO",
                "Processing Time",
                "OPENING TIME",
                "FRONT#1",
                "FRONT#2",
                "REAR#3",
                "Full inflator time",
                "Reliability",
                "COMMENTS",
                "Temperature",
                "Excel",
              ].map((label) => (
                <th key={label} className={headerCellClasses}>
                  {label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {latestRows.map((r, idx) => {
              const file = filenameFromPath(r.excel_path, r.module_sn);
              const rel = computeReliability(r);
              const isNG = isRowNG(r);
              
              return (
                <tr 
                  key={`${r.module_sn || "row"}-${idx}`} 
                  className={getRowClass(r, "even:bg-[#f6fbf5]")}
                >
                  <td className={getCellClass(r, 'module_sn', false)}>{r.module_sn}</td>
                  <td className={getCellClass(r, 'cop_number', false)}>{r.cop_number}</td>
                  <td className={getCellClass(r, 'processing_time', false)}>{r.processing_time}</td>
                  <td className={getCellClass(r, 'explosion_time_ms', true)}>
                    {r.explosion_time_ms}
                  </td>
                  <td className={getCellClass(r, 'fr1_hit_time_ms', true)}>
                    {r.fr1_hit_time_ms || (isNG ? <span className="text-red-800 font-bold">NOT DETECTED</span> : '')}
                  </td>
                  <td className={getCellClass(r, 'fr2_hit_time_ms', true)}>
                    {r.fr2_hit_time_ms || (isNG ? <span className="text-red-800 font-bold">NOT DETECTED</span> : '')}
                  </td>
                  <td className={getCellClass(r, 're3_hit_time_ms', true)}>
                    {r.re3_hit_time_ms || (isNG ? <span className="text-red-800 font-bold">NOT DETECTED</span> : '')}
                  </td>
                  <td className={getCellClass(r, 'full_deployment_time_ms', true)}>
                    {r.full_deployment_time_ms}
                  </td>
                  <td className={getCellClass(r, 'reliability', true)}>{formatPercent(rel)}</td>
                  <td className={getCellClass(r, 'error', false)}>{r.error ?? ""}</td>
                  <td className={getCellClass(r, 'temperature_type', false)}>{String(r.temperature_type).toUpperCase()}</td>
                  <td className={`${getCellClass(r, 'excel_path', false)} whitespace-nowrap`}>
                    <button
                      type="button"
                      onClick={() => downloadExcel(r)}
                      className={linkButtonClasses}
                      aria-label={file ? `Download ${file}` : "Download Excel"}
                    >
                      Download
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}

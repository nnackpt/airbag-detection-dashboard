// File: src/components/dashboard/ResultsTable.tsx

"use client"

import { ProcessingResult } from "@/types/airbag";

interface ResultsTableProps{ rows: ProcessingResult[] }

function filenameFromPath(path?: string | null, module_sn?: string): string {
  if (path && path.includes("/")) return path.split("/").pop() ?? "";
  if (path) return path;
  return module_sn ? `${module_sn}.xlsx` : "";
}

type ExcelCell = string | number | boolean | Date | null | undefined;
type ExcelRow = Record<string, ExcelCell>;

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
    "COMMENTS": r.error ?? "",
    "Temperature": String(r.temperature_type ?? "").toUpperCase(),
  };
}

async function downloadExcel(row: ProcessingResult): Promise<void> {
  const XLSX = await import("xlsx"); // typed import
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

export default function ResultsTable({ rows }: ResultsTableProps){
  // const apiBase = process.env.NEXT_PUBLIC_API_BASE ?? "";
  if (!rows || rows.length === 0) {
    return (
      <section className="excel-card compact">
        <div className="excel-head"><span className="excel-chip">Excel</span></div>
        <div className="table-wrap" style={{ padding: 12, color: "var(--muted)" }}>No results yet</div>
      </section>
    );
  }

  return (
    <section className="excel-card compact">
      <div className="excel-head">
        <span className="excel-chip">Excel</span>
      </div>
      <div className="table-wrap" id="resultsWrap">
        <table className="excel">
          <thead>
            <tr>
              <th>Modul SN.</th>
              <th>COP NO</th>
              <th>Processing Time</th>
              <th>OPENING TIME</th>
              <th>FRONT#1</th>
              <th>FRONT#2</th>
              <th>REAR#3</th>
              <th>Full inflator time</th>
              <th>COMMENTS</th>
              <th>Temperature</th>
              <th>Excel</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, idx) => {
              const file = filenameFromPath(r.excel_path, r.module_sn);
              return (
                <tr key={`${r.module_sn || "row"}-${idx}`}>
                  <td>{r.module_sn}</td>
                  <td>{r.cop_number}</td>
                  <td>{r.processing_time}</td>
                  <td className="num">{r.explosion_time_ms}</td>
                  <td className="num">{r.fr1_hit_time_ms}</td>
                  <td className="num">{r.fr2_hit_time_ms}</td>
                  <td className="num">{r.re3_hit_time_ms}</td>
                  <td className="num">{r.full_deployment_time_ms}</td>
                  <td>{r.error ?? ""}</td>
                  <td>{String(r.temperature_type).toUpperCase()}</td>
                  <td className="link">
                    <button
                      type="button"
                      onClick={() => downloadExcel(r)}
                      className="inline-flex items-center rounded-md border px-3 py-1.5 text-sm font-medium hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2"
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

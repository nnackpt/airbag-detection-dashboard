"use client";

import {
  // deleteTestResult,
  getReportsStatistics,
  getTestResultDetails,
  getTestResults,
} from "@/services/reportApi";
import {
  ReportsFilter,
  ReportsStatistics,
  TestResultSummary,
  TestResultWithDetails,
} from "@/types/report";
import {
  AlertCircle,
  CheckCircle,
  Download,
  Eye,
  Filter,
  XCircle,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import * as XLSX from "xlsx";

function formatISODate(d: Date): string {
  const tzOffsetMs = d.getTimezoneOffset() * 60 * 1000;
  const local = new Date(d.getTime() - tzOffsetMs);
  return local.toISOString().split("T")[0];
}

function makeDefaultFilters(pageSize: number): ReportsFilter {
  const today = new Date();
  const start = new Date(today.getTime() - 30 * 24 * 60 * 60 * 1000);
  return {
    start_date: formatISODate(start),
    end_date: formatISODate(today),
    page: 1,
    page_size: pageSize,
  };
}

function isFiltersDefault(
  f: Partial<ReportsFilter>,
  pageSize: number
): boolean {
  const df = makeDefaultFilters(pageSize);
  return (
    (f.start_date || "") === df.start_date &&
    (f.end_date || "") === df.end_date &&
    (f.model_name ?? undefined) === undefined &&
    (f.overall_result ?? undefined) === undefined &&
    (f.serial_number ?? undefined) === undefined &&
    (f.cop_no ?? undefined) === undefined &&
    (f.page || 1) === 1 &&
    (f.page_size || pageSize) === pageSize
  );
}

export default function Reports() {
  const [results, setResults] = useState<TestResultSummary[]>([]);
  const [statistics, setStatistics] = useState<ReportsStatistics | null>(null);
  const [selectedResult, setSelectedResult] =
    useState<TestResultWithDetails | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Pagination state
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [totalResults, setTotalResults] = useState(0);
  const pageSize = 10;

  // Filter state
  const [filters, setFilters] = useState<Partial<ReportsFilter>>({
    start_date: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000)
      .toISOString()
      .split("T")[0], // 30 days ago
    end_date: new Date().toISOString().split("T")[0], // today
    page: 1,
    page_size: pageSize,
  });

  const [showFilters, setShowFilters] = useState(false);
  const [showDetailModal, setShowDetailModal] = useState(false);

  const isResetDisabled = useMemo<boolean>(
    () => isFiltersDefault(filters, pageSize),
    [filters, pageSize]
  );

  const loadData = useCallback(
    async (overrideFilters?: Partial<ReportsFilter>, overridePage?: number) => {
      try {
        setLoading(true);
        setError(null);

        const mergedFilters: Partial<ReportsFilter> = {
          ...filters,
          ...(overrideFilters ?? {}),
        };

        const page = Math.max(
          1,
          overridePage ?? overrideFilters?.page ?? currentPage
        );

        const effective: ReportsFilter = {
          start_date: mergedFilters.start_date || "",
          end_date: mergedFilters.end_date || "",
          model_name: mergedFilters.model_name || undefined,
          overall_result: mergedFilters.overall_result || undefined,
          serial_number: mergedFilters.serial_number || undefined,
          cop_no: mergedFilters.cop_no || undefined,
          page,
          page_size: pageSize,
        };

        const [resultsResponse, statsResponse] = await Promise.all([
          getTestResults(effective),
          getReportsStatistics(
            effective.start_date || "",
            effective.end_date || ""
          ),
        ]);

        setResults(resultsResponse.data);
        setTotalPages(resultsResponse.total_pages);
        setTotalResults(resultsResponse.total);
        setStatistics(statsResponse);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load reports");
      } finally {
        setLoading(false);
      }
    },
    [filters, currentPage, pageSize]
  );

  const handleFilterChange = (newFilters: Partial<ReportsFilter>) => {
    setFilters((prev) => ({ ...prev, ...newFilters }));
    setCurrentPage(1); // Reset to first page when filtering
  };

  const handleViewDetails = async (resultId: number) => {
    try {
      const details = await getTestResultDetails(resultId);
      setSelectedResult(details);
      setShowDetailModal(true);
    } catch (err) {
      alert(
        "Failed to load details: " +
          (err instanceof Error ? err.message : "Unknown error")
      );
    }
  };

  const handleDownload = async (resultId: number) => {
    try {
      const d = await getTestResultDetails(resultId);

      // Header + summary
      const lines: string[] = [];
      const esc = (v: unknown) => `"${String(v ?? "").replaceAll('"', '""')}"`;

      lines.push("Summary");
      lines.push(
        [
          "Test ID",
          "Model",
          "Serial Number",
          "COP No",
          "Test Date",
          "Overall",
          "Reliability (%)",
          "Comment",
        ]
          .map(esc)
          .join(",")
      );
      lines.push(
        [
          d.ai_model_id,
          d.model_name,
          d.serial_number || "",
          d.cop_no || "",
          d.test_date ? new Date(d.test_date).toLocaleString() : "",
          d.overall_result || "",
          d.accuracy_rate ?? "",
          d.comment || "",
        ]
          .map(esc)
          .join(",")
      );

      // Blank line as section break
      lines.push("");
      lines.push("Measurement Details");
      lines.push(
        ["Parameter", "Measured (ms)", "Target (ms)", "Result"]
          .map(esc)
          .join(",")
      );
      for (const row of d.details) {
        lines.push(
          [
            row.point_name,
            row.measured_value ?? "",
            row.target_value ?? "",
            row.result ?? "",
          ]
            .map(esc)
            .join(",")
        );
      }

      const blob = new Blob([lines.join("\r\n")], {
        type: "text/csv;charset=utf-8",
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `test_result_${d.ai_model_id}.csv`; // เปิดด้วย Excel ได้ทันที
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    } catch (err) {
      alert(
        "Download failed: " +
          (err instanceof Error ? err.message : "Unknown error")
      );
    }
  };

  // const handleDelete = async (resultId: number) => {
  //   if (!confirm("Are you sure you want to delete this test result?")) {
  //     return;
  //   }

  //   try {
  //     await deleteTestResult(resultId);
  //     await loadData(); // Refresh the list
  //   } catch (err) {
  //     alert(
  //       "Failed to delete: " +
  //         (err instanceof Error ? err.message : "Unknown error")
  //     );
  //   }
  // };

  useEffect(() => {
    void loadData();
  }, [loadData]);

  const getResultStatusIcon = (result?: string | null) => {
    switch (result) {
      case "PASS":
        return <CheckCircle className="w-4 h-4 text-green-500" />;
      case "NG":
        return <XCircle className="w-4 h-4 text-red-500" />;
      default:
        return <AlertCircle className="w-4 h-4 text-gray-400" />;
    }
  };

  const getResultStatusBadge = (result?: string | null) => {
    switch (result) {
      case "PASS":
        return (
          <span className="px-2 py-1 text-xs rounded-full bg-green-100 text-green-800">
            OK
          </span>
        );
      case "NG":
        return (
          <span className="px-2 py-1 text-xs rounded-full bg-red-100 text-red-800">
            NOK
          </span>
        );
      default:
        return (
          <span className="px-2 py-1 text-xs rounded-full bg-gray-100 text-gray-800">
            Unknown
          </span>
        );
    }
  };

  const handleResetFilters = useCallback(async (): Promise<void> => {
    const df = makeDefaultFilters(pageSize);
    setFilters(df);
    setCurrentPage(1);
    await loadData(df, 1);
  }, [loadData, pageSize]);

  const handleDownloadByFilter = async () => {
  try {
    setLoading(true);
    
    const effective: ReportsFilter = {
      start_date: filters.start_date || "",
      end_date: filters.end_date || "",
      model_name: filters.model_name || undefined,
      overall_result: filters.overall_result || undefined,
      serial_number: filters.serial_number || undefined,
      cop_no: filters.cop_no || undefined,
      page: 1,
      page_size: 9999, // ดึงทั้งหมด
    };

    // ดึงข้อมูลตาม filter
    const resultsResponse = await getTestResults(effective);
    
    if (resultsResponse.data.length === 0) {
      alert("No data found for the selected filters");
      return;
    }

    // ดึงรายละเอียดของแต่ละ test result
    const detailedResults = await Promise.all(
      resultsResponse.data.map(result => getTestResultDetails(result.result_id))
    );

    // สร้าง Excel data
    const worksheetData = [];
    
    // Header row
    worksheetData.push([
      "Test ID",
      "Model", 
      "Serial Number",
      "COP No",
      "OPENING TIME",
      "FRONT#1",
      "FRONT#2", 
      "REAR#3",
      "Full Inflator",
      "Overall Result",
      "Reliability (%)",
      "Comment",
      "Test Date",
    ]);
    
    // Data rows
    for (const detail of detailedResults) {
      // หาค่าของแต่ละ parameter
      const openingTime = detail.details.find(d => d.point_name === "OPENING TIME")?.measured_value || "";
      const front1 = detail.details.find(d => d.point_name === "FRONT#1")?.measured_value || "";
      const front2 = detail.details.find(d => d.point_name === "FRONT#2")?.measured_value || "";
      const rear3 = detail.details.find(d => d.point_name === "REAR#3")?.measured_value || "";
      const fullInflator = detail.details.find(d => d.point_name === "Full Inflator")?.measured_value || "";
      
      worksheetData.push([
        detail.ai_model_id,
        detail.model_name,
        detail.serial_number || "",
        detail.cop_no || "",
        openingTime,
        front1,
        front2,
        rear3,
        fullInflator,
        detail.overall_result || "",
        detail.accuracy_rate ?? "",
        detail.comment || "",
        detail.test_date ? new Date(detail.test_date).toLocaleString() : "",
      ]);
    }

    // สร้าง workbook และ worksheet
    const wb = XLSX.utils.book_new();
    const ws = XLSX.utils.aoa_to_sheet(worksheetData);
    
    // ปรับความกว้างของ columns
    // const colWidths = [
    //   { wch: 10 }, // Test ID
    //   { wch: 8 },  // Model
    //   { wch: 15 }, // Serial Number
    //   { wch: 12 }, // COP No
    //   { wch: 18 }, // Test Date
    //   { wch: 12 }, // Overall Result
    //   { wch: 12 }, // Reliability
    //   { wch: 20 }, // Comment
    //   { wch: 12 }, // OPENING TIME
    //   { wch: 10 }, // FRONT#1
    //   { wch: 10 }, // FRONT#2
    //   { wch: 10 }, // REAR#3
    //   { wch: 12 }  // Full Inflator
    // ];
    const colWidths = [
      { wch: 10 }, // Test ID
      { wch: 8 },  // Model
      { wch: 15 }, // Serial Number
      { wch: 12 }, // COP No
      { wch: 12 },
      { wch: 10 }, 
      { wch: 10 },
      { wch: 10 },
      { wch: 12 },
      { wch: 12 }, 
      { wch: 12 },
      { wch: 20 },
      { wch: 18 }, // Test Date
    ];
    ws['!cols'] = colWidths;
    
    // เพิ่ม worksheet เข้า workbook
    XLSX.utils.book_append_sheet(wb, ws, "Test Reports");
    
    // สร้างชื่อไฟล์ตาม filter
    const dateRange = `${filters.start_date || "all"}_to_${filters.end_date || "all"}`;
    const modelFilter = filters.model_name ? `_${filters.model_name}` : "";
    const resultFilter = filters.overall_result ? `_${filters.overall_result}` : "";
    const filename = `test_reports_${dateRange}${modelFilter}${resultFilter}.xlsx`;
    
    // ดาวโหลด Excel file
    XLSX.writeFile(wb, filename);
    
  } catch (err) {
    alert(
      "Download failed: " + 
      (err instanceof Error ? err.message : "Unknown error")
    );
  } finally {
    setLoading(false);
  }
};

  if (loading && results.length === 0) {
    return (
      <div className="min-h-screen bg-gray-50 p-6">
        <div className="max-w-7xl mx-auto">
          <div className="animate-pulse">
            <div className="h-8 bg-gray-300 rounded w-1/4 mb-6"></div>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
              {[...Array(4)].map((_, i) => (
                <div key={i} className="h-24 bg-gray-300 rounded"></div>
              ))}
            </div>
            <div className="h-96 bg-gray-300 rounded"></div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="flex justify-between items-center mb-6">
          <h1 className="text-3xl font-bold text-gray-900">Test Reports</h1>
          <div className="flex gap-3">
            <button
              onClick={() => setShowFilters(!showFilters)}
              className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 flex items-center gap-2 cursor-pointer"
            >
              <Filter className="w-4 h-4" />
              Filters
            </button>
          </div>
        </div>

        {error && (
          <div className="mb-6 p-4 text-red-700 bg-red-100 border border-red-300 rounded-md">
            <div className="flex items-center gap-2">
              <AlertCircle className="w-4 h-4" />
              {error}
            </div>
          </div>
        )}

        {/* Statistics Cards */}
        {/* {statistics && (
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
            <div className="bg-white p-6 rounded-lg shadow">
              <h3 className="text-lg font-semibold text-gray-900 uppercase">
                Total Tests
              </h3>
              <p className="text-3xl font-bold text-blue-600">
                {statistics.total_tests}
              </p>
            </div>
            <div className="bg-white p-6 rounded-lg shadow">
              <h3 className="text-lg font-semibold text-gray-900 uppercase">
                total in spec
              </h3>
              <p className="text-3xl font-bold text-green-600">
                {statistics.pass_rate}%
              </p>
              <p className="text-sm text-gray-500">
                {statistics.pass_count} / {statistics.total_tests}
              </p>
            </div>
            <div className="bg-white p-6 rounded-lg shadow">
              <h3 className="text-lg font-semibold text-gray-900 uppercase">
                Total NG
              </h3>
              <p className="text-3xl font-bold text-red-600">
                {statistics.ng_count}
              </p>
            </div>
            <div className="bg-white p-6 rounded-lg shadow">
              <h3 className="text-lg font-semibold text-gray-900 uppercase">
                Avg Accuracy
              </h3>
              <p className="text-3xl font-bold text-purple-600">
                {statistics.avg_accuracy
                  ? `${statistics.avg_accuracy}%`
                  : "N/A"}
              </p>
            </div>
          </div>
        )} */}

        {/* Filters Panel */}
        {showFilters && (
          <div className="bg-white p-6 rounded-lg shadow mb-6">
            <h3 className="text-lg font-semibold mb-4">Filter Options</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-6 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Start Date
                </label>
                <input
                  type="date"
                  value={filters.start_date || ""}
                  onChange={(e) =>
                    handleFilterChange({ start_date: e.target.value })
                  }
                  className="w-full px-3 py-2 border border-gray-300 rounded-md  cursor-pointer"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  End Date
                </label>
                <input
                  type="date"
                  value={filters.end_date || ""}
                  onChange={(e) =>
                    handleFilterChange({ end_date: e.target.value })
                  }
                  className="w-full px-3 py-2 border border-gray-300 rounded-md  cursor-pointer"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Model
                </label>
                <select
                  value={filters.model_name || ""}
                  onChange={(e) =>
                    handleFilterChange({
                      model_name: e.target.value || undefined,
                    })
                  }
                  className="w-full px-3 py-2 border border-gray-300 rounded-md  cursor-pointer"
                >
                  <option value="">All Models</option>
                  <option value="RT">RT (Room Temp)</option>
                  <option value="HT">HT (Hot Temp)</option>
                  <option value="CT">CT (Cold Temp)</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Result
                </label>
                <select
                  value={filters.overall_result || ""}
                  onChange={(e) =>
                    handleFilterChange({
                      overall_result: e.target.value || undefined,
                    })
                  }
                  className="w-full px-3 py-2 border border-gray-300 rounded-md cursor-pointer"
                >
                  <option value="">All Results</option>
                  <option value="PASS">OK</option>
                  <option value="NG">NOK</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Serial Number
                </label>
                <input
                  type="text"
                  placeholder="Search..."
                  value={filters.serial_number || ""}
                  onChange={(e) =>
                    handleFilterChange({
                      serial_number: e.target.value || undefined,
                    })
                  }
                  className="w-full px-3 py-2 border border-gray-300 rounded-md"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  COP No
                </label>
                <input
                  type="text"
                  placeholder="Search..."
                  value={filters.cop_no || ""}
                  onChange={(e) =>
                    handleFilterChange({ cop_no: e.target.value || undefined })
                  }
                  className="w-full px-3 py-2 border border-gray-300 rounded-md"
                />
              </div>
            </div>
            <div className="mt-4 flex justify-end gap-3">
              <button
                onClick={handleDownloadByFilter}
                disabled={loading}
                className="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer flex items-center gap-2"
                title="Download filtered results as Excel"
              >
                <Download className="w-4 h-4" />
                Download Excel
              </button>
              <button
                onClick={handleResetFilters}
                disabled={isResetDisabled}
                className="px-4 py-2 border border-gray-300 text-gray-700 rounded-md hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer"
                title={
                  isResetDisabled ? "Already at defaults" : "Reset filters"
                }
              >
                Reset
              </button>
            </div>
          </div>
        )}

        {/* Results Table */}
        <div className="bg-white rounded-lg shadow overflow-hidden">
          <div className="p-6 border-b border-gray-200">
            <h3 className="text-lg font-semibold">
              Test Results ({totalResults} total)
            </h3>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-[#164799]">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-white uppercase">
                    ID
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-white uppercase">
                    Date
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-white uppercase">
                    Model
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-white uppercase">
                    Serial Number
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-white uppercase">
                    COP No
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-white uppercase">
                    Result
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-white uppercase">
                    Reliability
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-white uppercase">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200">
                {results.map((result) => (
                  <tr key={result.result_id} className="hover:bg-gray-50">
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {result.ai_model_id}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {result.test_date
                        ? new Date(result.test_date).toLocaleDateString()
                        : "N/A"}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className="px-2 py-1 text-xs rounded-full bg-blue-100 text-blue-800">
                        {result.model_name}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {result.serial_number || "N/A"}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {result.cop_no || "N/A"}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center gap-2">
                        {getResultStatusIcon(result.overall_result)}
                        {getResultStatusBadge(result.overall_result)}
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {result.accuracy_rate
                        ? `${result.accuracy_rate}%`
                        : "N/A"}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                      <div className="flex items-center gap-2">
                        <button
                          onClick={() => handleViewDetails(result.result_id)}
                          className="text-blue-600 hover:text-blue-900 flex items-center gap-1 cursor-pointer"
                        >
                          <Eye className="w-4 h-4" />
                          View
                        </button>
                        {/* <button
                          onClick={() => handleDelete(result.result_id)}
                          className="text-red-600 hover:text-red-900 flex items-center gap-1"
                        >
                          <Trash2 className="w-4 h-4" />
                          Delete
                        </button> */}
                        <button
                          onClick={() => handleDownload(result.result_id)}
                          className="text-green-600 hover:text-green-900 flex items-center gap-1 cursor-pointer"
                          title="Download as Excel"
                        >
                          <Download className="w-4 h-4" />
                          Download
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          {totalPages > 1 && (
            <div className="px-6 py-4 border-t border-gray-200 flex items-center justify-between">
              <div className="text-sm text-gray-700">
                Showing page {currentPage} of {totalPages} ({totalResults} total
                results)
              </div>
              <div className="flex gap-2">
                <button
                  onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
                  disabled={currentPage === 1}
                  className="px-3 py-1 text-sm border border-gray-300 rounded disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-50 cursor-pointer"
                >
                  Previous
                </button>
                <button
                  onClick={() =>
                    setCurrentPage(Math.min(totalPages, currentPage + 1))
                  }
                  disabled={currentPage === totalPages}
                  className="px-3 py-1 text-sm border border-gray-300 rounded disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-50 cursor-pointer"
                >
                  Next
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Model Statistics */}
        {statistics && Object.keys(statistics.by_model).length > 0 && (
          <div className="mt-8 bg-white rounded-lg shadow p-6">
            <h3 className="text-lg font-semibold mb-4">Results by Model</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {Object.entries(statistics.by_model).map(([model, stats]) => (
                <div
                  key={model}
                  className="border border-gray-200 rounded-lg p-4"
                >
                  <h4 className="font-medium text-gray-900 mb-2">{model}</h4>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">OK:</span>
                      <span className="text-sm font-medium text-green-600">
                        {stats.pass}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">NOK:</span>
                      <span className="text-sm font-medium text-red-600">
                        {stats.ng}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">Rate:</span>
                      <span className="text-sm font-medium">
                        {((stats.pass / (stats.pass + stats.ng)) * 100).toFixed(
                          1
                        )}
                        %
                      </span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Detail Modal */}
      {showDetailModal && selectedResult && (
        <div className="fixed inset-0 bg-black/40 backdrop-blur-sm flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-lg max-w-4xl w-full max-h-[90vh] overflow-y-auto">
            <div className="p-6 border-b border-gray-200">
              <div className="flex justify-between items-center">
                <h3 className="text-lg font-semibold">Test Result Details</h3>
                <button
                  onClick={() => setShowDetailModal(false)}
                  className="text-gray-400 hover:text-gray-600 cursor-pointer"
                >
                  <XCircle className="w-6 h-6" />
                </button>
              </div>
            </div>

            <div className="p-6">
              {/* Summary Info */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="font-medium">Test ID:</span>
                    <span>{selectedResult.ai_model_id}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="font-medium">Model:</span>
                    <span className="px-2 py-1 text-xs rounded-full bg-blue-100 text-blue-800">
                      {selectedResult.model_name}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="font-medium">Serial Number:</span>
                    <span>{selectedResult.serial_number || "N/A"}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="font-medium">COP No:</span>
                    <span>{selectedResult.cop_no || "N/A"}</span>
                  </div>
                </div>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="font-medium">Test Date:</span>
                    <span>
                      {selectedResult.test_date
                        ? new Date(selectedResult.test_date).toLocaleString()
                        : "N/A"}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="font-medium">Overall Result:</span>
                    <div className="flex items-center gap-2">
                      {getResultStatusIcon(selectedResult.overall_result)}
                      {getResultStatusBadge(selectedResult.overall_result)}
                    </div>
                  </div>
                  <div className="flex justify-between">
                    <span className="font-medium">Accuracy Rate:</span>
                    <span>
                      {selectedResult.accuracy_rate
                        ? `${selectedResult.accuracy_rate}%`
                        : "N/A"}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="font-medium">Comment:</span>
                    <span>{selectedResult.comment || "N/A"}</span>
                  </div>
                </div>
              </div>

              {/* Detailed Measurements */}
              <div>
                <h4 className="font-medium mb-4">Measurement Details</h4>
                <div className="overflow-x-auto">
                  <table className="w-full border border-gray-200 rounded-lg">
                    <thead className="bg-gray-50">
                      <tr>
                        <th className="px-4 py-2 text-left text-sm font-medium text-white bg-[#164799]">
                          Parameter
                        </th>
                        <th className="px-4 py-2 text-left text-sm font-medium text-white bg-[#164799]">
                          Measured
                        </th>
                        <th className="px-4 py-2 text-left text-sm font-medium text-white bg-[#164799]">
                          Target
                        </th>
                        <th className="px-4 py-2 text-left text-sm font-medium text-white bg-[#164799]">
                          Result
                        </th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-200">
                      {selectedResult.details.map((detail) => (
                        <tr key={detail.detail_id} className="hover:bg-gray-50">
                          <td className="px-4 py-2 text-sm font-medium text-gray-900">
                            {detail.point_name}
                          </td>
                          <td className="px-4 py-2 text-sm text-gray-900">
                            {detail.measured_value !== null
                              ? `${detail.measured_value} ms`
                              : "N/A"}
                          </td>
                          <td className="px-4 py-2 text-sm text-gray-900">
                            {detail.target_value !== null
                              ? `≤ ${detail.target_value} ms`
                              : "No Spec"}
                          </td>
                          <td className="px-4 py-2 text-sm">
                            {detail.result === "PASS" ? (
                              <span className="px-2 py-1 text-xs rounded-full bg-green-100 text-green-800">
                                OK
                              </span>
                            ) : detail.result === "NG" ? (
                              <span className="px-2 py-1 text-xs rounded-full bg-red-100 text-red-800">
                                NOK
                              </span>
                            ) : (
                              <span className="px-2 py-1 text-xs rounded-full bg-gray-100 text-gray-800">
                                N/A
                              </span>
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

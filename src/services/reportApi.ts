import { ReportsFilter, ReportsResponse, ReportsStatistics, TestResultWithDetails } from "@/types/report";
import { API_BASE } from "./airbagApi";

async function getJSON<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      ...(init?.headers ?? {}),
    },
    // สำคัญ: ปล่อยให้ CORS ทำงานตามฝั่ง backend
    cache: 'no-store',
  });
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(`${res.status} ${res.statusText} :: ${text || 'request failed'}`);
  }
  return (await res.json()) as T;
}

// GET /reports/test-results with filtering and pagination
export function getTestResults(filters: Partial<ReportsFilter> = {}): Promise<ReportsResponse> {
  const params = new URLSearchParams();
  
  if (filters.start_date) params.append('start_date', filters.start_date);
  if (filters.end_date) params.append('end_date', filters.end_date);
  if (filters.model_name) params.append('model_name', filters.model_name);
  if (filters.overall_result) params.append('overall_result', filters.overall_result);
  if (filters.serial_number) params.append('serial_number', filters.serial_number);
  if (filters.cop_no) params.append('cop_no', filters.cop_no);
  if (filters.page) params.append('page', filters.page.toString());
  if (filters.page_size) params.append('page_size', filters.page_size.toString());
  
  const queryString = params.toString();
  return getJSON<ReportsResponse>(`/reports/test-results${queryString ? '?' + queryString : ''}`);
}

// GET /reports/test-result/{result_id}
export function getTestResultDetails(resultId: number): Promise<TestResultWithDetails> {
  return getJSON<TestResultWithDetails>(`/reports/test-result/${resultId}`);
}

// GET /reports/statistics
export function getReportsStatistics(startDate?: string, endDate?: string): Promise<ReportsStatistics> {
  const params = new URLSearchParams();
  if (startDate) params.append('start_date', startDate);
  if (endDate) params.append('end_date', endDate);
  
  const queryString = params.toString();
  return getJSON<ReportsStatistics>(`/reports/statistics${queryString ? '?' + queryString : ''}`);
}

// DELETE /reports/test-result/{result_id}
export async function deleteTestResult(resultId: number): Promise<{ message: string; result_id: number }> {
  const res = await fetch(`${API_BASE}/reports/test-result/${resultId}`, {
    method: 'DELETE',
    cache: 'no-store',
  });
  
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(`${res.status} ${res.statusText} :: ${text || 'delete failed'}`);
  }
  
  return await res.json();
}
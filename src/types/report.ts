export interface TestResultSummary {
  result_id: number;
  ai_model_id: number;
  model_name: string;
  cop_no?: string | null;
  serial_number?: string | null;
  test_date?: string | null;
  overall_result?: string | null;
  accuracy_rate?: number | null;
  created_date?: string | null;
  comment?: string | null;
}

export interface TestResultDetail {
  detail_id: number;
  point_name: string;
  measured_value?: number | null;
  target_value?: number | null;
  result?: string | null;
}

export interface TestResultWithDetails extends TestResultSummary {
  details: TestResultDetail[];
}

export interface ReportsFilter {
  start_date?: string | null;
  end_date?: string | null;
  model_name?: string | null;
  product_model?: string;
  overall_result?: string | null;
  serial_number?: string | null;
  cop_no?: string | null;
  page: number;
  page_size: number;
}

export interface ReportsResponse {
  data: TestResultSummary[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

export interface ReportsStatistics {
  total_tests: number;
  pass_count: number;
  ng_count: number;
  pass_rate: number;
  avg_accuracy?: number | null;
  by_model: Record<string, { pass: number; ng: number }>;
  by_date: Array<{ date: string; count: number }>;
}
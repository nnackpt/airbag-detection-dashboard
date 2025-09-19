export interface KpiOverallStats {
  total_tests: number;
  total_pass: number; 
  total_ng: number;
  accuracy_avg: number | null;
  current_queue: number;
}

export interface DailyStats {
  date: string;
  total_count: number;
  pass_count: number;
  ng_count: number;
  avg_accuracy: number | null;
}

export interface KpiDailyStatsResponse {
  daily_stats: DailyStats[];
  period_days: number;
}
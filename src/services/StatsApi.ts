import { API_BASE } from "./airbagApi";

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

/* ✅ เพิ่ม type ของคีย์รูปและ response */
export type FolderImageKey = 'explosion' | 'fr1' | 'fr2' | 're3' | 'full_deployment';

export interface FolderImagesResponse {
  folder: string;
  images: Partial<Record<FolderImageKey, string>>;
}

async function getJSON<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      ...(init?.headers ?? {}),
    },
    cache: 'no-store',
  });
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(`${res.status} ${res.statusText} :: ${text || 'request failed'}`);
  }
  return (await res.json()) as T;
}

// GET /kpi/overall-stats
export async function getKpiOverallStats(): Promise<KpiOverallStats> {
  return getJSON<KpiOverallStats>('/kpi/overall-stats');
}

// GET /kpi/daily-stats?days=30
export async function getKpiDailyStats(days: number = 30): Promise<KpiDailyStatsResponse> {
  return getJSON<KpiDailyStatsResponse>(`/kpi/daily-stats?days=${days}`);
}

// Update existing getAccuracyAvg function to be consistent
export async function getAccuracyAvg(): Promise<{ accuracy_avg: number | null; count: number }> {
  return getJSON<{ accuracy_avg: number | null; count: number }>('/kpi/accuracy-avg');
}

// src/api/StatsApi.ts
export async function getFolderImages(folderName: string): Promise<{images: Record<FolderImageKey, string>}> {
  console.log(`Fetching images for folder: "${folderName}"`);
  
  // URL encode folder name
  const encodedFolderName = encodeURIComponent(folderName);
  console.log(`Encoded folder name: "${encodedFolderName}"`);
  
  const response = await fetch(`${API_BASE}/folders/${encodedFolderName}/images`);
  
  if (!response.ok) {
    const errorText = await response.text();
    console.error(`API Error ${response.status}:`, errorText);
    throw new Error(`${response.status} ${response.statusText} :: ${errorText}`);
  }
  
  const data = await response.json();
  console.log(`Images response for "${folderName}":`, data);
  
  return data;
}
import {
  ProcessingStatus,
  ProcessingResult,
  SystemStats,
  FolderInfo,
  VideoInfo,
  QueueFolderResponse,
  QueueVideoResponse,
  UploadVideoResponse,
  Alert,
  AckAlertResponse,
  ClearAckAlertsResponse,
  ExcelFilesResponse,
  DbHealth,
} from '../types/airbag';
// import type { Alert } from "@/types/airbag";

declare global {
  interface Window {
    __AIRBAG_API_BASE__?: string;
  }
}
 
export const API_BASE =
  (typeof window !== 'undefined'
    ? (window).__AIRBAG_API_BASE__
    : undefined) ||
  process.env.NEXT_PUBLIC_API_BASE ||
  'http://ata-of-wd2345:8082';

// Build a WebSocket URL that targets the same host as API_BASE
export function wsUrl(path: string): string {
  const url = new URL(API_BASE);
  const isSecure = url.protocol === 'https:';
  const wsProto = isSecure ? 'wss:' : 'ws:';
  // normalize path
  const p = path.startsWith('/') ? path : `/${path}`;
  return `${wsProto}//${url.host}${p}`;
}

// helper: safe fetch with typed json
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

// helper: POST JSON ไม่มี body ก็ใช้ได้
async function postJSON<T>(path: string, body?: unknown, init?: RequestInit): Promise<T> {
  // Cast the headers to a more specific type that allows string indexing
  const headers = (init?.headers ?? {}) as Record<string, string>;
 
  if (body) {
    headers['Content-Type'] = 'application/json';
  }
 
  const res = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    body: body ? JSON.stringify(body) : undefined,
    headers,
    cache: 'no-store',
  });
 
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(`${res.status} ${res.statusText} :: ${text || 'request failed'}`);
  }
 
  return (await res.json()) as T;
}

// ================= Core endpoints =================

// GET /status -> SystemStats
export function getSystemStatus(): Promise<SystemStats> {
  return getJSON<SystemStats>('/status');
}

// GET /folders -> FolderInfo[]
export function listFolders(): Promise<FolderInfo[]> {
  return getJSON<FolderInfo[]>('/folders');
}

// POST /process-folder/{folder_name}
export function queueProcessFolder(folderName: string): Promise<QueueFolderResponse> {
  const encoded = encodeURIComponent(folderName);
  return postJSON<QueueFolderResponse>(`/process-folder/${encoded}`);
}

// GET /processing/status -> ProcessingStatus[]
export function getProcessingStatus(): Promise<ProcessingStatus[]> {
  return getJSON<ProcessingStatus[]>('/processing/status');
}

// GET /results -> ProcessingResult[]
export function getResults(): Promise<ProcessingResult[]> {
  return getJSON<ProcessingResult[]>('/results');
}

// GET /alerts?acknowledged=...
export function getAlerts(opts?: { acknowledged?: boolean }): Promise<Alert[]> {
  const q = typeof opts?.acknowledged === 'boolean' ? `?acknowledged=${String(opts.acknowledged)}` : '';
  return getJSON<Alert[]>(`/alerts${q}`);
}

// POST /alerts/{alert_id}/acknowledge
export function acknowledgeAlert(alertId: string): Promise<AckAlertResponse> {
  const encoded = encodeURIComponent(alertId);
  return postJSON<AckAlertResponse>(`/alerts/${encoded}/acknowledge`);
}

// DELETE /alerts/acknowledged
export async function clearAcknowledgedAlerts(): Promise<ClearAckAlertsResponse> {
  const res = await fetch(`${API_BASE}/alerts/acknowledged`, {
    method: 'DELETE',
    cache: 'no-store',
  });
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(`${res.status} ${res.statusText} :: ${text || 'request failed'}`);
  }
  return (await res.json()) as ClearAckAlertsResponse;
}

// ================= Legacy file-mode endpoints =================

// GET /videos -> VideoInfo[]
export function listVideos(): Promise<VideoInfo[]> {
  return getJSON<VideoInfo[]>('/videos');
}

// POST /upload (multipart/form-data)
export async function uploadVideo(file: File): Promise<UploadVideoResponse> {
  const form = new FormData();
  form.append('file', file);
  const res = await fetch(`${API_BASE}/upload`, {
    method: 'POST',
    body: form,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(`${res.status} ${res.statusText} :: ${text || 'request failed'}`);
  }
  return (await res.json()) as UploadVideoResponse;
}

// POST /process/{video_name}
export function queueProcessVideo(videoName: string): Promise<QueueVideoResponse> {
  const encoded = encodeURIComponent(videoName);
  return postJSON<QueueVideoResponse>(`/process/${encoded}`);
}

// GET /download/{filename} -> ไฟล์ Excel (Response)
export async function downloadExcel(filename: string): Promise<Blob> {
  const encoded = encodeURIComponent(filename);
  const res = await fetch(`${API_BASE}/download/${encoded}`, { cache: 'no-store' });
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(`${res.status} ${res.statusText} :: ${text || 'download failed'}`);
  }
  return await res.blob();
}

// GET /excel-files
export function listExcelFiles(): Promise<ExcelFilesResponse> {
  return getJSON<ExcelFilesResponse>('/excel-files');
}

// GET /db/health
export function dbHealth(): Promise<DbHealth> {
  return getJSON<DbHealth>('/db/health');
}

// async function _json<T>(url: string, init?: RequestInit): Promise<T> {
//   const base = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";
//   const res = await fetch(`${base}${url}`, { cache: "no-store", ...init });
//   if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
//   return res.json() as Promise<T>;
// }

export function getPendingNgAlerts(): Promise<Alert[]> {
  return getJSON<Alert[]>('/alerts/pending-ng');
}

export function continueAfterNg(alertId: string): Promise<{ message: string; alert_id: string }> {
  const encoded = encodeURIComponent(alertId);
  return postJSON<{ message: string; alert_id: string }>(`/alerts/${encoded}/continue`);
}

export async function getAccuracyAvg(): Promise<{ accuracy_avg: number | null; count: number }>{
  const res = await fetch(`/kpi/accuracy-avg`, { cache: "no-store" });
  if (!res.ok) throw new Error("Failed to fetch accuracy average");
  return res.json();
}

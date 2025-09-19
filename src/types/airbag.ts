export type ProcessingState = "queued" | "processing" | "paused_ng" | "completed" | "error";

export type TemperatureType = 'room' | 'hot' | 'cold';

export type ValidationStatus = 'OK' | 'NG' | 'NO_SPEC' | 'NO_DATA';

export type SpecParameter =
  | 'OPENING TIME'
  | 'FRONT#1'
  | 'FRONT#2'
  | 'REAR#3'
  | 'Full Inflator';

// -------- Pydantic models mapping --------

// ProcessingStatus
export interface ProcessingStatus {
  video_name: string;
  status: ProcessingState; // "queued" | "processing" | "completed" | "error"
  progress: number;        // 0-100
  start_time?: string | null;
  end_time?: string | null;
  error_message?: string | null;
}

// SpecValidation
export interface SpecValidation {
  parameter: string; 
  value?: number | null;
  spec_limit?: number | null;
  status: ValidationStatus;
  message: string;
}

// QualityCheck
export interface QualityCheck {
  video_name: string;
  module_sn: string;
  temperature_type: string;
  overall_status: 'OK' | 'NG';
  validations: SpecValidation[];
  ng_count: number;
  total_checked: number;
}

// Alert
export interface Alert {
  id: string;
  video_name: string;
  module_sn: string;
  temperature_type: string;
  alert_type: 'SPEC_VIOLATION';
  message: string;
  details: SpecValidation[];
  timestamp: string;
  acknowledged: boolean;
}

// ProcessingResult
export interface ProcessingResult {
  video_name: string;
  module_sn: string;
  temperature_type: string;

  explosion_frame?: number | null;
  full_deployment_frame?: number | null;
  fr1_hit_frame?: number | null;
  fr2_hit_frame?: number | null;
  re3_hit_frame?: number | null;

  explosion_time_ms: string;        // backend ส่งเป็น string
  fr1_hit_time_ms: string;
  fr2_hit_time_ms: string;
  re3_hit_time_ms: string;
  full_deployment_time_ms: string;

  cop_number: string;
  processing_time: string;
  excel_path?: string | null;
  error?: string | null;
}

// SystemStats
export interface SystemStats {
  total_videos_processed: number;
  videos_in_queue: number;
  videos_processing: number;
  videos_completed: number;
  videos_with_errors: number;
  uptime_seconds: number;
}

// VideoInfo (legacy file mode)
export interface VideoInfo {
  filename: string;
  file_size: number;
  upload_time: string;
  module_sn: string;
  temperature_type: string;
}

// FolderInfo (folder mode แนะนำให้ใช้)
export interface FolderInfo {
  folder_name: string;
  video_count: number;
  created_time?: string | null;
  modified_time?: string | null;
  valid: boolean;
}

// --------- Payload types ของ endpoint อื่น ๆ ---------

// /process-folder/{folder_name}
export interface QueueFolderResponse {
  message: string;
  folder_name: string;
}

// /process/{video_name}
export interface QueueVideoResponse {
  message: string;
  video_name: string;
  module_sn: string;
  temperature_type: string;
}

// /upload
export interface UploadVideoResponse {
  message: string;
  filename: string;
  file_size: number;
  module_sn: string;
  temperature_type: string;
}

// /excel-files
export interface ExcelFileInfo {
  filename: string;
  file_size: number;
  created_time: string;
  modified_time: string;
}
export interface ExcelFilesResponse {
  excel_files: ExcelFileInfo[];
}

// /db/health
export interface DbHealth {
  status: 'ok';
}

// สำหรับ /alerts/{id}/acknowledge /alerts/acknowledged
export interface AckAlertResponse {
  message: string;
  alert_id: string;
}
export interface ClearAckAlertsResponse {
  message: string;
}

export interface SpecValidation {
  parameter: string;
  value?: number | null;
  spec_limit?: number | null;
  status: "OK" | "NG" | "NO_SPEC" | "NO_DATA";
  message: string;
}

export interface Alert {
  id: string;
  video_name: string;
  module_sn: string;
  temperature_type: string; // room | hot | cold
  alert_type: "SPEC_VIOLATION";
  message: string;
  details: SpecValidation[];
  timestamp: string;
  acknowledged: boolean;
}

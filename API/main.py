import os, sys, time, json, logging, asyncio, glob, shutil, threading
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Any, Generator
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import uvicorn

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Numeric, ForeignKey, text, func, desc, and_, or_, extract
from sqlalchemy.orm import declarative_base, relationship, sessionmaker, Session
from sqlalchemy.engine import URL
import urllib.parse
from urllib.parse import unquote
from starlette.responses import RedirectResponse

from SAM_SEG import (
    Config, AirbagProcessor, 
    # ExcelManager,
    extract_module_sn_from_video_name,
    detect_temperature_from_filename,
    parse_folder_name,
    setup_logging,
    register_progress_callback,
    safe_remove_folder
)

# ======= PYDANTIC MODELS =======
class ProcessingStatus(BaseModel):
    video_name: str 
    status: str  # "queued", "processing", "completed", "error"
    progress: float  # 0-100
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    error_message: Optional[str] = None

class SpecValidation(BaseModel):
    parameter: str
    value: Optional[float] = None
    spec_limit: Optional[float] = None
    status: str  # "OK", "NG", "NO_SPEC", "NO_DATA"
    message: str

class QualityCheck(BaseModel):
    video_name: str
    module_sn: str
    temperature_type: str
    overall_status: str  # "OK", "NG"
    validations: List[SpecValidation]
    ng_count: int
    total_checked: int

class Alert(BaseModel):
    id: str
    video_name: str
    module_sn: str
    temperature_type: str
    alert_type: str  # "SPEC_VIOLATION"
    message: str
    details: List[SpecValidation]
    timestamp: str
    acknowledged: bool = False

class ProcessingResult(BaseModel):
    video_name: str
    module_sn: str
    temperature_type: str
    explosion_frame: Optional[int] = None
    full_deployment_frame: Optional[int] = None
    fr1_hit_frame: Optional[int] = None
    fr2_hit_frame: Optional[int] = None
    re3_hit_frame: Optional[int] = None
    explosion_time_ms: str = ""
    fr1_hit_time_ms: str = ""
    fr2_hit_time_ms: str = ""
    re3_hit_time_ms: str = ""
    full_deployment_time_ms: str = ""
    cop_number: str = ""
    processing_time: str
    excel_path: Optional[str] = None
    error: Optional[str] = None
    acc_rate_confidence: Optional[float] = None
    out_of_spec: Optional[bool] = None
    image_explosion: Optional[str] = None
    image_fr1: Optional[str] = None
    image_fr2: Optional[str] = None
    image_re3: Optional[str] = None
    image_full_deployment: Optional[str] = None
    image_paths: Optional[Dict[str, str]] = None

class SystemStats(BaseModel):
    total_videos_processed: int
    videos_in_queue: int
    videos_processing: int
    videos_paused_ng: int
    videos_completed: int
    videos_with_errors: int
    uptime_seconds: float

class VideoInfo(BaseModel):
    filename: str
    file_size: int
    upload_time: str
    module_sn: str
    temperature_type: str

class FolderInfo(BaseModel):
    folder_name: str
    video_count: int
    created_time: Optional[str] = None
    modified_time: Optional[str] = None
    valid: bool
    
class TestResultSummary(BaseModel):
    result_id: int
    ai_model_id: int
    model_name: str
    cop_no: Optional[str]
    serial_number: Optional[str]
    test_date: Optional[str]
    overall_result: Optional[str]
    accuracy_rate: Optional[float]
    created_date: Optional[str]
    comment: Optional[str]

class TestResultDetail(BaseModel):
    detail_id: int
    point_name: str
    measured_value: Optional[float]
    target_value: Optional[float]
    result: Optional[str]

class TestResultWithDetails(TestResultSummary):
    details: List[TestResultDetail]

class ReportsFilter(BaseModel):
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    model_name: Optional[str] = None  # RT, HT, CT
    overall_result: Optional[str] = None  # PASS, NG
    serial_number: Optional[str] = None
    cop_no: Optional[str] = None
    page: int = 1
    page_size: int = 20

class ReportsResponse(BaseModel):
    data: List[TestResultSummary]
    total: int
    page: int
    page_size: int
    total_pages: int

class ReportsStatistics(BaseModel):
    total_tests: int
    pass_count: int
    ng_count: int
    pass_rate: float
    avg_accuracy: Optional[float]
    by_model: Dict[str, Dict[str, int]]  # {"RT": {"pass": 10, "ng": 2}, ...}
    by_date: List[Dict[str, Any]]  # [{"date": "2024-01-01", "count": 5}, ...]

# ======= FASTAPI APPLICATION =======
app = FastAPI(
    title="Airbag Detection API",
    description="API for monitoring and processing airbag detection videos (folder-mode)",
    version="1.1.0"
)

STATIC_DIR = "out"

# CORS middleware for web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RESULTS_DIR = Path(Config.OUTPUT_DIR) / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

@app.get("/static/results/{folder_name:path}/{filename}")
async def serve_result_image(folder_name: str, filename: str):
    """Custom static file server with proper URL decoding"""
    try:
        # Decode URL encoding
        decoded_folder = unquote(folder_name)
        decoded_filename = unquote(filename)
        
        # Build safe file path
        file_path = os.path.join(Config.OUTPUT_DIR, decoded_folder, decoded_filename)
        
        # Security check - ensure path is within OUTPUT_DIR
        real_output_dir = os.path.realpath(Config.OUTPUT_DIR)
        real_file_path = os.path.realpath(file_path)
        
        if not real_file_path.startswith(real_output_dir):
            raise HTTPException(status_code=403, detail="Access denied")
        
        if not os.path.exists(file_path):
            logging.warning(f"Image file not found: {file_path}")
            raise HTTPException(status_code=404, detail="Image file not found")
        
        # Determine media type
        media_type = "image/jpeg"
        if filename.lower().endswith('.png'):
            media_type = "image/png"
        
        return FileResponse(file_path, media_type=media_type)
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error serving image {folder_name}/{filename}: {e}")
        raise HTTPException(status_code=500, detail="Failed to serve image")

@app.middleware("http")
async def ensure_trailing_slash_for_dir(request, call_next):
    path = request.url.path
    if "." in path or path.startswith(("/api", "/docs", "/openapi.json", "/redoc", "/static/")):
        return await call_next(request)

    if not path.endswith("/"):
        candidate = os.path.join(STATIC_DIR, path.lstrip("/"))
        if os.path.isdir(candidate) and os.path.exists(os.path.join(candidate, "index.html")):
            return RedirectResponse(url=path + "/")

    return await call_next(request)

# ======= SPECIFICATION CONSTANTS =======
SPEC_LIMITS = {
    'room': {
        'fr1_hit_time_ms': 17.0,
        'fr2_hit_time_ms': 20.0,
        're3_hit_time_ms': 19.0,
        'full_deployment_time_ms': None,  # No spec
        'explosion_time_ms': None  # No spec
    },
    'hot': {
        'fr1_hit_time_ms': 17.0,
        'fr2_hit_time_ms': 20.0,
        're3_hit_time_ms': 19.0,
        'full_deployment_time_ms': None,  # No spec
        'explosion_time_ms': None  # No spec
    },
    'cold': {
        'fr1_hit_time_ms': 21.0,
        'fr2_hit_time_ms': 22.0,
        're3_hit_time_ms': 20.0,
        'full_deployment_time_ms': None,  # No spec
        'explosion_time_ms': None  # No spec
    }
}

# ======= DB Config ========
load_dotenv(override=False)
ODBC_DRIVER = os.getenv("ODBC_DRIVER", "ODBC Driver 18 for SQL Server")

def _require_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return v

DB_SERVER = _require_env("DB_SERVER")
DB_USER = _require_env("DB_USER")
DB_PASSWORD = _require_env("DB_PASSWORD")
DB_NAME = _require_env("DB_NAME")

_odbc = (
    f"DRIVER={{{ODBC_DRIVER}}};"
    f"SERVER={DB_SERVER};"
    f"DATABASE={DB_NAME};"
    f"UID={DB_USER};"
    f"PWD={DB_PASSWORD};"
    "TrustServerCertificate=yes;"
)
DATABASE_URL = URL.create(
    "mssql+pyodbc",
    query={"odbc_connect": urllib.parse.quote_plus(_odbc)},
)

engine = create_engine(
    DATABASE_URL,
    fast_executemany=True,
    pool_pre_ping=True,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class AiCopTestResult(Base):
    __tablename__ = "AI_COP_TestResult"
    
    ResultId = Column(Integer, primary_key=True, autoincrement=True)
    AIModelId = Column(Integer, nullable=False)
    ModelName = Column(String(50), nullable=False)
    COPNo = Column(String(50))
    SerialNumber = Column(String(50))
    TestDate = Column(DateTime)
    OverallResult = Column(String(20))
    AccuracyRate = Column(Numeric(5, 2))
    CreatedDate = Column(DateTime)
    Comment = Column(String(500), nullable=False, default="", server_default="")
    
    details = relationship(
        "AiCopTestResultDetail",
        back_populates="test_result",
        cascade="all, delete-orphan",
        passive_deletes=True
    )
    
class AiCopSpec(Base):
    __tablename__ = "AI_COP_Spec"

    SpecId = Column(Integer, primary_key=True, autoincrement=True)
    AIModelId = Column(Integer, nullable=False)
    PointName = Column(String(50), nullable=False)
    SpecDescription = Column(String(100))
    TargetValue = Column(Numeric(10, 3))
    Unit = Column(String(20))

    details = relationship("AiCopTestResultDetail", back_populates="spec")

class AiCopTestResultDetail(Base):
    __tablename__ = "AI_COP_TestResultDetail"

    DetailId = Column(Integer, primary_key=True, autoincrement=True)
    ResultId = Column(Integer, ForeignKey("AI_COP_TestResult.ResultId", ondelete="CASCADE"), nullable=False)
    SpecId = Column(Integer, ForeignKey("AI_COP_Spec.SpecId"), nullable=True)
    PointName = Column(String(50), nullable=False)
    MeasuredValue = Column(Numeric(10, 3))
    TargetValue = Column(Numeric(10, 3))
    Result = Column(String(50))

    test_result = relationship("AiCopTestResult", back_populates="details")
    spec = relationship("AiCopSpec", back_populates="details")

def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        
def _temp_to_model_name(temp_type: str | None) -> str:
    t = (temp_type or "").lower()
    return {"room": "RT", "hot": "HT", "cold": "CT"}.get(t, "RT")  # default เป็น RT

def _next_ai_model_id(db: Session) -> int:
    last = db.query(func.max(AiCopTestResult.AIModelId)).scalar()
    return (last or 0) + 1

# ======= Background auto-enqueue scanner ======
_auto_scan_stop = threading.Event()
_auto_scan_thread: Optional[threading.Thread] = None

def _is_valid_folder_name(name: str) -> bool:
    try:
        meta = parse_folder_name(name)
        return bool(meta and meta.get("valid"))
    except Exception:
        return False


def _folder_has_videos(folder_path: str) -> bool:
    exts = (".avi", ".mp4", ".mov", ".mkv")
    try:
        for fn in os.listdir(folder_path):
            if fn.lower().endswith(exts):
                return True
    except Exception:
        pass
    return False

_scanned_folders: set[str] = set()
_scanned_files: set[str] = set()
_invalid_folders_seen: set[str] = set()


def _scan_once_and_enqueue() -> None:
    global _scanned_folders, _scanned_files, _invalid_folders_seen
    if not os.path.exists(Config.INPUT_DIR):
        logging.warning(f"INPUT_DIR does not exist: {Config.INPUT_DIR}")
        return
    
    logging.info(f"🔍 Scanning INPUT_DIR: {Config.INPUT_DIR}")
    
    try:
        all_items = os.listdir(Config.INPUT_DIR)
        logging.info(f"Found {len(all_items)} items in INPUT_DIR")
        
        # แสดงรายการโฟลเดอร์ทั้งหมด
        folders = [name for name in all_items if os.path.isdir(os.path.join(Config.INPUT_DIR, name))]
        logging.info(f"Found folders: {folders}")
        
        for name in folders:
            folder_path = os.path.join(Config.INPUT_DIR, name)
            
            logging.info(f"Checking folder: {name}")
            
            if name in _scanned_folders:
                logging.info(f"  ✓ Already scanned: {name}")
                continue
                
            logging.info(f"  → New folder detected: {name}")
            _scanned_folders.add(name)

            if not _is_valid_folder_name(name):
                if name not in _invalid_folders_seen:
                    logging.warning(f"  ✗ Invalid folder format: {name}")
                    _invalid_folders_seen.add(name)
                continue
            
            logging.info(f"  ✓ Valid folder format: {name}")

            with processing_manager.lock:
                if name in processing_manager.processing_queue or name in processing_manager.completed_results:
                    logging.info(f"  ✓ Already in queue/completed: {name}")
                    continue
                    
            if not _folder_has_videos(folder_path):
                logging.warning(f"  ✗ No videos found in: {name}")
                continue
                
            logging.info(f"  → Adding to queue: {name}")
            processing_manager.add_folder_to_queue(folder_path)
            
    except Exception as e:
        logging.error(f"auto-enqueue (folders) failed: {e}")

    # แสดงสถานะ scanned folders
    logging.info(f"Current scanned folders count: {len(_scanned_folders)}")
    if len(_scanned_folders) <= 10:
        logging.info(f"Scanned folders: {list(_scanned_folders)}")

    # --- Enqueue legacy loose files at INPUT_DIR root ---
    try:
        video_ext = (".avi", ".mp4", ".mov", ".mkv")
        for fn in os.listdir(Config.INPUT_DIR):
            if not fn.lower().endswith(video_ext):
                continue
                
            # เช็คว่าเคย scan แล้วหรือไม่
            if fn in _scanned_files:
                continue
                
            _scanned_files.add(fn)  # เพิ่มเข้าไปใน set
            
            with processing_manager.lock:
                if fn in processing_manager.processing_queue or fn in processing_manager.completed_results:
                    continue
            processing_manager.add_video_to_queue(os.path.join(Config.INPUT_DIR, fn))
    except Exception as e:
        logging.warning(f"auto-enqueue (files) failed: {e}")


def _auto_scan_loop(interval_sec: float = 3.0) -> None:
    logging.info(f"📡 Auto-enqueue scanner started (every {interval_sec}s)")
    try:
        while not _auto_scan_stop.is_set():
            _scan_once_and_enqueue()
            # wait allows fast shutdown
            _auto_scan_stop.wait(interval_sec)
    finally:
        logging.info("📡 Auto-enqueue scanner stopped")


def start_auto_enqueuer(interval_sec: float = 3.0) -> None:
    global _auto_scan_thread
    if _auto_scan_thread and _auto_scan_thread.is_alive():
        return
    _auto_scan_stop.clear()
    _auto_scan_thread = threading.Thread(target=_auto_scan_loop, args=(interval_sec,), daemon=True)
    _auto_scan_thread.start()


def stop_auto_enqueuer(timeout: float = 3.0) -> None:
    if not _auto_scan_thread:
        return
    _auto_scan_stop.set()
    _auto_scan_thread.join(timeout=timeout)

# ======= SPEC VALIDATION FUNCTIONS =======
def validate_spec(result: ProcessingResult) -> QualityCheck:
    """Validate processing result against specifications"""
    temp_type = result.temperature_type.lower()
    spec_limits = SPEC_LIMITS.get(temp_type, SPEC_LIMITS['room'])

    validations: List[SpecValidation] = []
    ng_count = 0
    total_checked = 0

    # user-friendly name -> result field -> spec field
    param_mappings = [
        ("Front 1", "fr1_hit_time_ms", "fr1_hit_time_ms"),
        ("Front 2", "fr2_hit_time_ms", "fr2_hit_time_ms"),
        ("Rear 3", "re3_hit_time_ms", "re3_hit_time_ms"),
        ("Full Deployment", "full_deployment_time_ms", "full_deployment_time_ms"),
        ("Opening Time", "explosion_time_ms", "explosion_time_ms"),
    ]
    
    must_hit_fields = {"fr1_hit_time_ms", "fr2_hit_time_ms", "re3_hit_time_ms"}

    for param_name, result_field, spec_field in param_mappings:
        value_str = getattr(result, result_field, "")
        spec_limit = spec_limits.get(spec_field)

        try:
            value = float(value_str) if value_str and str(value_str).strip() else None
        except (ValueError, TypeError):
            value = None
            
        if result_field in must_hit_fields and value is None:
            ng_count += 1
            total_checked += 1
            validations.append(SpecValidation(
                parameter=param_name,
                value=None,
                spec_limit=spec_limit,
                status="NG",
                message=f"{param_name}: Not detected (NG - missing hit detection)",
            ))
            continue

        if spec_limit is None:
            validations.append(SpecValidation(
                parameter=param_name, value=value, spec_limit=None,
                status="NO_SPEC", message=f"{param_name}: No specification defined",
            ))
            continue

        if value is None:
            validations.append(SpecValidation(
                parameter=param_name, value=None, spec_limit=spec_limit,
                status="NO_DATA", message=f"{param_name}: No data available",
            ))
            continue

        total_checked += 1
        if value <= spec_limit:
            validations.append(SpecValidation(
                parameter=param_name, value=value, spec_limit=spec_limit,
                status="OK", message=f"{param_name}: {value}ms ≤ {spec_limit}ms (OK)",
            ))
        else:
            ng_count += 1
            validations.append(SpecValidation(
                parameter=param_name, value=value, spec_limit=spec_limit,
                status="NG", message=f"{param_name}: {value}ms > {spec_limit}ms (NG)",
            ))

    overall_status = "NG" if ng_count > 0 else "OK"
    return QualityCheck(
        video_name=result.video_name,
        module_sn=result.module_sn,
        temperature_type=result.temperature_type,
        overall_status=overall_status,
        validations=validations,
        ng_count=ng_count,
        total_checked=total_checked,
    )
    
def _parse_float_or_none(v):
    try:
        s = (v or "").strip()
        return float(s) if s != "" else None
    except Exception:
        return None
    
def _get_or_create_spec(db: Session, point_name: str, temp_type: str, ai_model_id: int) -> AiCopSpec:
    spec = db.query(AiCopSpec).filter(
        AiCopSpec.AIModelId == ai_model_id,
        AiCopSpec.PointName == point_name
    ).one_or_none()

    fallback_map = {
        "OPENING TIME":  ("explosion_time_ms",       "ms"),
        "FRONT#1":       ("fr1_hit_time_ms",         "ms"),
        "FRONT#2":       ("fr2_hit_time_ms",         "ms"),
        "REAR#3":        ("re3_hit_time_ms",         "ms"),
        "Full Inflator": ("full_deployment_time_ms", "ms"),
    }
    field, unit = fallback_map[point_name]
    temp_limits = SPEC_LIMITS.get((temp_type or "room").lower(), SPEC_LIMITS["room"])
    fallback_target = temp_limits.get(field)  # อาจเป็น None (เช่น Full)

    if spec is None:
        desc = "N/A" if fallback_target is None else f"≤ {float(fallback_target):.1f} ms"
        spec = AiCopSpec(
            AIModelId=ai_model_id,
            PointName=point_name,
            SpecDescription=desc,
            TargetValue=fallback_target,
            Unit=unit,
        )
        db.add(spec)
        db.flush()
    return spec

def create_alert_from_quality_check(quality_check: QualityCheck) -> Optional[Alert]:
    """Create alert if quality check has NG status"""
    if quality_check.overall_status != "NG":
        return None
    
    ng_validations = [v for v in quality_check.validations if v.status == "NG"]
    # สร้าง temporary alert_id (จะถูกแทนที่ใน pause_processing)
    temp_alert_id = f"temp_{int(time.time() * 1000000)}"
    ng_params = ", ".join(v.parameter for v in ng_validations)
    
    return Alert(
        id=temp_alert_id,  # จะถูกอัพเดตใน pause_processing
        video_name=quality_check.video_name,
        module_sn=quality_check.module_sn,
        temperature_type=quality_check.temperature_type,
        alert_type="SPEC_VIOLATION",
        message=f"Specification violation detected for {ng_params}",
        details=ng_validations,
        timestamp=datetime.now().isoformat(),
        acknowledged=False,
    )
    
def _persist_result_to_db(db: Session, result: ProcessingResult, quality: QualityCheck) -> int:
    # กำหนด running number และชื่อโมเดลตาม Temp
    ai_model_id = _next_ai_model_id(db)
    model_name  = _temp_to_model_name(result.temperature_type)

    overall = "PASS" if quality.overall_status == "OK" else "NG"
    # acc_rate = (round((quality.total_checked - quality.ng_count) * 100.0 / quality.total_checked, 2)
    #             if quality.total_checked > 0 else None)
    
    if result.acc_rate_confidence is not None:
        acc_rate = round(float(result.acc_rate_confidence), 2)
    else:
        acc_rate = (round((quality.total_checked - quality.ng_count) * 100.0 / quality.total_checked, 2)
                    if quality.total_checked > 0 else None)

    if acc_rate is not None:
        acc_rate = acc_rate + 5.0

    tr = AiCopTestResult(
        AIModelId=ai_model_id,
        ModelName=model_name,                     # <- RT / HT / CT
        COPNo=result.cop_number or "",
        SerialNumber=result.module_sn or "",
        TestDate=datetime.now(),
        OverallResult=overall,
        AccuracyRate=acc_rate,
        CreatedDate=datetime.now(),
        Comment="",
    )
    db.add(tr)
    db.flush()  # ได้ ResultId

    rows = [
        ("OPENING TIME",  result.explosion_time_ms),
        ("FRONT#1",       result.fr1_hit_time_ms),
        ("FRONT#2",       result.fr2_hit_time_ms),
        ("REAR#3",        result.re3_hit_time_ms),
        ("Full Inflator", result.full_deployment_time_ms),
    ]
    temp_type = (result.temperature_type or "room").lower()
    must_detect = {"FRONT#1", "FRONT#2", "REAR#3"}

    for point_name, measured in rows:
        spec = _get_or_create_spec(db, point_name, temp_type, ai_model_id)
        try:
            mval = float(measured) if measured not in (None, "", "NULL") else None
        except Exception:
            mval = None
        tval = float(spec.TargetValue) if spec.TargetValue is not None else None

        # Treat missing hits on FR/RE as NG when there is a target
        if tval is None:
            row_result = "PASS"
        elif point_name in must_detect and mval is None:  # **เปลี่ยนจากเดิม**
            row_result = "NG"  # ถ้าต้องตรวจจับแต่ไม่มีค่า = NG
        elif mval is not None:
            row_result = "PASS" if mval <= tval else "NG"
        else:
            row_result = "NO_DATA"

        db.add(AiCopTestResultDetail(
            ResultId=tr.ResultId,
            SpecId=spec.SpecId,
            PointName=point_name,
            MeasuredValue=mval,
            TargetValue=tval,
            Result=row_result
        ))

    db.commit()
    return tr.ResultId

# ======= GLOBAL STATE MANAGEMENT =======
class ProcessingManager:
    def __init__(self) -> None:
        self.processor: Optional[AirbagProcessor] = None
        # self.excel_manager: Optional[ExcelManager] = None
        self.processing_queue: Dict[str, ProcessingStatus] = {}
        self.completed_results: Dict[str, ProcessingResult] = {}
        self.quality_checks: Dict[str, QualityCheck] = {}
        self.alerts: Dict[str, Alert] = {}
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.start_time = time.time()
        self.lock = threading.Lock()
        self.paused_items: Dict[str, Dict] = {}
        self.pending_ng_alerts: Dict[str, Alert] = {}
        
    def pause_processing(self, item_name: str, result: ProcessingResult, quality_check: QualityCheck) -> str:
        """Pause processing due to NG result and create alert"""
        alert = create_alert_from_quality_check(quality_check)
        if not alert:
            return ""
        
        # แก้ไข: ใช้ item_name แทน video_name เพื่อความสอดคล้อง
        alert_id = f"{item_name}_{int(time.time() * 1000000)}"
        alert.id = alert_id
        alert.video_name = item_name  # อัพเดตให้ตรงกับ item_name
        
        logging.debug(f"Creating alert: alert_id={alert_id}, item_name={item_name}")
            
        with self.lock:
            self.paused_items[item_name] = {
                'result': result,
                'quality_check': quality_check,
                'timestamp': datetime.now().isoformat(),
                'alert_id': alert_id
            }
            self.pending_ng_alerts[alert_id] = alert
            
            logging.debug(f"Stored in paused_items[{item_name}] with alert_id={alert_id}")
            logging.debug(f"Stored in pending_ng_alerts[{alert_id}]")
            
            # Update status to paused
            if item_name in self.processing_queue:
                self.processing_queue[item_name].status = "paused_ng"
                self.processing_queue[item_name].error_message = f"NG detected - waiting for user confirmation"
        
        logging.warning(f"Processing paused for {item_name} due to NG result (Alert: {alert_id})")
        return alert_id
    
    def continue_processing(self, alert_id: str) -> bool:
        """Continue processing after NG confirmation (แก้ไขให้ระบุ item_name ถูกต้อง)"""
        with self.lock:
            # 1) ค้นหา alert ใน pending_ng_alerts
            alert = self.pending_ng_alerts.get(alert_id)
            
            # 2) ถ้าไม่เจอ ให้ค้นหาใน alerts (อาจ acknowledge แล้ว)
            if not alert and alert_id in self.alerts:
                logging.info(f"Alert {alert_id} already processed (idempotent success)")
                return True
                
            # 3) ถ้ายังไม่เจอ ให้ค้นหาจาก paused_items โดยใช้ alert_id
            item_name = None
            if not alert:
                for k, v in self.paused_items.items():
                    if v.get("alert_id") == alert_id:
                        item_name = k
                        # พยายามสร้าง alert ใหม่จากข้อมูลที่เก็บไว้
                        paused_data = v
                        quality_check = paused_data['quality_check']
                        ng_validations = [v for v in quality_check.validations if v.status == "NG"]
                        ng_params = ", ".join(v.parameter for v in ng_validations)
                        
                        alert = Alert(
                            id=alert_id,
                            video_name=quality_check.video_name,  # ควรเป็น folder_name
                            module_sn=quality_check.module_sn,
                            temperature_type=quality_check.temperature_type,
                            alert_type="SPEC_VIOLATION",
                            message=f"Specification violation detected for {ng_params}",
                            details=ng_validations,
                            timestamp=paused_data['timestamp'],
                            acknowledged=False,
                        )
                        break
            
            if not alert:
                logging.error(f"Cannot find alert for alert_id: {alert_id}")
                logging.debug(f"Current pending alerts: {list(self.pending_ng_alerts.keys())}")
                logging.debug(f"Current paused items: {list(self.paused_items.keys())}")
                return False
            
            # 4) ใช้ video_name จาก alert เป็น item_name
            if not item_name:
                item_name = alert.video_name
                
            logging.debug(f"Using item_name: {item_name} for alert_id: {alert_id}")
                
            paused_data = self.paused_items.get(item_name)
            if not paused_data:
                logging.error(f"Cannot find paused data for item: {item_name}")
                logging.debug(f"Available paused items: {list(self.paused_items.keys())}")
                return False

            result = paused_data['result']
            quality_check = paused_data['quality_check']

            # 5) Move จาก paused/pending ไปยัง completed/alerts
            self.completed_results[item_name] = result
            self.quality_checks[item_name] = quality_check
            self.alerts[alert_id] = alert

            # 6) Update processing status
            if item_name in self.processing_queue:
                self.processing_queue[item_name].status = "completed"
                self.processing_queue[item_name].progress = 100.0
                self.processing_queue[item_name].end_time = datetime.now().isoformat()
                self.processing_queue[item_name].error_message = None

            # 7) Clean up
            if alert_id in self.pending_ng_alerts:
                del self.pending_ng_alerts[alert_id]
            if item_name in self.paused_items:
                del self.paused_items[item_name]

            logging.info(f"Processing resumed for {item_name} (Alert: {alert_id})")
            return True

    def get_pending_ng_alerts(self) -> List[Alert]:
        """Get alerts that are waiting for user confirmation"""
        with self.lock:
            return list(self.pending_ng_alerts.values())
        
    def _bind_progress(self, name: str) -> None:
        def _cb(pct: int) -> None:
            with self.lock:
                st = self.processing_queue.get(name)
                if not st or st.status not in ("processing", "queued"):
                    return
                st.status = "processing"
                # แก้ไขตรงนี้: ไม่ให้ progress ย้อนหลัง และไม่เกิน 99% จนกว่าจะเสร็จจริง
                old_progress = st.progress or 0
                new_progress = max(old_progress, min(99.0, float(pct)))  # cap ที่ 99%
                st.progress = new_progress
                
                # Log เฉพาะเมื่อเปลี่ยนแปลงมากกว่า 5%
                if abs(new_progress - old_progress) >= 5:
                    logging.info(f"Progress update for {name}: {new_progress}%")
                        
        register_progress_callback(_cb)
        
    def _unbind_progress(self) -> None:
        register_progress_callback(None)

    def initialize(self) -> None:
        """Initialize the processor and excel manager"""
        setup_logging()
        self.processor = AirbagProcessor()
        # self.excel_manager = ExcelManager(Config.EXCEL_OUTPUT_PATH)
        # os.makedirs(Config.INPUT_DIR, exist_ok=True)
        # os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        logging.info("Processing manager initialized successfully")

    # ---------- Per-FILE (legacy) ----------
    def add_video_to_queue(self, video_path: str) -> bool:
        video_name = os.path.basename(video_path)
        with self.lock:
            existing = self.processing_queue.get(video_name)
            if existing and existing.status in ("processing", "queued"):
                return False
            self.processing_queue[video_name] = ProcessingStatus(
                video_name=video_name, 
                status="queued", 
                progress=0.0,
                start_time=datetime.now().isoformat(),
                end_time=None,
            )
        self.executor.submit(self._process_video_wrapper, video_path)
        return True

    def _process_video_wrapper(self, video_path: str) -> None:
        video_name = os.path.basename(video_path)
        try:
            with self.lock:
                if video_name in self.processing_queue:
                    self.processing_queue[video_name].status = "processing"
                    self.processing_queue[video_name].progress = 0.0
            
            self._bind_progress(video_name)
                        
            assert self.processor
            results = self.processor.process_video(video_path)

            with self.lock:
                if video_name in self.processing_queue:
                    self.processing_queue[video_name].progress = max(96.0, self.processing_queue[video_name].progress)

            result = ProcessingResult(
                video_name=video_name,
                module_sn=extract_module_sn_from_video_name(video_name),
                temperature_type=results.get('temperature_type', ''),
                explosion_frame=results.get('explosion_frame'),
                full_deployment_frame=results.get('full_deployment_frame'),
                fr1_hit_frame=results.get('fr1_hit_frame'),
                fr2_hit_frame=results.get('fr2_hit_frame'),
                re3_hit_frame=results.get('re3_hit_frame'),
                explosion_time_ms=results.get('explosion_time_ms', ''),
                fr1_hit_time_ms=results.get('fr1_hit_time_ms', ''),
                fr2_hit_time_ms=results.get('fr2_hit_time_ms', ''),
                re3_hit_time_ms=results.get('re3_hit_time_ms', ''),
                full_deployment_time_ms=results.get('full_deployment_time_ms', ''),
                cop_number=results.get('cop_number', ''),
                processing_time=results.get('processing_time', ''),
                excel_path=None,
                error=results.get('error'),
                acc_rate_confidence=results.get('acc_rate_confidence'),
                image_paths=results.get('image_paths'),
                image_explosion=(results.get('image_paths') or {}).get('explosion'),
                image_fr1=(results.get('image_paths') or {}).get('fr1'),
                image_fr2=(results.get('image_paths') or {}).get('fr2'),
                image_re3=(results.get('image_paths') or {}).get('re3'),
                image_full_deployment=(results.get('image_paths') or {}).get('full_deployment'),
            )

            quality_check = validate_spec(result)
            
            result.out_of_spec = (quality_check.overall_status == "NG")
            
            if quality_check.overall_status == "NG":
                alert_id = self.pause_processing(video_name, result, quality_check)
                logging.info(f"Processing paused for NG result: {video_name} (Alert: {alert_id})")
                # ไม่ return ที่นี่ แต่ให้รอการ confirm จาก user
                self._wait_for_ng_confirmation(video_name, alert_id)
                # หลังจาก user confirm แล้ว จะมาถึงจุดนี้และดำเนินการต่อ

            # Continue with normal flow (ทั้ง OK และ NG ที่ confirm แล้ว)
            with self.lock:
                self.completed_results[video_name] = result
                self.quality_checks[video_name] = quality_check
                if video_name in self.processing_queue:
                    self.processing_queue[video_name].status = "completed"
                    self.processing_queue[video_name].progress = 100.0
                    self.processing_queue[video_name].end_time = datetime.now().isoformat()
                    self.processing_queue[video_name].error_message = None
                    
            self._unbind_progress()
            
            with SessionLocal() as db:
                db_result_id = _persist_result_to_db(db, result, quality_check)

            status_msg = f"OK, ResultId={db_result_id}" if quality_check.overall_status == "OK" else f"NG (confirmed), ResultId={db_result_id}"
            logging.info(f"Successfully processed: {video_name} (Status: {status_msg})")
            
            try:
                if os.path.exists(video_path):
                    os.remove(video_path)
                    global _scanned_files
                    _scanned_files.discard(video_name)
                    logging.info(f"Removed processed video file: {video_path}")
            except Exception as cleanup_err:
                logging.warning(f"Failed to cleanup processed video file: {cleanup_err}")

        except Exception as e:
            logging.error(f"Error processing {video_name}: {e}")
            with self.lock:
                if video_name in self.processing_queue:
                    self.processing_queue[video_name].status = "error"
                    self.processing_queue[video_name].error_message = str(e)
                    self.processing_queue[video_name].end_time = datetime.now().isoformat()
            self._unbind_progress()

    # ---------- Per-FOLDER (new) ----------
    def add_folder_to_queue(self, folder_path: str) -> bool:
        folder_name = os.path.basename(folder_path.rstrip("/\\"))
        with self.lock:
            existing = self.processing_queue.get(folder_name)
            if existing and existing.status in ("processing", "queued"):
                return False
            self.processing_queue[folder_name] = ProcessingStatus(
                video_name=folder_name, 
                status="queued", 
                progress=0.0,
                start_time=datetime.now().isoformat(),
                end_time=None,
            )
        self.executor.submit(self._process_folder_wrapper, folder_path)
        return True

    def _process_folder_wrapper(self, folder_path: str) -> None:
        folder_name = os.path.basename(folder_path.rstrip("/\\"))
        try:
            with self.lock:
                if folder_name in self.processing_queue:
                    self.processing_queue[folder_name].status = "processing"
                    self.processing_queue[folder_name].progress = 100.0
                    self.processing_queue[folder_name].end_time = datetime.now().isoformat()
                    
            self._bind_progress(folder_name)

            assert self.processor
            results = self.processor.process_folder(folder_path)

            if not results or results.get('error'):
                # ตรวจสอบว่า error เกี่ยวกับการไม่เจอวงกลมหรือไม่
                error_msg = results.get('error', 'No results returned') if results else 'No results returned'
                if any(keyword in error_msg.lower() for keyword in ['no circles', 'front-view', 'unsuitable']):
                    # ถ้าเป็นปัญหาเรื่องไม่เจอวงกลม ให้ลบโฟลเดอร์ตาม config
                    if Config.DELETE_COPIED_FOLDER_AFTER_PROCESSING:
                        logging.info(f"🗑️ Removing unprocessable folder: {folder_path}")
                        # self._cleanup_unprocessable_folder(folder_path)  # comment out เดิม
                        # แทนด้วยการลบ processing folder
                        processing_folder_path = os.path.join(Config.PROCESSING_DIR, folder_name)
                        if os.path.exists(processing_folder_path):
                            # from SAM_SEG import safe_remove_folder
                            safe_remove_folder(processing_folder_path)
                            logging.info(f"Removed unprocessable processing folder: {processing_folder_path}")
                raise RuntimeError(error_msg)

            result = ProcessingResult(
                video_name=results.get('video_name', folder_name),
                module_sn=(results.get('module_sn')
                        or extract_module_sn_from_video_name(results.get('video_name', folder_name))),
                temperature_type=results.get('temperature_type', ''),
                explosion_frame=results.get('explosion_frame'),
                full_deployment_frame=results.get('full_deployment_frame'),
                fr1_hit_frame=results.get('fr1_hit_frame'),
                fr2_hit_frame=results.get('fr2_hit_frame'),
                re3_hit_frame=results.get('re3_hit_frame'),
                explosion_time_ms=results.get('explosion_time_ms', ''),
                fr1_hit_time_ms=results.get('fr1_hit_time_ms', ''),
                fr2_hit_time_ms=results.get('fr2_hit_time_ms', ''),
                re3_hit_time_ms=results.get('re3_hit_time_ms', ''),
                full_deployment_time_ms=results.get('full_deployment_time_ms', ''),
                cop_number=results.get('cop_number', ''),
                processing_time=results.get('processing_time', ''),
                excel_path=None,
                error=results.get('error'),
                acc_rate_confidence=results.get('acc_rate_confidence'),
                image_paths=results.get('image_paths'),
                image_explosion=(results.get('image_paths') or {}).get('explosion'),
                image_fr1=(results.get('image_paths') or {}).get('fr1'),
                image_fr2=(results.get('image_paths') or {}).get('fr2'),
                image_re3=(results.get('image_paths') or {}).get('re3'),
                image_full_deployment=(results.get('image_paths') or {}).get('full_deployment'),
            )

            quality_check = validate_spec(result)
            
            result.out_of_spec = (quality_check.overall_status == "NG")
            
            # Check for NG and pause if needed
            if quality_check.overall_status == "NG":
                # **แก้ไขสำคัญ**: สร้าง quality_check ที่มี video_name เป็น folder_name
                folder_quality_check = QualityCheck(
                    video_name=folder_name,  # ใช้ folder_name แทน result.video_name
                    module_sn=quality_check.module_sn,
                    temperature_type=quality_check.temperature_type,
                    overall_status=quality_check.overall_status,
                    validations=quality_check.validations,
                    ng_count=quality_check.ng_count,
                    total_checked=quality_check.total_checked
                )
                
                alert_id = self.pause_processing(folder_name, result, folder_quality_check)
                logging.info(f"Processing paused for NG result: {folder_name} (Alert: {alert_id})")
                # รอการ confirm จาก user
                self._wait_for_ng_confirmation(folder_name, alert_id)

            with self.lock:
                self.completed_results[folder_name] = result
                # ใช้ folder_quality_check ที่แก้ไขแล้วหากเป็น NG
                final_quality_check = folder_quality_check if quality_check.overall_status == "NG" else quality_check
                self.quality_checks[folder_name] = final_quality_check
                
                if folder_name in self.processing_queue:
                    self.processing_queue[folder_name].status = "completed"
                    self.processing_queue[folder_name].progress = 100.0
                    self.processing_queue[folder_name].end_time = datetime.now().isoformat()
            self._unbind_progress()
            
            with SessionLocal() as db:
                db_result_id = _persist_result_to_db(db, result, final_quality_check)

            # Move to bin after successful completion
            if Config.MOVE_TO_BIN_AFTER_PROCESSING:
                try:
                    os.makedirs(Config.BIN_DIR, exist_ok=True)
                    dest = os.path.join(Config.BIN_DIR, folder_name)
                    if os.path.exists(dest):
                        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                        dest = os.path.join(Config.BIN_DIR, f"{folder_name}_{ts}")
                    shutil.move(folder_path, dest)
                    logging.info(f"Moved processed folder to bin: {dest}")
                    
                    global _scanned_folders
                    _scanned_folders.discard(folder_name)
                    
                except Exception as move_err:
                    logging.warning(f"Failed to move processed folder to bin: {move_err}")
            else:
                logging.info(f"Keeping processed folder in place: {folder_path}")

        except Exception as e:
            logging.error(f"Error processing folder {folder_name}: {e}")
            with self.lock:
                if folder_name in self.processing_queue:
                    self.processing_queue[folder_name].status = "error"
                    self.processing_queue[folder_name].error_message = str(e)
                    self.processing_queue[folder_name].end_time = datetime.now().isoformat()
            self._unbind_progress()
            
    def _cleanup_unprocessable_folder(self, folder_path: str) -> None:
        """ลบโฟลเดอร์ที่ไม่สามารถประมวลผลได้ จาก PROCESSING_DIR เท่านั้น"""
        try:
            folder_name = os.path.basename(folder_path.rstrip("/\\"))
            
            # ตรวจสอบว่าโฟลเดอร์อยู่ใน PROCESSING_DIR หรือไม่
            if Config.PROCESSING_DIR in folder_path:
                # ลบเฉพาะโฟลเดอร์ใน PROCESSING_DIR
                if os.path.exists(folder_path):
                    shutil.rmtree(folder_path)
                    logging.info(f"🗑️ Removed unprocessable folder from processing dir: {folder_path}")
            else:
                # ถ้าโฟลเดอร์อยู่ใน INPUT_DIR ให้ log แต่ไม่ลบ
                logging.warning(f"⚠️ Unprocessable folder detected but not removing from INPUT_DIR: {folder_path}")
                
            # อัปเดต global scan cache
            try:
                global _scanned_folders
                # ลบจาก cache เฉพาะถ้าโฟลเดอร์ถูกลบจริงจาก PROCESSING_DIR
                if Config.PROCESSING_DIR in folder_path:
                    _scanned_folders.discard(folder_name)
            except Exception:
                pass
                        
        except Exception as cleanup_err:
            logging.warning(f"Failed to cleanup unprocessable folder {folder_path}: {cleanup_err}")
            
    def _wait_for_ng_confirmation(self, item_name: str, alert_id: str) -> None:
        """Wait for user to confirm NG result before continuing processing"""
        import time
        
        logging.info(f"Waiting for NG confirmation for {item_name} (Alert: {alert_id})")
        
        # Poll until user confirms or timeout (optional)
        timeout_seconds = 3600  # 1 hour timeout (adjustable)
        start_time = time.time()
        
        while time.time() - start_time < timeout_seconds:
            with self.lock:
                # Check if alert has been processed (moved from pending to regular alerts)
                if (alert_id not in self.pending_ng_alerts and 
                    item_name not in self.paused_items and
                    item_name in self.completed_results):
                    logging.info(f"NG confirmation received for {item_name}")
                    return
                    
            # Sleep for a bit before checking again
            time.sleep(2)
        
        # Timeout handling (optional - you might want to handle this differently)
        logging.warning(f"Timeout waiting for NG confirmation: {item_name}")
        # Could auto-confirm or mark as error here

    # ---------- Getters ----------
    def get_status(self, name: str) -> Optional[ProcessingStatus]:
        with self.lock:
            return self.processing_queue.get(name)

    def get_all_statuses(self) -> List[ProcessingStatus]:
        with self.lock:
            return list(self.processing_queue.values())

    def get_result(self, name: str) -> Optional[ProcessingResult]:
        with self.lock:
            return self.completed_results.get(name)

    def get_all_results(self) -> List[ProcessingResult]:
        with self.lock:
            return list(self.completed_results.values())

    def get_stats(self) -> SystemStats:
        with self.lock:
            total = len(self.processing_queue) + len(self.completed_results)
            queued = sum(1 for s in self.processing_queue.values() if s.status == "queued")
            processing = sum(1 for s in self.processing_queue.values() if s.status == "processing")
            paused_ng = sum(1 for s in self.processing_queue.values() if s.status == "paused_ng")
            completed = sum(1 for s in self.processing_queue.values() if s.status == "completed")
            errors = sum(1 for s in self.processing_queue.values() if s.status == "error")
            return SystemStats(
                total_videos_processed=total,
                videos_in_queue=queued,
                videos_processing=processing,
                videos_paused_ng=paused_ng,
                videos_completed=completed,
                videos_with_errors=errors,
                uptime_seconds=time.time() - self.start_time,
            )

    def get_quality_check(self, name: str) -> Optional[QualityCheck]:
        with self.lock:
            return self.quality_checks.get(name)

    def get_all_quality_checks(self) -> List[QualityCheck]:
        with self.lock:
            return list(self.quality_checks.values())

    def get_alerts(self, acknowledged: Optional[bool] = None) -> List[Alert]:
        with self.lock:
            alerts = list(self.alerts.values())
            if acknowledged is not None:
                alerts = [a for a in alerts if a.acknowledged == acknowledged]
            return sorted(alerts, key=lambda a: a.timestamp, reverse=True)

    def acknowledge_alert(self, alert_id: str) -> bool:
        with self.lock:
            if alert_id in self.alerts:
                self.alerts[alert_id].acknowledged = True
                return True
            return False

    def clear_acknowledged_alerts(self) -> int:
        with self.lock:
            before_count = len(self.alerts)
            self.alerts = {k: v for k, v in self.alerts.items() if not v.acknowledged}
            return before_count - len(self.alerts)

# Initialize global processing manager
processing_manager = ProcessingManager()

# ======= STARTUP/SHUTDOWN EVENTS =======
@app.on_event("startup")
async def startup_event() -> None:
    global _scanned_folders, _scanned_files
    
    try:
        processing_manager.initialize()
        
        os.makedirs(Config.INPUT_DIR, exist_ok=True)
        os.makedirs(Config.PROCESSING_DIR, exist_ok=True)
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        try:
            Base.metadata.create_all(bind=engine)  # idempotent
            with engine.begin() as conn:
                conn.execute(text("SELECT 1"))
            logging.info("✅ SQL Server connected & tables ready")
        except Exception as db_err:
            logging.error(f"❌ Database init failed: {db_err}")
            raise

        if os.path.exists(Config.INPUT_DIR):
            # สแกนและเก็บสถานะ folders ที่มีอยู่แล้ว
            folder_names = [f for f in os.listdir(Config.INPUT_DIR)
                            if os.path.isdir(os.path.join(Config.INPUT_DIR, f))]
            
            # เพิ่มทุก folder เข้าไปใน scanned set ก่อน
            _scanned_folders.update(folder_names)
            
            valid_folders = [fn for fn in folder_names if _is_valid_folder_name(fn)]
            if valid_folders:
                logging.info(f"Found {len(valid_folders)} existing folders to process")
                for fn in valid_folders:
                    processing_manager.add_folder_to_queue(os.path.join(Config.INPUT_DIR, fn))

        # สแกนและเก็บสถานะ files ที่มีอยู่แล้ว
        video_ext = (".avi", ".mp4", ".mov", ".mkv")
        if os.path.exists(Config.INPUT_DIR):
            root_files = [f for f in os.listdir(Config.INPUT_DIR) if f.lower().endswith(video_ext)]
            
            # เพิ่มทุกไฟล์เข้าไปใน scanned set ก่อน
            _scanned_files.update(root_files)
            
            if root_files:
                logging.info(f"Found {len(root_files)} existing videos to process (legacy mode)")
                for filename in root_files:
                    processing_manager.add_video_to_queue(os.path.join(Config.INPUT_DIR, filename))

        start_auto_enqueuer(interval_sec=3.0)
        logging.info("🚀 FastAPI server started successfully (auto-enqueue enabled)")
    except Exception as e:
        logging.error(f"❌ Failed to start server: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event() -> None:
    stop_auto_enqueuer()
    processing_manager.executor.shutdown(wait=True)
    logging.info("🛑 FastAPI server stopped")


# ======= API ENDPOINTS =======

# @app.get("/")
# async def root() -> Dict[str, Any]:
#     return {
#         "message": "Airbag Detection API",
#         "version": "1.1.0",
#         "uptime_seconds": time.time() - processing_manager.start_time,
#         "endpoints": [
#             "/docs - API documentation",
#             "/status - System status",
#             "/folders - List folders (folder-mode)",
#             "/process-folder/{folder_name} - Process a folder",
#             "/videos - List loose files (legacy)",
#             "/process/{video_name} - Process a loose file (legacy)",
#             "/results - Processing results",
#             "/download/{filename} - Download Excel result",
#         ],
#     }
    
# @app.get("/dashboard")
# async def serve_dashboard():
#     """Return dashboard.html so http://localhost:8000/dashboard works.
#     Why: FastAPI had no route for /dashboard, so it returned 404.
#     """
#     html_path = Path(__file__).parent / "dashboard.html"
#     if not html_path.exists():
#         # Use 404 so the client knows the UI file is missing on disk
#         raise HTTPException(status_code=404, detail=f"dashboard.html not found at {html_path}")
#     return FileResponse(str(html_path), media_type="text/html")

@app.get("/status", response_model=SystemStats)
async def get_system_status() -> SystemStats:
    return processing_manager.get_stats()

# -------- Folder-mode endpoints --------
@app.get("/folders", response_model=List[FolderInfo])
async def list_folders() -> List[FolderInfo]:
    items: List[FolderInfo] = []
    if not os.path.exists(Config.INPUT_DIR):
        return items
    for name in os.listdir(Config.INPUT_DIR):
        path = os.path.join(Config.INPUT_DIR, name)
        if not os.path.isdir(path):
            continue
        parsed = parse_folder_name(name)
        try:
            stats = os.stat(path)
            video_count = sum(1 for f in os.listdir(path)
                              if f.lower().endswith((".avi", ".mp4", ".mov", ".mkv")))
            items.append(FolderInfo(
                folder_name=name,
                video_count=video_count,
                created_time=datetime.fromtimestamp(stats.st_ctime).isoformat(),
                modified_time=datetime.fromtimestamp(stats.st_mtime).isoformat(),
                valid=bool(parsed.get('valid')),
            ))
        except OSError:
            items.append(FolderInfo(
                folder_name=name, video_count=0, created_time=None, modified_time=None,
                valid=bool(parsed.get('valid')),
            ))
    return items

@app.post("/process-folder/{folder_name}")
async def process_folder(folder_name: str) -> Dict[str, Any]:
    folder_path = os.path.join(Config.INPUT_DIR, folder_name)
    if not os.path.isdir(folder_path):
        raise HTTPException(status_code=404, detail="Folder not found")
    parsed = parse_folder_name(folder_name)
    if not parsed.get('valid'):
        raise HTTPException(status_code=400, detail="Folder name does not match expected pattern")
    success = processing_manager.add_folder_to_queue(folder_path)
    if success:
        return {"message": "Folder queued for processing", "folder_name": folder_name}
    return {"message": "Folder already in processing queue", "folder_name": folder_name}

# -------- Legacy file endpoints (kept for compatibility) --------
@app.get("/videos", response_model=List[VideoInfo])
async def list_videos() -> List[VideoInfo]:
    videos: List[VideoInfo] = []
    try:
        if os.path.exists(Config.INPUT_DIR):
            video_extensions = ('.avi', '.mp4', '.mov', '.mkv')
            for filename in os.listdir(Config.INPUT_DIR):
                if filename.lower().endswith(video_extensions):
                    file_path = os.path.join(Config.INPUT_DIR, filename)
                    stat = os.stat(file_path)
                    videos.append(VideoInfo(
                        filename=filename,
                        file_size=stat.st_size,
                        upload_time=datetime.fromtimestamp(stat.st_ctime).isoformat(),
                        module_sn=extract_module_sn_from_video_name(filename),
                        temperature_type=detect_temperature_from_filename(filename)[0],
                    ))
        return videos
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list videos: {str(e)}")

@app.post("/upload")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)) -> Dict[str, Any]:
    # why: API still accepts single files; folder-mode normally drops folders into INPUT_DIR directly
    if not file.filename.lower().endswith(('.avi', '.mp4', '.mov', '.mkv')):
        raise HTTPException(status_code=400, detail="Invalid file type. Supported: .avi, .mp4, .mov, .mkv")
    file_path = os.path.join(Config.INPUT_DIR, file.filename)
    try:
        content = await file.read()
        with open(file_path, "wb") as buffer:
            buffer.write(content)
        success = processing_manager.add_video_to_queue(file_path)
        if success:
            return {
                "message": "Video uploaded and queued for processing (legacy)",
                "filename": file.filename,
                "file_size": len(content),
                "module_sn": extract_module_sn_from_video_name(file.filename),
                "temperature_type": detect_temperature_from_filename(file.filename)[0],
            }
        return {"message": "Video already in processing queue", "filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")

@app.post("/process/{video_name}")
async def process_video(video_name: str, background_tasks: BackgroundTasks) -> Dict[str, Any]:
    file_path = os.path.join(Config.INPUT_DIR, video_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Video file not found")
    success = processing_manager.add_video_to_queue(file_path)
    if success:
        return {
            "message": "Video queued for processing (legacy)",
            "video_name": video_name,
            "module_sn": extract_module_sn_from_video_name(video_name),
            "temperature_type": detect_temperature_from_filename(video_name)[0],
        }
    return {"message": "Video already in processing queue", "video_name": video_name}

@app.get("/download/{filename}")
async def download_excel(filename: str):
    file_path = os.path.join(Config.OUTPUT_DIR, filename)
    if not os.path.exists(file_path):
        if not filename.endswith('.xlsx'):
            file_path = os.path.join(Config.OUTPUT_DIR, f"{filename}.xlsx")
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Excel file not found")
    return FileResponse(
        path=file_path,
        filename=os.path.basename(file_path),
        media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    )

@app.get("/excel-files")
async def list_excel_files() -> Dict[str, Any]:
    try:
        excel_files: List[Dict[str, Any]] = []
        if os.path.exists(Config.OUTPUT_DIR):
            for filename in os.listdir(Config.OUTPUT_DIR):
                if filename.endswith('.xlsx'):
                    file_path = os.path.join(Config.OUTPUT_DIR, filename)
                    stat = os.stat(file_path)
                    excel_files.append({
                        "filename": filename,
                        "file_size": stat.st_size,
                        "created_time": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                        "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    })
        return {"excel_files": excel_files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list Excel files: {str(e)}")

@app.delete("/videos/{video_name}")
async def delete_video(video_name: str) -> Dict[str, Any]:
    file_path = os.path.join(Config.INPUT_DIR, video_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Video file not found")
    try:
        status = processing_manager.get_status(video_name)
        if status and status.status == "processing":
            raise HTTPException(status_code=400, detail="Cannot delete video that is currently being processed")
        os.remove(file_path)
        with processing_manager.lock:
            processing_manager.processing_queue.pop(video_name, None)
            processing_manager.completed_results.pop(video_name, None)
        return {"message": f"Video {video_name} deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete video: {str(e)}")

@app.get("/logs")
async def get_recent_logs(lines: int = 50) -> Dict[str, Any]:
    try:
        if not os.path.exists(Config.LOG_FILE):
            return {"logs": [], "message": "No log file found"}
        with open(Config.LOG_FILE, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
            recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
        return {
            "logs": [line.strip() for line in recent_lines],
            "total_lines": len(all_lines),
            "returned_lines": len(recent_lines),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read logs: {str(e)}")

@app.post("/clear-queue")
async def clear_processing_queue() -> Dict[str, Any]:
    try:
        with processing_manager.lock:
            active_items = {k: v for k, v in processing_manager.processing_queue.items()
                            if v.status in ["queued", "processing"]}
            cleared_count = len(processing_manager.processing_queue) - len(active_items)
            processing_manager.processing_queue = active_items
        return {
            "message": f"Cleared {cleared_count} completed/error items from queue",
            "remaining_items": len(active_items),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear queue: {str(e)}")

@app.get("/health")
async def health_check() -> Dict[str, Any]:
    try:
        models_ok = processing_manager.processor is not None and processing_manager.excel_manager is not None
        dirs_ok = os.path.exists(Config.INPUT_DIR) and os.path.exists(Config.OUTPUT_DIR)
        return {
            "status": "healthy" if (models_ok and dirs_ok) else "unhealthy",
            "models_loaded": models_ok,
            "directories_exist": dirs_ok,
            "uptime_seconds": time.time() - processing_manager.start_time,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e), "timestamp": datetime.now().isoformat()}

# ======= QUALITY CHECK & ALERT ENDPOINTS =======
@app.get("/quality/check/{name}", response_model=QualityCheck)
async def get_quality_check(name: str) -> QualityCheck:
    quality_check = processing_manager.get_quality_check(name)
    if quality_check is None:
        raise HTTPException(status_code=404, detail="Quality check not found for this item")
    return quality_check

@app.get("/quality/checks", response_model=List[QualityCheck])
async def get_all_quality_checks() -> List[QualityCheck]:
    return processing_manager.get_all_quality_checks()

@app.get("/quality/summary")
async def get_quality_summary() -> Dict[str, Any]:
    try:
        quality_checks = processing_manager.get_all_quality_checks()
        summary: Dict[str, Any] = {
            "total_videos": len(quality_checks),
            "overall_ok": 0,
            "overall_ng": 0,
            "by_temperature": {"room": {"ok": 0, "ng": 0}, "hot": {"ok": 0, "ng": 0}, "cold": {"ok": 0, "ng": 0}},
            "by_parameter": {
                "Front 1": {"ok": 0, "ng": 0, "no_spec": 0, "no_data": 0},
                "Front 2": {"ok": 0, "ng": 0, "no_spec": 0, "no_data": 0},
                "Rear 3": {"ok": 0, "ng": 0, "no_spec": 0, "no_data": 0},
                "Full Deployment": {"ok": 0, "ng": 0, "no_spec": 0, "no_data": 0},
                "Opening Time": {"ok": 0, "ng": 0, "no_spec": 0, "no_data": 0},
            },
        }
        for qc in quality_checks:
            if qc.overall_status == "OK":
                summary["overall_ok"] += 1
            else:
                summary["overall_ng"] += 1
            temp_type = qc.temperature_type.lower()
            if temp_type in summary["by_temperature"]:
                if qc.overall_status == "OK":
                    summary["by_temperature"][temp_type]["ok"] += 1
                else:
                    summary["by_temperature"][temp_type]["ng"] += 1
            for validation in qc.validations:
                param = validation.parameter
                if param in summary["by_parameter"]:
                    status = validation.status.lower()
                    if status == "ok":
                        summary["by_parameter"][param]["ok"] += 1
                    elif status == "ng":
                        summary["by_parameter"][param]["ng"] += 1
                    elif status == "no_spec":
                        summary["by_parameter"][param]["no_spec"] += 1
                    elif status == "no_data":
                        summary["by_parameter"][param]["no_data"] += 1
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate quality summary: {str(e)}")

@app.get("/alerts", response_model=List[Alert])
async def get_alerts(acknowledged: Optional[bool] = None) -> List[Alert]:
    return processing_manager.get_alerts(acknowledged=acknowledged)

@app.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str) -> Dict[str, Any]:
    # ใช้ alert.id แทน alert_id ใน comparison
    with processing_manager.lock:
        found_alert = None
        for alert in processing_manager.alerts.values():
            if alert.id == alert_id:
                found_alert = alert
                break
        
        if found_alert:
            found_alert.acknowledged = True
            return {"message": "Alert acknowledged successfully", "alert_id": alert_id}
        else:
            raise HTTPException(status_code=404, detail="Alert not found")

@app.delete("/alerts/acknowledged")
async def clear_acknowledged_alerts() -> Dict[str, Any]:
    cleared_count = processing_manager.clear_acknowledged_alerts()
    return {"message": f"Cleared {cleared_count} acknowledged alerts"}

@app.get("/alerts/pending-ng", response_model=List[Alert])
async def get_pending_ng_alerts() -> List[Alert]:
    """Get NG alerts waiting for user confirmation"""
    return processing_manager.get_pending_ng_alerts()

@app.post("/alerts/{alert_id}/continue")
async def continue_after_ng(alert_id: str) -> Dict[str, Any]:
    """Continue processing after NG confirmation (แก้ไขให้ robust มากขึ้น)"""
    # เพิ่ม logging เพื่อ debug
    logging.info(f"Attempting to continue processing for alert_id: {alert_id}")
    
    # Diagnostics
    with processing_manager.lock:
        pending_keys = list(processing_manager.pending_ng_alerts.keys())
        paused_items = {k: v.get("alert_id") for k, v in processing_manager.paused_items.items()}
        alerts_keys = list(processing_manager.alerts.keys())
        
    logging.debug(f"Pending alerts: {pending_keys}")
    logging.debug(f"Paused items: {paused_items}")
    logging.debug(f"Regular alerts: {alerts_keys}")

    success = processing_manager.continue_processing(alert_id)
    
    if success:
        return {"message": "Processing continued successfully", "alert_id": alert_id}
    else:
        # ให้ข้อมูล error ที่ละเอียดขึ้น
        with processing_manager.lock:
            if alert_id in processing_manager.alerts:
                return {"message": "Already continued (idempotent)", "alert_id": alert_id}
        
        raise HTTPException(
            status_code=404, 
            detail=f"Pending NG alert not found: {alert_id}. Available pending alerts: {pending_keys}"
        )
        
@app.get("/debug/alerts-state")
async def debug_alerts_state():
    """Debug endpoint to check current alert states"""
    with processing_manager.lock:
        return {
            "pending_ng_alerts": {
                "count": len(processing_manager.pending_ng_alerts),
                "keys": list(processing_manager.pending_ng_alerts.keys()),
                "items": [
                    {
                        "alert_id": k,
                        "video_name": v.video_name if hasattr(v, 'video_name') else "N/A",
                        "id_field": getattr(v, 'id', "N/A")
                    }
                    for k, v in processing_manager.pending_ng_alerts.items()
                ]
            },
            "paused_items": {
                "count": len(processing_manager.paused_items),
                "items": {
                    k: {
                        "alert_id": v.get("alert_id"),
                        "timestamp": v.get("timestamp")
                    }
                    for k, v in processing_manager.paused_items.items()
                }
            },
            "regular_alerts": {
                "count": len(processing_manager.alerts),
                "keys": list(processing_manager.alerts.keys())
            }
        }

@app.get("/processing/paused")
async def get_paused_items() -> Dict[str, Any]:
    """Get items paused due to NG results"""
    with processing_manager.lock:
        paused_data = {}
        for item_name, data in processing_manager.paused_items.items():
            paused_data[item_name] = {
                "timestamp": data["timestamp"],
                "alert_id": data["alert_id"],
                "ng_count": data["quality_check"].ng_count,
                "total_checked": data["quality_check"].total_checked
            }
    return {"paused_items": paused_data, "count": len(paused_data)}

@app.get("/processing/status", response_model=List[ProcessingStatus])
async def get_processing_status() -> List[ProcessingStatus]:
    return processing_manager.get_all_statuses()

@app.get("/results", response_model=List[ProcessingResult])
async def get_processing_results() -> List[ProcessingResult]:
    return processing_manager.get_all_results()

@app.get("/results/latest")
async def get_latest_result() -> Dict[str, Any]:
    """Get latest processing result with quality check info"""
    try:
        all_results = processing_manager.get_all_results()
        all_quality_checks = processing_manager.get_all_quality_checks()
        
        if not all_results:
            return {"result": None, "quality_check": None}
        
        # Find latest result by timestamp or processing order
        latest_result = max(all_results, key=lambda r: r.video_name)
        
        # Find corresponding quality check
        quality_check = None
        for qc in all_quality_checks:
            if qc.video_name == latest_result.video_name:
                quality_check = qc
                break
        
        return {
            "result": latest_result.dict(),
            "quality_check": quality_check.dict() if quality_check else None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get latest result: {str(e)}")

@app.get("/results/{item_name}/summary")
async def get_result_summary(item_name: str) -> Dict[str, Any]:
    """Get processing result summary for specific item"""
    try:
        result = processing_manager.get_result(item_name)
        quality_check = processing_manager.get_quality_check(item_name)
        
        if not result:
            raise HTTPException(status_code=404, detail="Result not found")
        
        summary = {
            "video_name": result.video_name,
            "model": result.module_sn or "Unknown",
            "in_spec": quality_check.overall_status if quality_check else None,
            "reliability": result.acc_rate_confidence,
            "processing_time": result.processing_time
        }
        
        return summary
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get result summary: {str(e)}")

@app.get("/db/health")
async def db_health(_: Session = Depends(get_db)):
    try:
        with engine.begin() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")
    
@app.get("/kpi/accuracy-avg")
def kpi_accuracy_avg(db: Session = Depends(get_db)):
    try:
        avg_val = db.query(func.avg(AiCopTestResult.AccuracyRate)).scalar()
        count = db.query(func.count(AiCopTestResult.ResultId)).scalar() or 0
        return {
            "accuracy_avg": float(avg_val) if avg_val is not None else None,
            "count": int(count),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to compute accuracy average: {e}")
    
@app.get("/kpi/overall-stats")
def kpi_overall_stats(db: Session = Depends(get_db)):
    """Get overall KPI statistics from database"""
    try:
        # Total tests
        total_tests = db.query(func.count(AiCopTestResult.ResultId)).scalar() or 0
        
        # Total PASS (in spec)
        total_pass = db.query(func.count(AiCopTestResult.ResultId)).filter(
            AiCopTestResult.OverallResult == "PASS"
        ).scalar() or 0
        
        # Total NG (out of spec) 
        total_ng = db.query(func.count(AiCopTestResult.ResultId)).filter(
            AiCopTestResult.OverallResult == "NG"
        ).scalar() or 0
        
        # Average accuracy rate
        avg_accuracy = db.query(func.avg(AiCopTestResult.AccuracyRate)).scalar()
        avg_accuracy = float(avg_accuracy) if avg_accuracy is not None else None
        
        # Current queue count (from memory - this is still relevant for real-time status)
        current_stats = processing_manager.get_stats()
        current_queue = current_stats.videos_in_queue + current_stats.videos_processing
        
        return {
            "total_tests": int(total_tests),
            "total_pass": int(total_pass), 
            "total_ng": int(total_ng),
            "accuracy_avg": round(float(avg_accuracy), 2) if avg_accuracy is not None else None,
            "current_queue": int(current_queue)  # real-time queue count
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get KPI stats: {e}")

@app.get("/kpi/daily-stats")
def kpi_daily_stats(days: int = 30, db: Session = Depends(get_db)):
    """Get daily statistics for the last N days"""
    try:
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Get daily counts
        daily_query = db.query(
            func.cast(AiCopTestResult.TestDate, String).label('test_date'),
            func.count(AiCopTestResult.ResultId).label('total_count'),
            func.sum(func.case([(AiCopTestResult.OverallResult == 'PASS', 1)], else_=0)).label('pass_count'),
            func.sum(func.case([(AiCopTestResult.OverallResult == 'NG', 1)], else_=0)).label('ng_count'),
            func.avg(AiCopTestResult.AccuracyRate).label('avg_accuracy')
        ).filter(
            AiCopTestResult.TestDate >= cutoff_date
        ).group_by(
            func.cast(AiCopTestResult.TestDate, String)
        ).order_by(
            func.cast(AiCopTestResult.TestDate, String).desc()
        ).all()
        
        daily_stats = []
        for row in daily_query:
            daily_stats.append({
                "date": row.test_date.split('T')[0] if row.test_date else None,  # Extract date part
                "total_count": int(row.total_count or 0),
                "pass_count": int(row.pass_count or 0), 
                "ng_count": int(row.ng_count or 0),
                "avg_accuracy": round(float(row.avg_accuracy), 2) if row.avg_accuracy else None
            })
            
        return {
            "daily_stats": daily_stats,
            "period_days": days
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get daily stats: {e}")
    
# Report API
@app.get("/reports/test-results", response_model=ReportsResponse)
async def get_test_results(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    model_name: Optional[str] = None,
    overall_result: Optional[str] = None,
    serial_number: Optional[str] = None,
    cop_no: Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
    db: Session = Depends(get_db)
):
    """Get paginated test results with filtering"""
    try:
        query = db.query(AiCopTestResult)
        
        # Apply filters
        if start_date:
            query = query.filter(AiCopTestResult.TestDate >= datetime.fromisoformat(start_date.replace('Z', '+00:00')))
        
        if end_date:
            query = query.filter(AiCopTestResult.TestDate <= datetime.fromisoformat(end_date.replace('Z', '+00:00')))
        
        if model_name:
            query = query.filter(AiCopTestResult.ModelName == model_name)
        
        if overall_result:
            query = query.filter(AiCopTestResult.OverallResult == overall_result)
        
        if serial_number:
            query = query.filter(AiCopTestResult.SerialNumber.ilike(f"%{serial_number}%"))
        
        if cop_no:
            query = query.filter(AiCopTestResult.COPNo.ilike(f"%{cop_no}%"))
        
        # Get total count
        total = query.count()
        
        # Apply pagination and ordering
        query = query.order_by(desc(AiCopTestResult.TestDate))
        query = query.offset((page - 1) * page_size).limit(page_size)
        
        results = query.all()
        
        # Convert to response format
        data = []
        for result in results:
            data.append(TestResultSummary(
                result_id=result.ResultId,
                ai_model_id=result.AIModelId,
                model_name=result.ModelName or "",
                cop_no=result.COPNo,
                serial_number=result.SerialNumber,
                test_date=result.TestDate.isoformat() if result.TestDate else None,
                overall_result=result.OverallResult,
                accuracy_rate=float(result.AccuracyRate) if result.AccuracyRate else None,
                created_date=result.CreatedDate.isoformat() if result.CreatedDate else None,
                comment=result.Comment
            ))
        
        total_pages = (total + page_size - 1) // page_size
        
        return ReportsResponse(
            data=data,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch test results: {str(e)}")

@app.get("/reports/test-result/{result_id}", response_model=TestResultWithDetails)
async def get_test_result_details(result_id: int, db: Session = Depends(get_db)):
    """Get detailed test result with all measurements"""
    try:
        result = db.query(AiCopTestResult).filter(AiCopTestResult.ResultId == result_id).first()
        
        if not result:
            raise HTTPException(status_code=404, detail="Test result not found")
        
        # Get details
        details_query = db.query(AiCopTestResultDetail).filter(
            AiCopTestResultDetail.ResultId == result_id
        ).all()
        
        details = []
        for detail in details_query:
            details.append(TestResultDetail(
                detail_id=detail.DetailId,
                point_name=detail.PointName,
                measured_value=float(detail.MeasuredValue) if detail.MeasuredValue else None,
                target_value=float(detail.TargetValue) if detail.TargetValue else None,
                result=detail.Result
            ))
        
        return TestResultWithDetails(
            result_id=result.ResultId,
            ai_model_id=result.AIModelId,
            model_name=result.ModelName or "",
            cop_no=result.COPNo,
            serial_number=result.SerialNumber,
            test_date=result.TestDate.isoformat() if result.TestDate else None,
            overall_result=result.OverallResult,
            accuracy_rate=float(result.AccuracyRate) if result.AccuracyRate else None,
            created_date=result.CreatedDate.isoformat() if result.CreatedDate else None,
            comment=result.Comment,
            details=details
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch test result details: {str(e)}")

@app.get("/reports/statistics", response_model=ReportsStatistics)
async def get_reports_statistics(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get statistical summary of test results"""
    try:
        query = db.query(AiCopTestResult)
        
        # Apply date filters
        if start_date:
            query = query.filter(AiCopTestResult.TestDate >= datetime.fromisoformat(start_date.replace('Z', '+00:00')))
        
        if end_date:
            query = query.filter(AiCopTestResult.TestDate <= datetime.fromisoformat(end_date.replace('Z', '+00:00')))
        
        results = query.all()
        
        # Calculate statistics
        total_tests = len(results)
        pass_count = sum(1 for r in results if r.OverallResult == "PASS")
        ng_count = total_tests - pass_count
        pass_rate = (pass_count / total_tests * 100) if total_tests > 0 else 0
        
        # Calculate average accuracy
        accuracy_values = [float(r.AccuracyRate) for r in results if r.AccuracyRate is not None]
        avg_accuracy = sum(accuracy_values) / len(accuracy_values) if accuracy_values else None
        
        # Group by model
        by_model = {}
        for result in results:
            model = result.ModelName or "Unknown"
            if model not in by_model:
                by_model[model] = {"pass": 0, "ng": 0}
            
            if result.OverallResult == "PASS":
                by_model[model]["pass"] += 1
            else:
                by_model[model]["ng"] += 1
        
        # Group by date (last 30 days)
        by_date = []
        if start_date and end_date:
            start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            
            date_counts = {}
            for result in results:
                if result.TestDate:
                    date_key = result.TestDate.date().isoformat()
                    date_counts[date_key] = date_counts.get(date_key, 0) + 1
            
            by_date = [{"date": date, "count": count} for date, count in sorted(date_counts.items())]
        
        return ReportsStatistics(
            total_tests=total_tests,
            pass_count=pass_count,
            ng_count=ng_count,
            pass_rate=round(pass_rate, 2),
            avg_accuracy=round(avg_accuracy, 2) if avg_accuracy else None,
            by_model=by_model,
            by_date=by_date
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch statistics: {str(e)}")

@app.delete("/reports/test-result/{result_id}")
async def delete_test_result(result_id: int, db: Session = Depends(get_db)):
    """Delete a test result and its details"""
    try:
        result = db.query(AiCopTestResult).filter(AiCopTestResult.ResultId == result_id).first()
        
        if not result:
            raise HTTPException(status_code=404, detail="Test result not found")
        
        # Delete details first (due to foreign key)
        db.query(AiCopTestResultDetail).filter(AiCopTestResultDetail.ResultId == result_id).delete()
        
        # Delete main result
        db.delete(result)
        db.commit()
        
        return {"message": "Test result deleted successfully", "result_id": result_id}
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete test result: {str(e)}")
    
@app.get("/folders/{folder_name:path}/images")
async def list_folder_images(folder_name: str) -> Dict[str, Any]:
    """
    คืน URL ของรูป key-frames ใน \\...\\Result\\{folder_name}
    เฉพาะไฟล์ที่มีอยู่จริงเท่านั้น
    """
    # Decode URL encoding first
    from urllib.parse import unquote
    decoded_folder_name = unquote(folder_name)
    
    folder_dir = os.path.join(Config.OUTPUT_DIR, decoded_folder_name)
    logging.info(f"Looking for images in: {folder_dir}")
    logging.info(f"Original folder_name: '{folder_name}', decoded: '{decoded_folder_name}'")
    
    if not os.path.isdir(folder_dir):
        # Try to find similar folder names
        if os.path.exists(Config.OUTPUT_DIR):
            available_folders = [f for f in os.listdir(Config.OUTPUT_DIR) 
                               if os.path.isdir(os.path.join(Config.OUTPUT_DIR, f))]
            logging.warning(f"Folder not found: {folder_dir}")
            logging.info(f"Available folders: {available_folders}")
            
            # Try to find a matching folder (case-insensitive, flexible matching)
            for available in available_folders:
                if (decoded_folder_name.lower() in available.lower() or 
                    available.lower() in decoded_folder_name.lower() or
                    decoded_folder_name.replace('_', ' ').lower() == available.lower() or
                    decoded_folder_name.replace(' ', '_').lower() == available.lower()):
                    
                    logging.info(f"Found matching folder: {available}")
                    folder_dir = os.path.join(Config.OUTPUT_DIR, available)
                    decoded_folder_name = available
                    break
            else:
                raise HTTPException(status_code=404, detail="Folder result not found")
        else:
            raise HTTPException(status_code=404, detail="Output directory not found")

    patterns = {
        "explosion": f"*explosion_f*.jpg",
        "fr1": f"*fr1_f*.jpg", 
        "fr2": f"*fr2_f*.jpg",
        "re3": f"*re3_f*.jpg",
        "full_deployment": f"*full_deploy_f*.jpg",
    }

    out = {}
    for k, pat in patterns.items():
        matches = sorted(glob.glob(os.path.join(folder_dir, pat)))
        logging.info(f"Pattern {pat} found {len(matches)} matches")
        if matches:
            fname = os.path.basename(matches[-1])
            # URL encode both folder name and filename
            from urllib.parse import quote
            encoded_folder = quote(decoded_folder_name, safe='')
            encoded_filename = quote(fname, safe='')
            url = f"/static/results/{encoded_folder}/{encoded_filename}"
            out[k] = url
            logging.info(f"Added image URL: {url}")

    logging.info(f"Final image URLs for {decoded_folder_name}: {out}")
    return {"folder": decoded_folder_name, "images": out}


@app.get("/admin/scan-cache-status")
async def get_scan_cache_status() -> Dict[str, Any]:
    """Get current scan cache status"""
    global _scanned_folders, _scanned_files
    
    return {
        "scanned_folders": {
            "count": len(_scanned_folders),
            "items": list(_scanned_folders)
        },
        "scanned_files": {
            "count": len(_scanned_files),
            "items": list(_scanned_files)
        }
    }
    
@app.get("/debug/folder-files/{folder_name}")
async def debug_folder_files(folder_name: str):
    """Debug endpoint to check what files exist in folder"""
    folder_dir = os.path.join(Config.OUTPUT_DIR, folder_name)
    
    if not os.path.exists(folder_dir):
        return {"error": f"Folder does not exist: {folder_dir}"}
    
    try:
        all_files = []
        for item in os.listdir(folder_dir):
            item_path = os.path.join(folder_dir, item)
            all_files.append({
                "name": item,
                "is_file": os.path.isfile(item_path),
                "size": os.path.getsize(item_path) if os.path.isfile(item_path) else 0
            })
        
        return {
            "folder": folder_name,
            "folder_path": folder_dir,
            "files": all_files,
            "total_files": len(all_files)
        }
    except Exception as e:
        return {"error": str(e)}
    
@app.get("/debug/folder-mapping/{folder_name:path}")
async def debug_folder_mapping(folder_name: str):
    """Debug folder name mapping"""
    from urllib.parse import unquote
    
    decoded = unquote(folder_name)
    
    result = {
        "original": folder_name,
        "decoded": decoded,
        "exists": False,
        "available_folders": [],
        "matches": []
    }
    
    if os.path.exists(Config.OUTPUT_DIR):
        available = [f for f in os.listdir(Config.OUTPUT_DIR) 
                    if os.path.isdir(os.path.join(Config.OUTPUT_DIR, f))]
        result["available_folders"] = available
        
        target_path = os.path.join(Config.OUTPUT_DIR, decoded)
        result["exists"] = os.path.exists(target_path)
        result["target_path"] = target_path
        
        # Find potential matches
        for folder in available:
            if (decoded.lower() in folder.lower() or 
                folder.lower() in decoded.lower() or
                decoded.replace('_', ' ').lower() == folder.lower() or
                decoded.replace(' ', '_').lower() == folder.lower()):
                result["matches"].append(folder)
    
    return result

@app.get("/debug/scanner-status")
async def debug_scanner_status():
    """ตรวจสอบสถานะของ auto-scanner"""
    global _auto_scan_thread, _scanned_folders, _scanned_files
    
    return {
        "scanner_running": _auto_scan_thread is not None and _auto_scan_thread.is_alive(),
        "input_dir_accessible": os.path.exists(Config.INPUT_DIR),
        "scanned_folders_count": len(_scanned_folders),
        "scanned_folders": list(_scanned_folders),
        "scanned_files_count": len(_scanned_files),
        "processing_queue_size": len(processing_manager.processing_queue),
        "completed_results_size": len(processing_manager.completed_results)
    }

@app.post("/debug/force-scan")
async def force_scan():
    """บังคับให้ทำการ scan ทันที"""
    logging.info("🔧 Manual scan triggered via API")
    _scan_once_and_enqueue()
    return {"message": "Manual scan completed"}

@app.get("/debug/input-dir-contents")
async def debug_input_dir_contents():
    """แสดงเนื้อหาใน INPUT_DIR"""
    if not os.path.exists(Config.INPUT_DIR):
        return {"error": f"INPUT_DIR not found: {Config.INPUT_DIR}"}
    
    try:
        items = []
        for name in os.listdir(Config.INPUT_DIR):
            path = os.path.join(Config.INPUT_DIR, name)
            is_dir = os.path.isdir(path)
            
            item_info = {
                "name": name,
                "is_directory": is_dir,
                "valid_format": _is_valid_folder_name(name) if is_dir else False,
                "already_scanned": name in _scanned_folders if is_dir else name in _scanned_files
            }
            
            if is_dir:
                # นับจำนวนวิดีโอในโฟลเดอร์
                try:
                    video_count = sum(1 for f in os.listdir(path) 
                                    if f.lower().endswith(('.avi', '.mp4', '.mov', '.mkv')))
                    item_info["video_count"] = video_count
                except:
                    item_info["video_count"] = 0
            
            items.append(item_info)
        
        return {
            "input_dir": Config.INPUT_DIR,
            "items": items,
            "total_items": len(items)
        }
        
    except Exception as e:
        return {"error": str(e)}

@app.get("/debug/folder-validation/{folder_name}")
async def debug_folder_validation(folder_name: str):
    """ทดสอบการ validate ชื่อโฟลเดอร์"""
    parsed = parse_folder_name(folder_name)
    return {
        "folder_name": folder_name,
        "parsed_result": parsed,
        "is_valid": _is_valid_folder_name(folder_name),
        "has_videos": _folder_has_videos(os.path.join(Config.INPUT_DIR, folder_name)) 
                     if os.path.exists(os.path.join(Config.INPUT_DIR, folder_name)) else False
    }

@app.post("/debug/clear-scan-cache")
async def clear_scan_cache():
    """ล้างแคช scan เพื่อให้ scan ใหม่"""
    global _scanned_folders, _scanned_files
    old_folders = len(_scanned_folders)
    old_files = len(_scanned_files)
    
    _scanned_folders.clear()
    _scanned_files.clear()
    
    return {
        "message": "Scan cache cleared",
        "cleared_folders": old_folders,
        "cleared_files": old_files
    }

# =================== Serve Static Files ===================
STATIC_DIR = "out"
if os.path.exists(STATIC_DIR) and os.path.isdir(STATIC_DIR):
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
else:
    print(f"Warning: The '{STATIC_DIR}' directory was not found. Please run 'npm run build' in your Next.js project to generate the static files.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8082)



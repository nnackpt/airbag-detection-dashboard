# YOLO+SAM AND SEGMENTATION Model

import contextlib
import os
import sys
import time
import json
import pandas as pd
import logging
from datetime import datetime
from pathlib import Path
from watchdog.observers.polling import PollingObserver as Observer
from watchdog.events import FileSystemEventHandler
import cv2
import numpy as np
import torch
from scipy.ndimage import uniform_filter1d
from segment_anything import sam_model_registry, SamPredictor
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq, TextIteratorStreamer
from threading import Thread
import re
import shutil
import subprocess
import glob
import signal
from typing import Optional, Tuple


PROGRESS_CALLBACK = None


# Config
NETWORK_PATH = r"\\ata-la-wd2201\Video"  #  10.83.20.25    UNC path  r"\\10.86.16.40\AI_ML Project\MLAB_Video"  Test environment   r"\\10.83.20.25\Video" = Lab
USERNAME = "z-jadet.jewpattanaku"
PASSWORD = "Onelabel62"

VIDEO_EXTENSIONS = (".avi", ".mp4", ".mov", ".mkv")
MAX_RETRY = 3
RETRY_INTERVAL = 5  # วินาที

def connect_unc_path(path, user, pwd):
    """Authenticate UNC path session"""
    try:
        # ลบ session เก่า
        subprocess.run(f'net use "{path}" /delete /y', shell=True, check=False)
        time.sleep(1)
        # login session ใหม่
        cmd = f'net use "{path}" /user:{user} {pwd}'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Authenticated to {path} successfully.")
            return True
        else:
            print(f"Failed to authenticate to {path}.")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def scan_videos(path):
    """Scan for video files recursively"""
    video_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.lower().endswith(VIDEO_EXTENSIONS):
                video_files.append(os.path.join(root, file))
    return video_files

# Retry loop for authentication
for attempt in range(1, MAX_RETRY + 1):
    if os.path.exists(NETWORK_PATH):
        print(f"Path exists: {NETWORK_PATH}")
        break
    else:
        print(f"Attempt {attempt}: Path not accessible, trying to authenticate...")
        if connect_unc_path(NETWORK_PATH, USERNAME, PASSWORD):
            time.sleep(2)  # ให้ session update
            if os.path.exists(NETWORK_PATH):
                print(f"Path now accessible: {NETWORK_PATH}")
                break
        time.sleep(RETRY_INTERVAL)
else:
    print("Failed to access network path after retries.")
    exit(1)

# Scan video files
videos = scan_videos(NETWORK_PATH)
if videos:
    print("Found video files:")
    for v in videos:
        print(v)
else:
    print("No video files found in folder.")



def register_progress_callback(cb):
    global PROGRESS_CALLBACK
    PROGRESS_CALLBACK = cb
    
class _ProgressHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        global PROGRESS_CALLBACK
        if PROGRESS_CALLBACK is None:
            return
        try:
            msg = self.format(record)
        except Exception:
            msg = record.getMessage()
        m = re.search(r"progress\s*=\s*(\d+)%", msg)
        if m:
            try:
                PROGRESS_CALLBACK(int(m.group(1)))
            except Exception:
                pass
            
WARNING_COUNTERS: dict[Tuple[str, str], int] = {}
WARNING_LIMITS = {
    'invalid_folder_format': 1,
    'invalid_file_format': 1,
    'no_circles_early_skip': 1
}


def log_limited_warning(warning_type: str, message: str, *, key: Optional[str] = None):
    """Log warning จำกัดจำนวนแบบราย key (เช่น รายโฟลเดอร์/ไฟล์)"""
    global WARNING_COUNTERS, WARNING_LIMITS
    k = (warning_type, key or message)
    limit = WARNING_LIMITS.get(warning_type, 3)
    cnt = WARNING_COUNTERS.get(k, 0)
    if cnt < limit:
        logging.warning(message)
        cnt += 1
        WARNING_COUNTERS[k] = cnt
        if cnt >= limit:
            logging.info(
                f"Warning limit reached for '{warning_type}' (key='{k[1]}') - suppressing further duplicates"
            )

# ======= CONFIGURATION =======
class Config:
    # Model paths
    YOLO_OBJECT_MODEL_PATH = r"Model\YOLO_OBJECT.pt"
    YOLO_NAME_MODEL_PATH = r"Model\YOLO_NAME.pt"
    YOLO_SEGMENTATION_MODEL_PATH = r"Model\Segmentation.pt"
    SAM_CHECKPOINT = r"Model\sam_vit_h_4b8939.pth"
    OCR_MODEL_PATH = r"Model\ocr"

    # Directories - UPDATED PATHS os.system(r'net use M: \\10.86.16.40\AI_ML Project\MLAB_Video /user:username password') r"M:\ProjectData\video\sample.mp4

   # INPUT_DIR = r"\\ath-ma-wd2502\AI_ML Project\MLAB_Video\P703 DBL CAB HT N1WB-E042D94-AEDYACB25266110402_C001H001S0001"  # monitor dir
    INPUT_DIR = NETWORK_PATH  #=r"\\10.86.16.40\AI_ML Project\MLAB_Video"
    # INPUT_DIR = r"\\ata-la-wd2201\Video"
    
    
    PROCESSING_DIR = r"\\ata-of-wd2345\AI Project\ML_AB\videos\Video"
    BIN_DIR = r"\\ata-of-wd2345\AI Project\ML_AB\videos\Bin"     # processed folders destination
    OUTPUT_DIR = r"\\ata-of-wd2345\AI Project\ML_AB\videos\Result"  # Excel results directory
    EXCEL_OUTPUT_PATH = r"\\ata-of-wd2345\AI Project\ML_AB\videos\Result\airbag_results.xlsx"
    LOG_FILE = r"\\ata-of-wd2345\AI Project\ML_AB\videos\Result\automation.log"
    
    PER_VIDEO_TIMEOUT_SECONDS = 420        # รวมทั้งคลิป (~7 นาที)
    DETECTION_STAGE_TIMEOUT_SECONDS = 150  # เฉพาะช่วงหา FR1/FR2/RE3 + explosion
    DEPLOY_STAGE_TIMEOUT_SECONDS = 120     # ช่วง full deployment window
    TIMEOUT_ACTION = "skip"                # "skip" หรือ "restart"
    RESTART_GRACE_SECONDS = 3              # หน่วงเวลาก่อนรีสตาร์ท

    # Processing thresholds
    CONFIDENCE_THRESHOLD = 0.5
    MOTION_THRESHOLD = 1500
    
    DELETE_COPIED_FOLDER_AFTER_PROCESSING = False  # True = ลบ, False = ไม่ลบ

    # Stabilization / gating
    WARMUP_MIN_FRAMES = 4
    MERGE_RADIUS = 32
    MIN_HITS_PER_CENTER = 3
    GATE_RADIUS = 180

    # Robust circle search
    MAX_FRAMES_FOR_CIRCLE_SEARCH = 80
    CIRCLE_MIN_DIST = 50
    CIRCLE_PARAM_SETS = [
        dict(dp=1.2, param1=80, param2=30, minRadius=8, maxRadius=60),
        dict(dp=1.0, param1=70, param2=24, minRadius=8, maxRadius=72),
        dict(dp=1.2, param1=60, param2=20, minRadius=6, maxRadius=85),
    ]
    
    EARLY_SKIP_NO_CIRCLE_FRAME = 50

    # Temperature-specific settings
    TEMP_CONFIGS = {
        'room': {'start_frame': 109, 'end_frame': 119, 'smooth_size': 3, 'plateau_alpha': 0.97, 'temp_range': (20, 25)},
        'hot': {'start_frame': 100, 'end_frame': 133, 'smooth_size': 3, 'plateau_alpha': 0.97, 'temp_range': (65, 75)},
        'cold': {'start_frame': 125, 'end_frame': 145, 'smooth_size': 3, 'plateau_alpha': 0.97, 'temp_range': (-35, -25)}
    }
    
    PER_VIDEO_MODE = True
    CLEANUP_LEGACY_ON_START = True
    
    MOVE_TO_BIN_AFTER_PROCESSING = False


# ======= LOGGING SETUP =======
def setup_logging():
    os.makedirs(os.path.dirname(Config.LOG_FILE), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(Config.LOG_FILE, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.getLogger().addHandler(_ProgressHandler())

def _ensure_dir(p: str) -> None:
    try:
        os.makedirs(p, exist_ok=True)
    except Exception as e:
        logging.warning(f"Cannot create dir {p}: {e}")

def _apply_mask_overlay(frame_bgr: np.ndarray, mask: np.ndarray, color=(0, 255, 255), alpha=0.6) -> np.ndarray:
    """Apply colored mask overlay on frame"""
    try:
        # Ensure mask is binary
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        
        # Resize mask if needed
        if mask.shape != frame_bgr.shape[:2]:
            mask = cv2.resize(mask, (frame_bgr.shape[1], frame_bgr.shape[0]))
        
        # Create colored overlay
        overlay = frame_bgr.copy()
        overlay[mask > 0] = color
        
        # Blend with original frame
        result = cv2.addWeighted(frame_bgr, 1 - alpha, overlay, alpha, 0)
        
        # Add contour for better visibility
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, color, 2)
        
        return result
    except Exception as e:
        logging.warning(f"Failed to apply mask overlay: {e}")
        return frame_bgr

def _save_frame_with_mask(frame_bgr: np.ndarray, mask: np.ndarray, out_path: str, overlay_color=(255, 255, 0)) -> str:
    """Save frame with mask overlay"""
    try:
        _ensure_dir(os.path.dirname(out_path))
        
        # Apply mask overlay
        frame_with_mask = _apply_mask_overlay(frame_bgr, mask, overlay_color)
        
        ok = cv2.imwrite(out_path, frame_with_mask)
        if ok:
            logging.info(f"Saved frame with mask: {out_path}")
            return out_path
        logging.warning(f"Failed to save image: {out_path}")
    except Exception as e:
        logging.warning(f"Save image with mask error ({out_path}): {e}")
    return ""

def _now() -> float:
    return time.time()

def _deadline(seconds: int | float | None) -> float | None:
    return (_now() + float(seconds)) if seconds and seconds > 0 else None

def _check_deadline(deadline: float | None, stage: str) -> None:
    # why: hard-stop runaway loops; surface clear reason in logs
    if deadline is not None and _now() > deadline:
        raise TimeoutError(f"{stage} timeout exceeded")

def schedule_self_restart(delay: int = 2) -> None:
    # why: trigger a clean in-place restart to recover stuck CUDA/driver states
    def _do():
        try:
            logging.error(f"🔁 Restarting service in {delay}s due to timeout...")
            time.sleep(max(0, delay))
            python = sys.executable
            os.execv(python, [python] + sys.argv)  # replace current process
        except Exception as e:
            logging.error(f"Failed to self-restart: {e}")
            os._exit(1)  # last resort
    Thread(target=_do, daemon=True).start()

# ======= UTILITY FUNCTIONS - UPDATED =======
def _apply_mask_overlay(frame_bgr: np.ndarray, mask: np.ndarray, color=(0, 255, 255), alpha=0.6) -> np.ndarray:
    """Apply colored mask overlay on frame"""
    try:
        # Ensure mask is binary
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        
        # Resize mask if needed
        if mask.shape != frame_bgr.shape[:2]:
            mask = cv2.resize(mask, (frame_bgr.shape[1], frame_bgr.shape[0]))
        
        # Create colored overlay
        overlay = frame_bgr.copy()
        overlay[mask > 0] = color
        
        # Blend with original frame
        result = cv2.addWeighted(frame_bgr, 1 - alpha, overlay, alpha, 0)
        
        # Add contour for better visibility
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, color, 2)
        
        return result
    except Exception as e:
        logging.warning(f"Failed to apply mask overlay: {e}")
        return frame_bgr
    
def parse_folder_name(folder_name: str) -> dict:
    try:
        name = folder_name.strip()
        if not has_allowed_prefix(name):
            log_limited_warning(
                'invalid_folder_format',
                f"Folder '{folder_name}' ignored: must start with 'P703 CAB DBL' or 'P703 DBL CAB'",
                key=folder_name
            )
            return {'temp_type': 'room', 'module_sn': '', 'valid': False}

        # Pattern ที่ยืดหยุ่นขึ้น - ไม่บังคับ suffix
        pattern = r'^(P703\s+CAB\s+DBL|P703\s+DBL\s+CAB)\s+(RT|CT|HT)\s+([A-Z0-9\-]+)'
        m = re.search(pattern, name, flags=re.IGNORECASE)
        if not m:
            log_limited_warning(
                'invalid_folder_format',
                f"Folder name '{folder_name}' doesn't match expected pattern",
                key=folder_name
            )
            return {'temp_type': 'room', 'module_sn': '', 'valid': False}

        temp_code = m.group(2).upper()
        sn_core = m.group(3).split('_', 1)[0].strip()
        temp_map = {'RT': 'room', 'CT': 'cold', 'HT': 'hot'}
        temp_type = temp_map.get(temp_code, 'room')
        return {'temp_type': temp_type, 'module_sn': sn_core, 'valid': True}
    except Exception as e:
        logging.error(f"Error parsing folder name '{folder_name}': {e}")
        return {'temp_type': 'room', 'module_sn': '', 'valid': False}


def extract_module_sn_from_video_name(video_name: str) -> str:
    """
    Extract module SN from a name that looks like:
    '... CT N1WB-E042D95-AEDYACB25265110001_C002H001S0001'
    Read SN as the token right after RT/CT/HT, and ignore everything after '_'.
    """
    base = os.path.splitext(os.path.basename(str(video_name)))[0]
    # Find 'RT|CT|HT' then the SN token, which may be followed by an underscore-suffix
    m = re.search(r'\b(RT|CT|HT)\s+([A-Z0-9\-]+)(?:_[A-Z0-9_]+)?', base, flags=re.IGNORECASE)
    if m:
        return m.group(2).split('_', 1)[0].strip()
    # Fallback: keep the last dashy token before underscore
    return base.split('_', 1)[0].strip().split()[-1]


def detect_temperature_from_folder(folder_path: str):
    """Detect temperature type from folder name instead of filename.
    
    Returns tuple: (temp_type, temp_config)
    """
    folder_name = os.path.basename(folder_path)
    parsed = parse_folder_name(folder_name)
    temp_type = parsed['temp_type']
    
    if temp_type in Config.TEMP_CONFIGS:
        return temp_type, Config.TEMP_CONFIGS[temp_type]
    else:
        logging.warning(f"Unknown temperature type '{temp_type}', defaulting to room")
        return 'room', Config.TEMP_CONFIGS['room']


def adjust_gamma(img, gamma=1.2):
    inv = 1.0 / max(gamma, 1e-6)
    table = (np.array([((i / 255.0) ** inv) * 255 for i in range(256)])).astype("uint8")
    return cv2.LUT(img, table)


def auto_roi(frame, name_det_result=None, default_band=(0.20, 0.85)):
    h = frame.shape[0]
    if name_det_result is not None:
        boxes = name_det_result[0].boxes.xyxy.cpu().numpy().astype(int)
        if boxes.shape[0] > 0:
            ys = []
            for (x1, y1, x2, y2) in boxes:
                ys += [y1, y2]
            top = max(0, min(ys) - int(0.10 * h))
            bot = min(h, max(ys) + int(0.10 * h))
            return top, bot
    t = int(h * default_band[0]); b = int(h * default_band[1])
    return t, b

def wait_for_file_access(filepath: str, timeout: int = 30) -> bool:
    """รอจนกว่าไฟล์จะพร้อมให้เข้าถึงได้"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with open(filepath, 'rb') as f:
                pass
            return True
        except (PermissionError, IOError):
            time.sleep(1)
    return False


def copy_folder_file_by_file(src_path: str, dst_path: str) -> bool:
    """Copy ทีละไฟล์ถ้า copytree ล้มเหลว"""
    try:
        os.makedirs(dst_path, exist_ok=True)
        
        copied_count = 0
        failed_files = []
        
        for root, dirs, files in os.walk(src_path):
            # สร้าง subdirectories
            rel_path = os.path.relpath(root, src_path)
            dst_dir = os.path.join(dst_path, rel_path)
            os.makedirs(dst_dir, exist_ok=True)
            
            # Copy แต่ละไฟล์
            for file in files:
                src_file = os.path.join(root, file)
                dst_file = os.path.join(dst_dir, file)
                
                # ลองหลายครั้ง
                for attempt in range(3):
                    try:
                        shutil.copy2(src_file, dst_file)
                        copied_count += 1
                        break
                    except PermissionError:
                        if attempt < 2:
                            time.sleep(2)
                        else:
                            failed_files.append(src_file)
                            logging.warning(f"Cannot copy file: {src_file}")
        
        if failed_files:
            logging.warning(f"Copied {copied_count} files, but {len(failed_files)} failed")
            for f in failed_files[:5]:  # แสดงแค่ 5 ไฟล์แรก
                logging.warning(f"  - {f}")
            if len(failed_files) > 5:
                logging.warning(f"  ... and {len(failed_files) - 5} more files")
            return len(failed_files) == 0
        
        logging.info(f"Successfully copied {copied_count} files")
        return True
        
    except Exception as e:
        logging.error(f"File-by-file copy failed: {e}")
        return False
    
def ensure_network_access(path: str, username: str, password: str, max_retries: int = 2) -> bool:
    """
    ตรวจสอบและ authenticate network path ถ้าจำเป็น
    """
    # ลองเข้าถึงก่อน
    try:
        os.listdir(path)
        return True
    except PermissionError:
        logging.warning(f"Permission denied on {path}, attempting to authenticate...")
    except Exception as e:
        logging.warning(f"Cannot access {path}: {e}, attempting to authenticate...")
    
    for attempt in range(max_retries):
        if connect_unc_path(path, username, password):
            time.sleep(2)  # รอให้ session update
            try:
                os.listdir(path)
                logging.info(f"Successfully authenticated and accessed: {path}")
                return True
            except Exception as e:
                logging.warning(f"Authentication seemed successful but still cannot access: {e}")
        
        if attempt < max_retries - 1:
            time.sleep(3)
    
    logging.error(f"Failed to authenticate and access: {path}")
    return False


def safe_copy_folder(src_path: str, dst_path: str, max_retries: int = 3) -> bool:
    """
    คัดลอกโฟลเดอร์อย่างปลอดภัย พร้อมระบบ retry, authentication และ fallback
    
    Features:
    - Auto-authenticate network paths ถ้าจำเป็น
    - ตรวจสอบสิทธิ์การเข้าถึงก่อน copy
    - รอให้ไฟล์วิดีโอ unlock
    - Retry mechanism หลายครั้ง
    - Fallback: copy ทีละไฟล์ถ้า copytree ล้มเหลว
    - Detailed logging
    """
    
    # ====== Network authentication ======
    # ตรวจสอบว่าเป็น UNC path หรือไม่
    if src_path.startswith('\\\\'):
        network_root = '\\\\' + src_path.split('\\')[2] + '\\' + src_path.split('\\')[3]
        if not ensure_network_access(network_root, USERNAME, PASSWORD):
            logging.error(f"Cannot authenticate source network path: {network_root}")
            return False
    
    if dst_path.startswith('\\\\'):
        network_root = '\\\\' + dst_path.split('\\')[2] + '\\' + dst_path.split('\\')[3]
        if not ensure_network_access(network_root, USERNAME, PASSWORD):
            logging.error(f"Cannot authenticate destination network path: {network_root}")
            return False
    
    # ====== Pre-flight checks ======
    if not os.path.exists(src_path):
        logging.error(f"Source folder does not exist: {src_path}")
        return False
    
    if not os.access(src_path, os.R_OK):
        logging.error(f"No read permission on source folder: {src_path}")
        return False
    
    # ตรวจสอบ destination parent
    dst_parent = os.path.dirname(dst_path)
    if not os.path.exists(dst_parent):
        try:
            os.makedirs(dst_parent, exist_ok=True)
        except Exception as e:
            logging.error(f"Cannot create destination parent directory: {e}")
            return False
    
    if not os.access(dst_parent, os.W_OK):
        logging.error(f"No write permission on destination: {dst_parent}")
        return False
    
    # ====== Log folder stats ======
    try:
        total_size = sum(os.path.getsize(os.path.join(root, f)) 
                        for root, dirs, files in os.walk(src_path) 
                        for f in files)
        file_count = sum(len(files) for _, _, files in os.walk(src_path))
        logging.info(f"Preparing to copy: {file_count} files, {total_size / (1024*1024):.2f} MB")
    except Exception as e:
        logging.warning(f"Cannot get folder stats: {e}")
    
    # ====== ตรวจสอบว่าไฟล์วิดีโอถูกล็อคหรือไม่ ======
    video_files = []
    for root, dirs, files in os.walk(src_path):
        for file in files:
            if file.lower().endswith(('.avi', '.mp4', '.mov', '.mkv')):
                video_files.append(os.path.join(root, file))
    
    if video_files:
        logging.info(f"Checking {len(video_files)} video file(s) for locks...")
        for video_file in video_files:
            if not wait_for_file_access(video_file, timeout=30):
                logging.error(f"Video file still locked after 30s timeout: {os.path.basename(video_file)}")
                return False
        logging.info("All video files are accessible")
    
    # ====== Handle existing destination ======
    original_dst_path = dst_path
    if os.path.exists(dst_path):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        folder_name = os.path.basename(dst_path)
        parent_dir = os.path.dirname(dst_path)
        dst_path = os.path.join(parent_dir, f"{folder_name}_{timestamp}")
        logging.info(f"Destination exists, using timestamped path: {os.path.basename(dst_path)}")
    
    # ====== Main copy with retry ======
    start_time = time.time()
    
    for attempt in range(max_retries):
        try:
            logging.info(f"Copy attempt {attempt + 1}/{max_retries}: {os.path.basename(src_path)}")
            
            shutil.copytree(src_path, dst_path)
            
            # ตรวจสอบว่า copy สำเร็จจริง
            if not os.path.exists(dst_path):
                logging.error(f"Copy appeared successful but destination not found: {dst_path}")
                return False
            
            elapsed = time.time() - start_time
            logging.info(f"Successfully copied folder in {elapsed:.2f}s: {src_path} -> {dst_path}")
            return True
            
        except PermissionError as e:
            logging.warning(f"Attempt {attempt + 1}/{max_retries}: Permission denied")
            logging.warning(f"  Error details: {str(e)}")
            
            # ลอง re-authenticate ถ้าเป็น network path
            if src_path.startswith('\\\\') or dst_path.startswith('\\\\'):
                logging.info("Attempting to re-authenticate network paths...")
                if src_path.startswith('\\\\'):
                    network_root = '\\\\' + src_path.split('\\')[2] + '\\' + src_path.split('\\')[3]
                    ensure_network_access(network_root, USERNAME, PASSWORD)
                if dst_path.startswith('\\\\'):
                    network_root = '\\\\' + dst_path.split('\\')[2] + '\\' + dst_path.split('\\')[3]
                    ensure_network_access(network_root, USERNAME, PASSWORD)
            
            if attempt < max_retries - 1:
                wait_time = 5 * (attempt + 1)
                logging.info(f"  Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logging.error(f"All {max_retries} copytree attempts failed")
                
        except OSError as e:
            logging.error(f"OS error during copy (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                logging.error(f"All {max_retries} copytree attempts failed")
                
        except Exception as e:
            logging.error(f"Unexpected error copying folder: {type(e).__name__}: {e}")
            return False
    
    # ====== Fallback: copy file-by-file ======
    logging.info("Falling back to file-by-file copy method...")
    
    # ลบ partial copy ถ้ามี
    if os.path.exists(dst_path):
        try:
            shutil.rmtree(dst_path)
            logging.info("Cleaned up partial copy")
        except Exception as e:
            logging.warning(f"Cannot cleanup partial copy: {e}")
    
    success = copy_folder_file_by_file(src_path, dst_path)
    
    if success:
        elapsed = time.time() - start_time
        logging.info(f"File-by-file copy completed in {elapsed:.2f}s")
    else:
        logging.error(f"Both copytree and file-by-file methods failed for: {src_path}")
    
    return success

def safe_remove_folder(folder_path: str) -> bool:
    """
    ลบโฟลเดอร์หลังประมวลผลเสร็จ
    """
    try:
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
            logging.info(f"Removed processed folder: {folder_path}")
            return True
        return True  # ถ้าไม่มีโฟลเดอร์อยู่แล้วถือว่าสำเร็จ
    except Exception as e:
        logging.error(f"Failed to remove folder {folder_path}: {e}")
        return False


class CenterConsensus:
    def __init__(self, merge_radius=32):
        self.merge_radius = merge_radius
        self.clusters = []  # {'xy': np.array([x,y]), 'n': int}

    def _dist(self, a, b):
        return float(np.linalg.norm(a - b))

    def add(self, pts):
        for x, y in pts:
            p = np.array([float(x), float(y)], dtype=np.float32)
            assigned = False
            for c in self.clusters:
                if self._dist(p, c['xy']) <= self.merge_radius:
                    c['xy'] = (c['xy'] * c['n'] + p) / (c['n'] + 1)
                    c['n'] += 1
                    assigned = True
                    break
            if not assigned:
                self.clusters.append({'xy': p, 'n': 1})

    def topk(self, k=10):
        return sorted(self.clusters, key=lambda c: c['n'], reverse=True)[:k]

    def centers(self, k=10):
        return [(int(c['xy'][0]), int(c['xy'][1])) for c in self.topk(k)]

    def has_k_stable(self, k=3, min_hits=3):
        return sum(1 for c in self.clusters if c['n'] >= min_hits) >= k


# Circle helpers (from find_tune logic)
def find_circles_multi(gray_roi):
    for ps in Config.CIRCLE_PARAM_SETS:
        circles = cv2.HoughCircles(
            gray_roi, cv2.HOUGH_GRADIENT,
            dp=ps["dp"], minDist=Config.CIRCLE_MIN_DIST,
            param1=ps["param1"], param2=ps["param2"],
            minRadius=ps["minRadius"], maxRadius=ps["maxRadius"]
        )
        if circles is not None:
            yield np.uint16(np.around(circles[0, :]))


def gate_centers_by_labels(centers, label_pts, gate):
    if not label_pts:
        return centers
    out = []
    for c in centers:
        dmin = min(np.linalg.norm(np.array(c) - np.array(lp)) for lp in label_pts)
        if dmin <= gate:
            out.append(c)
    return out


def match_labels_to_circles_unique(label_centers, candidate_centers, max_dist=None):
    # why: enforce 1:1 label→circle assignment with distance gate
    if len(candidate_centers) < len(label_centers):
        return {}

    labels = [l for l, _ in label_centers]
    P = np.array([p for _, p in label_centers], dtype=np.float32)
    C = np.array(candidate_centers, dtype=np.float32)

    cost = np.linalg.norm(P[:, None, :] - C[None, :, :], axis=2)
    if max_dist is not None:
        cost = np.where(cost <= float(max_dist), cost, 1e9)

    rows, cols = linear_sum_assignment(cost)
    out = {}
    for r, c in zip(rows, cols):
        if cost[r, c] >= 1e9:
            return {}
        out[labels[r]] = (int(C[c, 0]), int(C[c, 1]))

    if len(out) != len(set(out.values())):
        return {}
    return out


def is_far_enough(new_center, centers, min_dist=50):
    new_center = np.array(new_center, dtype=np.float32)
    for c in centers:
        c = np.array(c, dtype=np.float32)
        if np.linalg.norm(new_center - c) < min_dist:
            return False
    return True

def is_readable_clip(path_or_name: str) -> bool:
    """ตรวจสอบว่าชื่อไฟล์ขึ้นต้นด้วย 'P703 CAB DBL' หรือ 'P703 DBL CAB' เท่านั้น"""
    stem = os.path.splitext(os.path.basename(str(path_or_name)))[0]
    if not has_allowed_prefix(stem):
        log_limited_warning('invalid_file_format', 
            f"File '{os.path.basename(str(path_or_name))}' ignored: must start with 'P703 CAB DBL' or 'P703 DBL CAB'",
            key=os.path.basename(str(path_or_name)))
        return False
    
    return True

def has_allowed_prefix(name: str) -> bool:
    """
    True เฉพาะชื่อที่ขึ้นต้นด้วย:
      - 'P703 CAB DBL' หรือ
      - 'P703 DBL CAB'
    ไม่สนตัวพิมพ์เล็ก/ใหญ่ และรองรับช่องว่างหลายช่อง
    """
    stem = str(name).strip()
    return re.match(r'^(P703\s+CAB\s+DBL|P703\s+DBL\s+CAB)\b', stem, flags=re.IGNORECASE) is not None


# ======= OCR CLASS =======
class OCR:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        try:
            self.processor = AutoProcessor.from_pretrained(
                Config.OCR_MODEL_PATH, trust_remote_code=True, use_fast=True
            )
            self.model = AutoModelForVision2Seq.from_pretrained(
                Config.OCR_MODEL_PATH, trust_remote_code=True, torch_dtype=torch.float32
            ).to(self.device).eval()
        except Exception as e:
            logging.error(f"Failed to load OCR model: {e}")
            raise

    def _is_valid_cop_format(self, cop_str: str) -> bool:
        """
        Validate COPNo format: should be like 2504355-CAB-LH-RT or similar
        Pattern: [7digits]-[3letters]-[2letters]-[2letters]
        """
        if not cop_str or not isinstance(cop_str, str):
            return False
        
        # Remove whitespace and convert to uppercase for checking
        cop_clean = cop_str.strip().upper()
        
        # Pattern: 7 digits, dash, 3 letters, dash, 2 letters, dash, 2 letters
        pattern = r'^\d{7}-[A-Z]{3}-[A-Z]{2}-[A-Z]{2}$'
        return bool(re.match(pattern, cop_clean))

    def extract_ms_and_cop(self, image: Image.Image, max_new_tokens: int = 32) -> dict:
        if image is None:
            return {"error": "No image provided."}

        # Enhanced prompt for better COP detection
        query = (
            "Read the time (in milliseconds) and COP NO shown near the top-left corner of the image. "
            "The COP NO should be exactly 7 digits followed by dashes and letters (like 2504355-CAB-LH-RT). "
            "Return exactly in this format: ms=13.6, cop=2504355-CAB-LH-RT "
            "(The time should be a number only, without '+' or '-' sign)"
        )

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": query}
            ]
        }]

        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[prompt], images=[image], return_tensors="pt",
            padding=True, truncation=True, max_length=2048
        ).to(self.device)

        streamer = TextIteratorStreamer(self.processor, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = {
            **inputs, "streamer": streamer, "max_new_tokens": max_new_tokens,
            "do_sample": True, "temperature": 0.3, "top_p": 0.9, "top_k": 30, "repetition_penalty": 1.1
        }

        Thread(target=self.model.generate, kwargs=generation_kwargs).start()
        output_text = "".join(token for token in streamer).strip().replace("<|im_end|>", "").strip()

        match = re.search(r"ms\s*=\s*([+-]?\d+\.?\d*)[, ]+cop\s*=\s*([A-Za-z0-9\-]+)", output_text, re.IGNORECASE)
        if match:
            ms_val = match.group(1).lstrip("+")
            cop_val = match.group(2).strip()
            
            # Validate COP format
            if self._is_valid_cop_format(cop_val):
                return {"ms": ms_val, "cop": cop_val}
            else:
                logging.warning(f"Invalid COP format detected: '{cop_val}' from OCR output: '{output_text}'")
                return {"ms": ms_val, "cop_invalid": cop_val, "error": "Invalid COP format"}
        
        return {"error": "Could not parse OCR output", "raw_output": output_text}


# ======= MAIN PROCESSING CLASS =======
class AirbagProcessor:
    def __init__(self):
        self.setup_models()
        self.ocr = OCR()

    def setup_models(self):
        try:
            self.yolo_object_model = YOLO(Config.YOLO_OBJECT_MODEL_PATH)
            self.yolo_name_model = YOLO(Config.YOLO_NAME_MODEL_PATH)
            self.yolo_segmentation_model = YOLO(Config.YOLO_SEGMENTATION_MODEL_PATH)
            self.sam = sam_model_registry["vit_h"](checkpoint=Config.SAM_CHECKPOINT).to("cuda")
            self.predictor = SamPredictor(self.sam)
            logging.info("✅ All models loaded successfully")
        except Exception as e:
            logging.error(f"❌ Failed to load models: {e}")
            raise

    def _ocr_ms_cop_from_frame(self, frame_bgr: np.ndarray) -> dict:
        try:
            pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            return self.ocr.extract_ms_and_cop(pil)
        except Exception as e:
            logging.error(f"OCR exception: {e}")
            return {"error": str(e)}
        
    def _ocr_with_cop_retry(self, frame_bgr: np.ndarray, max_retries: int = 3) -> dict:
        """
        OCR with retry logic specifically for COPNo validation
        """
        try:
            pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            
            for attempt in range(max_retries):
                result = self.ocr.extract_ms_and_cop(pil)
                
                # If we got a valid COP, return immediately
                if 'cop' in result and not result.get('error'):
                    return result
                
                # If invalid COP format but we got some result, try again with different parameters
                if 'cop_invalid' in result:
                    logging.warning(f"OCR attempt {attempt + 1}: Invalid COP format '{result['cop_invalid']}', retrying...")
                    continue
                
                # If complete failure, try again
                logging.warning(f"OCR attempt {attempt + 1}: Failed to extract COP, retrying...")
            
            # After all retries failed, return the last result
            logging.error("OCR failed to get valid COP after all retries")
            return {"error": "Failed to extract valid COP after retries"}
            
        except Exception as e:
            logging.error(f"OCR exception: {e}")
            return {"error": str(e)}

    def process_folder(self, folder_path: str):
        """Process all videos in a folder based on folder name format."""
        try:
            folder_name = os.path.basename(folder_path)
            
            # Parse SN from folder name
            parsed = parse_folder_name(folder_name)
            if not parsed['valid']:
                logging.warning(f"Skipping folder with invalid format: {folder_name}")
                return {
                    'folder_name': folder_name,
                    'error': 'invalid_folder_name_format',
                    'processing_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }

            module_sn = parsed.get("module_sn", "") or ""
            
            # *** เพิ่มส่วนคัดลอกโฟลเดอร์ ***
            processing_folder_path = os.path.join(Config.PROCESSING_DIR, folder_name)
            
            # สร้างโฟลเดอร์ PROCESSING_DIR ถ้ายังไม่มี
            os.makedirs(Config.PROCESSING_DIR, exist_ok=True)
            
            # คัดลอกโฟลเดอร์จากต้นทางมายังโฟลเดอร์ประมวลผล
            if not safe_copy_folder(folder_path, processing_folder_path):
                return {
                    'folder_name': folder_name,
                    'error': 'failed_to_copy_folder',
                    'processing_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            
            # ใช้ processing_folder_path แทน folder_path สำหรับประมวลผล
            save_dir = os.path.join(Config.OUTPUT_DIR, folder_name)
            base_name = module_sn or folder_name

            logging.info(f"Processing folder {folder_name} from copied location (Module: {module_sn})")

            # *** ประมวลผลจากโฟลเดอร์ที่คัดลอกมา ***
            video_extensions = ('.avi', '.mp4', '.mov', '.mkv')
            all_video_files = [os.path.join(processing_folder_path, f)  # ใช้ processing_folder_path
                            for f in os.listdir(processing_folder_path)    # ใช้ processing_folder_path
                            if f.lower().endswith(video_extensions)]

            if not all_video_files:
                # ลบโฟลเดอร์ที่คัดลอกมาถ้าไม่มีวิดีโอ
                safe_remove_folder(processing_folder_path)
                logging.warning(f"No video files found in folder: {processing_folder_path}")
                return {
                    'folder_name': folder_name,
                    'error': 'no_video_files',
                    'processing_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
            video_files = [p for p in all_video_files if is_readable_clip(p)]
            
            if not video_files:
                # ลบโฟลเดอร์ที่คัดลอกมาถ้าไม่มีคลิปที่อ่านได้
                safe_remove_folder(processing_folder_path)
                log_limited_warning('invalid_file_format', 
                    f"No readable clips (_CxxxH001S0001 with P703 prefix) in folder: {folder_name}")
                return {
                    'folder_name': folder_name,
                    'error': 'no_readable_clips',
                    'processing_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }

            # เลือกไฟล์วิดีโอ
            side_candidates = [p for p in video_files if re.search(r"(?i)(?:^|[_\-\s])side$", Path(p).stem)]
            if len(video_files) == 1:
                selected_path = video_files[0]
            else:
                selected_path = side_candidates[0] if side_candidates else max(video_files, key=lambda p: os.path.getmtime(p))

            video_path = selected_path

            # Temperature จากชื่อไฟล์
            video_stem = os.path.splitext(os.path.basename(video_path))[0]
            temp_type, temp_config = detect_temperature_from_filename(video_stem)
            temp_config = temp_config.copy()
            temp_config['temp_type'] = temp_type

            logging.info(f"Processing folder {folder_name} (Temperature: {temp_type}, Module: {module_sn})")

            # Analyze วิดีโอ
            results = self._analyze_video(video_path, temp_config, save_dir=save_dir, base_name=base_name)
            results['video_name'] = os.path.basename(video_path)
            results['folder_name'] = folder_name
            results['temperature_type'] = temp_type
            results['module_sn'] = module_sn
            results['processing_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # *** ลบโฟลเดอร์ที่คัดลอกมาหลังประมวลผลเสร็จ ***
            if Config.DELETE_COPIED_FOLDER_AFTER_PROCESSING:
                cleanup_success = safe_remove_folder(processing_folder_path)
                if not cleanup_success:
                    logging.warning(f"Failed to cleanup processing folder: {processing_folder_path}")
                    # ไม่ return error เพราะการประมวลผลสำเร็จแล้ว
            else:
                logging.info(f"Keeping copied folder as configured: {processing_folder_path}")
                # ไม่ return error เพราะการประมวลผลสำเร็จแล้ว
            
            return results

        except TimeoutError as e:
            # ลบโฟลเดอร์ที่คัดลอกมาในกรณี timeout
            processing_folder_path = os.path.join(Config.PROCESSING_DIR, os.path.basename(folder_path))
            if Config.DELETE_COPIED_FOLDER_AFTER_PROCESSING:
                safe_remove_folder(processing_folder_path)
            
            logging.error(f"Timeout while processing {folder_path}: {e}")
            if str(Config.TIMEOUT_ACTION).lower() == "restart":
                schedule_self_restart(delay=int(Config.RESTART_GRACE_SECONDS))
            return {
                'folder_name': os.path.basename(folder_path),
                'error': f'timeout: {e}',
                'processing_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        except Exception as e:
            # ลบโฟลเดอร์ที่คัดลอกมาในกรณี error
            processing_folder_path = os.path.join(Config.PROCESSING_DIR, os.path.basename(folder_path))
            if Config.DELETE_COPIED_FOLDER_AFTER_PROCESSING:
                safe_remove_folder(processing_folder_path)
            
            logging.error(f"Error processing folder {folder_path}: {e}")
            return {
                'folder_name': os.path.basename(folder_path),
                'error': str(e),
                'processing_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

    def process_video(self, video_path: str):
        """Legacy method for backward compatibility."""
        try:
            if not is_readable_clip(video_path):
                logging.warning(f"Skipping non-readable clip (requires allowed prefix + _CxxxH001S0001): {os.path.basename(video_path)}")
                return {
                    'video_name': os.path.splitext(os.path.basename(video_path))[0],
                    'error': 'non_readable_clip',
                    'processing_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }

            video_name = os.path.splitext(os.path.basename(video_path))[0]
            temp_type, temp_config = detect_temperature_from_filename(video_name)
            logging.info(f"🎬 Processing {video_name} (Temperature from filename: {temp_type})")
            save_dir = os.path.join(Config.OUTPUT_DIR, video_name)
            base_name = extract_module_sn_from_video_name(video_name)

            results = self._analyze_video(video_path, temp_config, save_dir=save_dir, base_name=base_name)
            results['video_name'] = video_name
            results['temperature_type'] = temp_type
            results['processing_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            return results
        except Exception as e:
            logging.error(f"❌ Error processing {video_path}: {e}")
            return {
                'video_name': os.path.splitext(os.path.basename(video_path))[0],
                'error': str(e),
                'processing_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

    def _analyze_video(self, video_path: str, temp_config: dict, save_dir: str, base_name: str):
        confidence_scores = []  # Initialize confidence_scores as an empty list

        results = {
            'explosion_frame': None,
            'full_deployment_frame': None,
            'fr1_hit_frame': None,
            'fr2_hit_frame': None,
            're3_hit_frame': None,
            'explosion_time_ms': '',
            'fr1_hit_time_ms': '',
            'fr2_hit_time_ms': '',
            're3_hit_time_ms': '',
            'full_deployment_time_ms': '',
            'cop_number': '',
            'image_paths': {
                'explosion': '',
                'fr1': '',
                'fr2': '',
                're3': '',
                'full_deployment': ''
            }
        }
        
        ocr_retry_frames = []
        
        video_deadline = _deadline(Config.PER_VIDEO_TIMEOUT_SECONDS)
        detect_deadline = _deadline(Config.DETECTION_STAGE_TIMEOUT_SECONDS)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception(f"Cannot open video: {video_path}")

        try:
            # ===== Detection phase: robust circle + label matching (find_tune) =====
            frame_count = 0
            consensus = CenterConsensus(merge_radius=Config.MERGE_RADIUS)
            label_to_center = {}
            fixed_centers = []
            done_detecting = False

            frame17_bgr = None
            frame18_bgr = None

            while cap.isOpened() and not done_detecting:
                _check_deadline(video_deadline, "Per-video")
                _check_deadline(detect_deadline, "Detection")

                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
                
                if frame_count >= Config.EARLY_SKIP_NO_CIRCLE_FRAME and len(consensus.clusters) == 0:
                    error_msg = f"Early-skip: no circles detected by frame {Config.EARLY_SKIP_NO_CIRCLE_FRAME}; likely front-view video"
                    log_limited_warning('no_circles_early_skip', 
                        f"{error_msg} for clip {os.path.basename(video_path)}")
                    
                    cap.release()
                    
                    self._cleanup_unprocessable_video(video_path)

                    raise TimeoutError(error_msg)

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                gray = clahe.apply(gray)
                gray = adjust_gamma(gray, gamma=1.2)

                name_detections_try = self.yolo_name_model.predict(
                    source=frame, conf=0.25, show=False, save=False, stream=False, verbose=False
                )

                roi_top, roi_bottom = auto_roi(frame, name_detections_try, default_band=(0.20, 0.85))
                gray_roi = cv2.GaussianBlur(gray[roi_top:roi_bottom, :], (9, 9), sigmaX=2, sigmaY=2)

                found_this_frame = []
                for cset in find_circles_multi(gray_roi):
                    for c in cset:
                        center = (int(c[0]), int(c[1]) + roi_top)
                        if is_far_enough(center, fixed_centers, min_dist=45):
                            found_this_frame.append(center)
                    if len(found_this_frame) + len(fixed_centers) >= 3:
                        break

                if found_this_frame:
                    consensus.add(found_this_frame)
                    logging.info(f"Frame {frame_count}: added {len(found_this_frame)} centers (clusters={len(consensus.clusters)})")
                else:
                    logging.info(f"Frame {frame_count}: no new circles (clusters={len(consensus.clusters)})")

                # Attempt to finalize mapping when stable enough
                try:
                    name_detections = self.yolo_name_model.predict(
                        source=frame, conf=0.30, show=False, save=False, stream=False, verbose=False
                    )
                    
                    try:
                        boxes = name_detections[0].boxes.xyxy.cpu().numpy().astype(int)
                        classes = name_detections[0].boxes.cls.cpu().numpy().astype(int)
                        classnames = name_detections[0].names
                        confs = name_detections[0].boxes.conf.detach().cpu().numpy().tolist()
                    
                        label_confs = {}
                        for (box, cls_id, c) in zip(boxes, classes, confs):
                            lbl = classnames[cls_id]
                            if lbl in {'FR1', 'FR2', 'RE3'}:
                                label_confs[lbl] = max(float(c), label_confs.get(lbl, 0.0))
                        if label_confs:
                            confidence_scores.append(sum(label_confs.values()) / len(label_confs))
                    except Exception:
                        pass
                    
                    # boxes = name_detections[0].boxes.xyxy.cpu().numpy().astype(int)
                    # classes = name_detections[0].boxes.cls.cpu().numpy().astype(int)
                    # classnames = name_detections[0].names

                    target_labels = {'FR1', 'FR2', 'RE3'}
                    label_best = {}
                    for box, cls_id in zip(boxes, classes):
                        x1, y1, x2, y2 = box
                        label = classnames[cls_id]
                        if label in target_labels:
                            cx = (x1 + x2) // 2
                            cy = (y1 + y2) // 2
                            area = (x2 - x1) * (y2 - y1)
                            if label not in label_best or area > label_best[label][1]:
                                label_best[label] = ((cx, cy), area)

                    label_centers = [(lbl, pt_area[0]) for lbl, pt_area in label_best.items() if lbl in target_labels]
                    label_pts = [pt for _, pt in label_centers]

                    candidate_centers = consensus.centers(k=10)
                    candidate_centers = gate_centers_by_labels(candidate_centers, label_pts, gate=Config.GATE_RADIUS)

                    if (
                        frame_count >= Config.WARMUP_MIN_FRAMES and
                        consensus.has_k_stable(k=3, min_hits=Config.MIN_HITS_PER_CENTER) and
                        len(candidate_centers) >= 3 and
                        len(label_centers) == 3
                    ):
                        temp_map = match_labels_to_circles_unique(label_centers, candidate_centers, max_dist=Config.GATE_RADIUS)
                        if len(temp_map) == 3 and all(k in temp_map for k in ['FR1', 'FR2', 'RE3']):
                            label_to_center = temp_map
                            fixed_centers = [label_to_center['FR1'], label_to_center['FR2'], label_to_center['RE3']]
                            done_detecting = True
                            logging.info("✅ Completed detection with unique label-to-circle assignment (stable & gated)")
                        else:
                            logging.info("Assignment not ready (unique/gated not satisfied)")
                    else:
                        logging.info("Not stable enough or labels/centers incomplete")
                except Exception as e:
                    logging.warning(f"Name+circle finalize attempt failed: {e}")

                # Fallback after max frames using label boxes directly
                if (not done_detecting) and frame_count >= Config.MAX_FRAMES_FOR_CIRCLE_SEARCH:
                    boxes = name_detections_try[0].boxes.xyxy.cpu().numpy().astype(int)
                    classes = name_detections_try[0].boxes.cls.cpu().numpy().astype(int)
                    classnames = name_detections_try[0].names

                    temp_map = {}
                    for box, cls_id in zip(boxes, classes):
                        x1, y1, x2, y2 = box
                        label = classnames[cls_id]
                        if label in ['FR1', 'FR2', 'RE3']:
                            cx = (x1 + x2) // 2
                            cy = (y1 + y2) // 2
                            temp_map[label] = (cx, cy)
                    if all(l in temp_map for l in ['FR1', 'FR2', 'RE3']):
                        label_to_center = temp_map
                        fixed_centers = list(temp_map.values())
                        done_detecting = True
                        logging.info("✅ Fallback: using label centers for FR1/FR2/RE3")
                    else:
                        logging.error("Fallback failed: not enough labels detected - likely front-view video")
                        cap.release()
                        self._cleanup_unprocessable_video(video_path)
                        raise TimeoutError("Cannot detect FR1/FR2/RE3 labels - likely front-view video")
                        
                if frame_count >= max(2 * Config.MAX_FRAMES_FOR_CIRCLE_SEARCH, 250) and not done_detecting:
                    cap.release()
                    self._cleanup_unprocessable_video(video_path)
                    raise TimeoutError("Detection phase exhausted frames without stable assignment - likely unsuitable video")

            if not done_detecting:
                cap.release()
                self._cleanup_unprocessable_video(video_path)
                raise TimeoutError("Detection phase timeout/failed to determine centers")

            # ===== Main processing: SAM masks + hits + explosion (17 vs 18) =====
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            logging.info(f"[Stage: Detect Explosion & Circles] start — total_frames={total_frames}")
            frame_count = 0
            last_logged_pct = -1
            frame17_mask = None
            frame18_mask = None
            explosion_resolved = False
            hit_center_labels = set()
            
            temp_type = temp_config.get('temp_type', 'room')
            use_sam = (temp_type == 'cold')
            
            logging.info(f"Using {'SAM+YOLO' if use_sam else 'Segmentation'} for temperature: {temp_type}")
        
            # Progress tracking variables
            detection_progress_start = 10  # Start progress reporting at 10%
            detection_progress_end = 85    # End detection phase at 85%
            deployment_progress_end = 100  # Full deployment ends at 100%

            while cap.isOpened():
                _check_deadline(video_deadline, "Per-video")

                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1

                # Smooth progress calculation
                if total_frames > 0:
                    # Base progress from frame processing
                    frame_pct = int(frame_count * 100 / total_frames)
                    
                    # Scale to detection phase range (10-85%)
                    scaled_pct = detection_progress_start + int((frame_pct / 100.0) * (detection_progress_end - detection_progress_start))
                    
                    # Add bonus progress for completed tasks
                    bonus_pct = 0
                    if explosion_resolved:
                        bonus_pct += 5
                    if len(hit_center_labels) > 0:
                        bonus_pct += len(hit_center_labels) * 3  # 3% per detected hit
                        
                    final_pct = min(detection_progress_end, scaled_pct + bonus_pct)
                    
                    # Report progress every 2% change to reduce noise
                    if abs(final_pct - last_logged_pct) >= 2:
                        logging.info(f"[Stage: Detect Explosion & Circles] progress={final_pct}% (frame {frame_count}/{total_frames}, explosion:{explosion_resolved}, hits:{len(hit_center_labels)})")
                        last_logged_pct = final_pct
                        
                if use_sam:
                    # ใช้ SAM+YOLO สำหรับ Cold
                    yolo_results = self.yolo_object_model.predict(
                        source=frame, conf=Config.CONFIDENCE_THRESHOLD, show=False, save=False, stream=False, verbose=False
                    )
                    
                    try:
                        confs = yolo_results[0].boxes.conf.detach().cpu().numpy().tolist()
                        if confs:
                            confidence_scores.append(float(sum(confs) / len(confs)))
                    except Exception:
                        pass
                    
                    boxes = yolo_results[0].boxes.xyxy.cpu().numpy().astype(int)
                    if len(boxes) == 0:
                        continue

                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.predictor.set_image(image_rgb)

                    for box in boxes:
                        x1, y1, x2, y2 = box
                        masks, scores, _ = self.predictor.predict(
                            box=np.array([x1, y1, x2, y2])[None, :], multimask_output=True
                        )
                        try:
                            if scores is not None and len(scores) > 0:
                                confidence_scores.append(float(np.max(scores)))  # SAM mask score ∈ [0,1]
                        except Exception:
                            pass
                        best_mask = masks[np.argmax(scores)]
                        
                        # Explosion determination (frame 17 vs 18)
                        if frame_count == 17:
                            frame17_mask = best_mask.copy()
                            frame17_bgr = frame.copy()
                        elif frame_count == 18:
                            frame18_mask = best_mask.copy()
                            frame18_bgr = frame.copy()
                            if frame17_mask is not None and not explosion_resolved:
                                diff = np.logical_xor(frame17_mask.astype(bool), frame18_mask.astype(bool)).astype(np.uint8)
                                motion_score = int(np.sum(diff))
                                if motion_score > Config.MOTION_THRESHOLD:
                                    results['explosion_frame'] = 18
                                    chosen_frame = frame18_bgr
                                    chosen_mask = frame18_mask
                                else:
                                    results['explosion_frame'] = 17
                                    chosen_frame = frame17_bgr
                                    chosen_mask = frame17_mask
                                    
                                # บันทึกรูปภาพ explosion พร้อม mask
                                explosion_image_path = os.path.join(save_dir, f"{base_name}_explosion_f{results['explosion_frame']}.jpg")
                                saved_path = _save_frame_with_mask(chosen_frame, chosen_mask, explosion_image_path, overlay_color=(255, 255, 0))  # Red for explosion
                                if saved_path:
                                    results['image_paths']['explosion'] = saved_path
                
                                # OCR on chosen explosion frame
                                o = self._ocr_with_cop_retry(chosen_frame)
                                if 'ms' in o:
                                    results['explosion_time_ms'] = o['ms']
                                if 'cop' in o and not results['cop_number']:
                                    results['cop_number'] = o['cop']
                                if not results['cop_number']:
                                    ocr_retry_frames.append(('explosion', chosen_frame))
                                    
                                logging.info(f"Explosion resolved at frame {results['explosion_frame']}")
                                explosion_resolved = True

                        # Label hit check
                        for label, center in label_to_center.items():
                            if label in hit_center_labels:
                                continue
                            cx, cy = center
                            if 0 <= cy < best_mask.shape[0] and 0 <= cx < best_mask.shape[1] and best_mask[cy, cx]:
                                results[f'{label.lower()}_hit_frame'] = frame_count
                                hit_center_labels.add(label)
                                
                                # กำหนดสีตาม label
                                label_colors = {
                                    'FR1': (255, 255, 0),    # Blue
                                    'FR2': (255, 255, 0),    # Green  
                                    'RE3': (255, 255, 0)     # Red
                                }
                                color = label_colors.get(label, (0, 255, 0))  # Default cyan
                                
                                # บันทึกรูปภาพสำหรับ hit frame
                                hit_image_path = os.path.join(save_dir, f"{base_name}_{label.lower()}_f{frame_count}.jpg")
                                saved_path = _save_frame_with_mask(frame, best_mask, hit_image_path, overlay_color=color)
                                if saved_path:
                                    results['image_paths'][label.lower()] = saved_path
                                
                                # Use retry logic for hit frame OCR
                                o = self._ocr_with_cop_retry(frame)
                                key = f"{label.lower()}_hit_time_ms"
                                if 'ms' in o:
                                    results[key] = o['ms']
                                if 'cop' in o and not results['cop_number']:
                                    results['cop_number'] = o['cop']
                                # Store frame for later retry if COP still empty
                                if not results['cop_number']:
                                    ocr_retry_frames.append((label, frame.copy()))
                                    
                                logging.info(f"Detected {label} at frame {frame_count}")
                        
                        # เพิ่มเฉพาะ mask แรกที่พบ (สำหรับ SAM)
                        break
                else:

                    # ใช้ YOLO Segmentation แทน YOLO Object + SAM
                    yolo_results = self.yolo_segmentation_model.predict(
                        source=frame, conf=Config.CONFIDENCE_THRESHOLD, show=False, save=False, stream=False, verbose=False
                    )
                    
                    try:
                        if hasattr(yolo_results[0], 'boxes') and yolo_results[0].boxes is not None:
                            confs = yolo_results[0].boxes.conf.detach().cpu().numpy().tolist()
                            if confs:
                                confidence_scores.append(float(sum(confs) / len(confs)))
                    except Exception:
                        pass
                
                    # ตรวจสอบว่ามี segmentation masks หรือไม่
                    if yolo_results[0].masks is None or len(yolo_results[0].masks) == 0:
                        continue

                    # วนลูปผ่าน masks แต่ละตัว
                    for i, mask_data in enumerate(yolo_results[0].masks.data):
                        mask = mask_data.cpu().numpy().astype(np.uint8)
                        
                        # Resize mask ให้ตรงกับขนาด frame
                        if mask.shape != frame.shape[:2]:
                            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

                        # Explosion determination (frame 17 vs 18)
                        if frame_count == 17:
                            frame17_mask = mask.copy()
                            frame17_bgr = frame.copy()
                        elif frame_count == 18:
                            frame18_mask = mask.copy()
                            frame18_bgr = frame.copy()
                            if frame17_mask is not None and not explosion_resolved:
                                diff = np.logical_xor(frame17_mask.astype(bool), frame18_mask.astype(bool)).astype(np.uint8)
                                motion_score = int(np.sum(diff))
                                if motion_score > Config.MOTION_THRESHOLD:
                                    results['explosion_frame'] = 18
                                    chosen_frame = frame18_bgr
                                    chosen_mask = frame18_mask
                                else:
                                    results['explosion_frame'] = 17
                                    chosen_frame = frame17_bgr
                                    chosen_mask = frame17_mask
                                    
                                # บันทึกรูปภาพ explosion พร้อม mask
                                explosion_image_path = os.path.join(save_dir, f"{base_name}_explosion_f{results['explosion_frame']}.jpg")
                                saved_path = _save_frame_with_mask(chosen_frame, chosen_mask, explosion_image_path, overlay_color=(255, 255, 0))  # Red
                                if saved_path:
                                    results['image_paths']['explosion'] = saved_path
                                    
                                # OCR on chosen explosion frame
                                o = self._ocr_ms_cop_from_frame(chosen_frame)
                                if 'ms' in o:
                                    results['explosion_time_ms'] = o['ms']
                                if 'cop' in o and not results['cop_number']:
                                    results['cop_number'] = o['cop']
                                logging.info(f"✅ Explosion resolved at frame {results['explosion_frame']}")
                                explosion_resolved = True

                        # Label hit check
                        for label, center in label_to_center.items():
                            if label in hit_center_labels:
                                continue
                            cx, cy = center
                            if 0 <= cy < mask.shape[0] and 0 <= cx < mask.shape[1] and mask[cy, cx]:
                                results[f'{label.lower()}_hit_frame'] = frame_count
                                hit_center_labels.add(label)
                                
                                # กำหนดสีตาม label
                                label_colors = {
                                    'FR1': (255, 255, 0),    # Blue
                                    'FR2': (255, 255, 0),    # Green  
                                    'RE3': (255, 255, 0)     # Red
                                }
                                color = label_colors.get(label, (255, 255, 0))  # Default cyan
                                
                                # บันทึกรูปภาพสำหรับ hit frame
                                hit_image_path = os.path.join(save_dir, f"{base_name}_{label.lower()}_f{frame_count}.jpg")
                                saved_path = _save_frame_with_mask(frame, mask, hit_image_path, overlay_color=color)
                                if saved_path:
                                    results['image_paths'][label.lower()] = saved_path
                                
                                o = self._ocr_ms_cop_from_frame(frame)
                                key = f"{label.lower()}_hit_time_ms"
                                if 'ms' in o:
                                    results[key] = o['ms']
                                if 'cop' in o and not results['cop_number']:
                                    results['cop_number'] = o['cop']
                                logging.info(f"🎯 Detected {label} at frame {frame_count}")
                    # break

                # early-exit condition
                if explosion_resolved and len(hit_center_labels) == 3:
                    final_detection_pct = min(detection_progress_end, last_logged_pct + 5)
                    logging.info(f"[Stage: Detect Explosion & Circles] progress={final_detection_pct}% (early exit - all targets found)")
                    break

                # if early_stop:
                #     cur_pct = int(frame_count * 100 / total_frames) if total_frames > 0 else 100
                #     logging.info(f"⏭️ Early exit at frame {frame_count}: explosion + FR1/FR2/RE3 resolved — progress={cur_pct}%")
                #     break  # break frames loop

            cap.release()
            
            if not results['cop_number'] and ocr_retry_frames:
                logging.info("Attempting final COP retry on stored frames...")
                for frame_type, frame_bgr in ocr_retry_frames:
                    o = self._ocr_with_cop_retry(frame_bgr, max_retries=5)  # More retries for final attempt
                    if 'cop' in o:
                        results['cop_number'] = o['cop']
                        logging.info(f"Successfully extracted COP from {frame_type} frame: {o['cop']}")
                        break

            # ===== Full deployment analysis (plateau on smoothed SAM area) =====
            logging.info(f"[Stage: Full Deployment] progress={detection_progress_end + 5}% (starting deployment analysis)")
            
            deploy_deadline = _deadline(Config.DEPLOY_STAGE_TIMEOUT_SECONDS)
            
            deployment_frame, deployment_frame_bgr = self._analyze_full_deployment_with_deadline(
                video_path, temp_config, deploy_deadline
            )
            
            results['full_deployment_frame'] = deployment_frame
            if deployment_frame_bgr is not None:
                path_fd = os.path.join(save_dir, f"{base_name}_full_deploy_f{deployment_frame}.jpg")
                
                temp_type = temp_config.get('temp_type', 'room')
                use_sam = (temp_type == 'cold')
                
                if use_sam:
                    # SAM+YOLO สำหรับ Cold
                    yolo_results = self.yolo_object_model.predict(
                        source=deployment_frame_bgr, conf=Config.CONFIDENCE_THRESHOLD, show=False, save=False, stream=False, verbose=False
                    )
                    boxes = yolo_results[0].boxes.xyxy.cpu().numpy().astype(int)
                    if len(boxes) > 0:
                        x1, y1, x2, y2 = boxes[0]
                        image_rgb = cv2.cvtColor(deployment_frame_bgr, cv2.COLOR_BGR2RGB)
                        self.predictor.set_image(image_rgb)
                        masks, scores, _ = self.predictor.predict(box=np.array([x1, y1, x2, y2])[None, :], multimask_output=True)
                        best_mask = masks[np.argmax(scores)]
                        
                        # บันทึกรูปภาพ full deployment พร้อม mask (สีม่วง)
                        saved_path = _save_frame_with_mask(deployment_frame_bgr, best_mask, path_fd, overlay_color=(255, 255, 0))
                        if saved_path:
                            results['image_paths']['full_deployment'] = saved_path
                else:
                    # Segmentation สำหรับ Room/Hot
                    yolo_results = self.yolo_segmentation_model.predict(
                        source=deployment_frame_bgr, conf=Config.CONFIDENCE_THRESHOLD, show=False, save=False, stream=False, verbose=False
                    )
                    if yolo_results[0].masks is not None and len(yolo_results[0].masks) > 0:
                        mask_data = yolo_results[0].masks.data[0]
                        mask = mask_data.cpu().numpy().astype(np.uint8)
                        if mask.shape != deployment_frame_bgr.shape[:2]:
                            mask = cv2.resize(mask, (deployment_frame_bgr.shape[1], deployment_frame_bgr.shape[0]))
                        
                        # บันทึกรูปภาพ full deployment พร้อม mask (สีม่วง)
                        saved_path = _save_frame_with_mask(deployment_frame_bgr, mask, path_fd, overlay_color=(255, 255, 0))
                        if saved_path:
                            results['image_paths']['full_deployment'] = saved_path
                    
                # OCR
                o = self._ocr_ms_cop_from_frame(deployment_frame_bgr)
                if 'ms' in o: 
                    results['full_deployment_time_ms'] = o['ms']
                if 'cop' in o and not results['cop_number']: 
                    results['cop_number'] = o['cop']
                
                # results['image_paths']['full_deployment'] = _save_frame_with_mask(deployment_frame_bgr, path_fd)
                
                o = self._ocr_ms_cop_from_frame(deployment_frame_bgr)
                if 'ms' in o: results['full_deployment_time_ms'] = o['ms']
                if 'cop' in o and not results['cop_number']: results['cop_number'] = o['cop']
                    
            def _to_float_or_none(v):
                try:
                    if v is None:
                        return None
                    s = str(v).strip()
                    if not s:
                        return None
                    return float(s)
                except ValueError:
                    return None
                
            opening_candidates = [
                _to_float_or_none(results.get('explosion_time_ms')),
                _to_float_or_none(results.get('fr1_hit_time_ms')),
                _to_float_or_none(results.get('fr2_hit_time_ms')),
                _to_float_or_none(results.get('re3_hit_time_ms')),
                _to_float_or_none(results.get('re3_hit_time_ms')),
            ]
            
            results['opening_time_ms'] = next((x for x in opening_candidates if x is not None), 0.0)

            # Final progress report
            logging.info(f"[Stage: Complete] progress={deployment_progress_end}%")
            if confidence_scores:
                # Scale to percentage for DB layer (0–100)
                results['acc_rate_confidence'] = round((sum(confidence_scores) / len(confidence_scores)) * 100.0, 2)
            else:
                results['acc_rate_confidence'] = None
            return results
        
        except Exception as e:
            # ทำความสะอาดและปล่อย resources ในกรณีที่เกิด error
            if cap.isOpened():
                cap.release()
            
            # ถ้าเป็น TimeoutError ที่เกี่ยวกับการไม่เจอวงกลม ให้ลบวิดีโอ
            if isinstance(e, TimeoutError) and any(keyword in str(e).lower() 
                                                for keyword in ['no circles', 'front-view', 'unsuitable']):
                self._cleanup_unprocessable_video(video_path)
            
            raise e
    
    
    def _cleanup_unprocessable_video(self, video_path: str) -> None:
        """ลบวิดีโอที่ไม่สามารถประมวลผลได้ (เช่น front-view หา circles ไม่เจอ) จาก PROCESSING_DIR เท่านั้น"""
        try:
            video_name = os.path.basename(video_path)
            
            # ตรวจสอบว่าไฟล์อยู่ใน PROCESSING_DIR หรือไม่
            if Config.PROCESSING_DIR in video_path:
                # ลบเฉพาะไฟล์ใน PROCESSING_DIR
                if os.path.exists(video_path):
                    os.remove(video_path)
                    logging.info(f"🗑️ Removed unprocessable video from processing dir: {video_path}")
            else:
                # ถ้าไฟล์อยู่ใน INPUT_DIR ให้ log แต่ไม่ลบ
                logging.warning(f"⚠️ Unprocessable video detected but not removing from INPUT_DIR: {video_path}")
                    
            # อัปเดต global scan cache ถ้าเป็นไฟล์ loose (เฉพาะกรณี legacy mode)
            try:
                from main import _scanned_files
                # ลบจาก cache เฉพาะถ้าไฟล์ถูกลบจริงจาก PROCESSING_DIR
                if Config.PROCESSING_DIR in video_path:
                    _scanned_files.discard(video_name)
            except ImportError:
                pass  # ถ้าไม่ใช่ context ของ API
                    
        except Exception as cleanup_err:
            logging.warning(f"Failed to cleanup unprocessable video {video_path}: {cleanup_err}")

    def _analyze_full_deployment_with_deadline(self, video_path, temp_config, deadline: float | None):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, temp_config['start_frame'] - 1)

        mask_area_data = []
        frame_count = 0
        total_deployment_frames = temp_config['end_frame'] - temp_config['start_frame'] + 1
        temp_type = temp_config.get('temp_type', 'room')
        use_sam = (temp_type == 'cold')
        
        while True:
            _check_deadline(deadline, "Deployment")
            
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) + 1
            if current_frame > temp_config['end_frame']:
                break
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            
            # Report progress
            if total_deployment_frames > 0 and frame_count % 5 == 0:
                deployment_pct = 85 + int((frame_count / total_deployment_frames) * 10)
                logging.info(f"[Stage: Full Deployment] progress={deployment_pct}% (analyzing frame {current_frame})")

            # เลือกใช้ model ตาม temperature
            if use_sam:
                # SAM+YOLO สำหรับ Cold
                yolo_results = self.yolo_object_model.predict(
                    source=frame, conf=Config.CONFIDENCE_THRESHOLD, show=False, save=False, stream=False, verbose=False
                )
                boxes = yolo_results[0].boxes.xyxy.cpu().numpy().astype(int)
                if len(boxes) > 0:
                    x1, y1, x2, y2 = boxes[0]
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.predictor.set_image(image_rgb)
                    masks, scores, _ = self.predictor.predict(box=np.array([x1, y1, x2, y2])[None, :], multimask_output=True)
                    best_mask = masks[np.argmax(scores)]
                    area = int(np.sum(best_mask))
                    mask_area_data.append((current_frame, area))
            else:
                # Segmentation สำหรับ Room/Hot
                yolo_results = self.yolo_segmentation_model.predict(
                    source=frame, conf=Config.CONFIDENCE_THRESHOLD, show=False, save=False, stream=False, verbose=False
                )
                if yolo_results[0].masks is not None and len(yolo_results[0].masks) > 0:
                    mask_data = yolo_results[0].masks.data[0]
                    mask = mask_data.cpu().numpy().astype(np.uint8)
                    if mask.shape != frame.shape[:2]:
                        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                    area = int(np.sum(mask))
                    mask_area_data.append((current_frame, area))

        cap.release()
        
        # Final analysis progress
        logging.info(f"[Stage: Full Deployment] progress=95% (analysis complete, determining plateau)")

        if len(mask_area_data) == 0:
            logging.warning("No airbag mask data in configured window")
            return None, None

        areas = np.array([a for _, a in mask_area_data])
        frames = [f for f, _ in mask_area_data]
        smoothed = uniform_filter1d(areas, size=temp_config['smooth_size']) if len(areas) >= temp_config['smooth_size'] else areas

        peak_index = int(np.argmax(smoothed))
        plateau_frame = frames[peak_index]
        threshold = smoothed[peak_index] * temp_config['plateau_alpha'] if smoothed[peak_index] > 0 else 0

        for i in range(peak_index, len(smoothed)):
            if smoothed[i] < threshold:
                break
            plateau_frame = frames[i]

        logging.info(f"Full deployment at frame {plateau_frame} (smoothed peak={smoothed[peak_index]:.0f})")

        # Return the BGR frame for OCR
        cap_shot = cv2.VideoCapture(video_path)
        cap_shot.set(cv2.CAP_PROP_POS_FRAMES, plateau_frame - 1)
        ok, frame_shot = cap_shot.read()
        cap_shot.release()
        
        logging.info(f"[Stage: Full Deployment] progress=98% (OCR processing)")
        
        return plateau_frame, (frame_shot if ok else None)


# ======= LEGACY TEMPERATURE DETECTION (for backward compatibility) =======
def detect_temperature_from_filename(filename: str):
    """Detect temperature from filename.
    1) Try RT/CT/HT pattern (same as folder name format).
    2) Fallback to numeric markers: _23, _-30, _70.
    """
    try:
        base = os.path.splitext(os.path.basename(filename))[0]

        # 1) โครงเดียวกับชื่อโฟลเดอร์
        parsed = parse_folder_name(base)
        if parsed.get('valid'):
            temp_type = parsed['temp_type']
            return temp_type, Config.TEMP_CONFIGS[temp_type]

        # 2) Fallback: marker เป็นตัวเลขท้ายชื่อ
        m = re.search(r'_(-?\d+)(?=_[A-Za-z]+$|$)', base)
        if m:
            temp_val = int(m.group(1))
            if temp_val == 23:
                return 'room', Config.TEMP_CONFIGS['room']
            if temp_val == -30:
                return 'cold', Config.TEMP_CONFIGS['cold']
            if temp_val == 70:
                return 'hot', Config.TEMP_CONFIGS['hot']

        # default
        logging.warning(f"Cannot detect temperature from '{filename}', default room.")
        return 'room', Config.TEMP_CONFIGS['room']
    except Exception as e:
        logging.error(f"Error detect_temperature_from_filename('{filename}'): {e}")
        return 'room', Config.TEMP_CONFIGS['room']


# ======= FOLDER WATCHER (UPDATED) =======
class FolderHandler(FileSystemEventHandler):
    def __init__(self, processor, excel_manager, automation_system):
        self.processor = processor
        self.excel_manager = excel_manager
        self.processing = set()
        self.automation_system = automation_system

    def on_created(self, event):
        if not event.is_directory:
            return
            
        folder_path = event.src_path
        folder_name = os.path.basename(folder_path)
        
        if folder_name in self.automation_system.scanned_folders:
            return
            
        self.automation_system.scanned_folders.add(folder_name)
        
        parsed = parse_folder_name(folder_name)
        if parsed['valid']:
            logging.info(f"New folder detected: {folder_name}")
            self._process_folder_after_delay(folder_path)

    def _process_folder_after_delay(self, folder_path):
        time.sleep(3)  # Wait a bit longer for folder creation to complete
        
        if folder_path in self.processing:
            return
            
        self.processing.add(folder_path)
        
        try:
            # Wait for folder to be stable (no new files being added)
            prev_file_count = -1
            stable_count = 0
            
            while stable_count < 3:
                if not os.path.exists(folder_path):
                    logging.warning(f"Folder disappeared: {folder_path}")
                    return
                    
                try:
                    current_file_count = len([f for f in os.listdir(folder_path) 
                                            if f.lower().endswith(('.avi', '.mp4', '.mov', '.mkv'))])
                except OSError:
                    current_file_count = 0
                    
                if current_file_count == prev_file_count and current_file_count > 0:
                    stable_count += 1
                else:
                    stable_count = 0
                prev_file_count = current_file_count
                time.sleep(2)

            logging.info(f"📊 Processing stable folder: {folder_path}")
            results = self.processor.process_folder(folder_path)
            
            if results and 'error' not in results:
                saved_path = self.excel_manager.add_result(results)
                
                # Move folder to bin directory after successful processing
                self._move_folder_to_bin(folder_path)
                
                logging.info(f"✅ Completed processing: {folder_path} -> {saved_path}")
            else:
                error_msg = results.get('error', 'Unknown error') if results else 'No results returned'
                logging.error(f"❌ Processing failed for {folder_path}: {error_msg}")
                
        except Exception as e:
            logging.error(f"❌ Error processing folder {folder_path}: {e}")
        finally:
            self.processing.discard(folder_path)

    def _move_folder_to_bin(self, folder_path):
        """Move processed folder to bin directory."""
        try:
            # Ensure bin directory exists
            os.makedirs(Config.BIN_DIR, exist_ok=True)
            
            folder_name = os.path.basename(folder_path)
            destination = os.path.join(Config.BIN_DIR, folder_name)
            
            # Handle duplicate folder names by adding timestamp
            if os.path.exists(destination):
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                folder_name_with_ts = f"{folder_name}_{timestamp}"
                destination = os.path.join(Config.BIN_DIR, folder_name_with_ts)
            
            shutil.move(folder_path, destination)
            logging.info(f"📦 Moved processed folder to bin: {destination}")
            
        except Exception as e:
            logging.error(f"❌ Failed to move folder to bin: {e}")

# ======= MAIN AUTOMATION SYSTEM (UPDATED) =======
class AirbagAutomationSystem:
    def __init__(self):
        setup_logging()
        self.processor = AirbagProcessor()
        # self.excel_manager = ExcelManager(Config.EXCEL_OUTPUT_PATH)
        self.observer = Observer()
        self.scanned_folders = set()
        
        # Create directories
        os.makedirs(Config.INPUT_DIR, exist_ok=True)
        os.makedirs(Config.PROCESSING_DIR, exist_ok=True)
        os.makedirs(Config.BIN_DIR, exist_ok=True)
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        
        self._cleanup_legacy_files()
        
    def _cleanup_legacy_files(self) -> None:
        if not getattr(Config, 'CLEANUP_LEGACY_ON_START', False):
            return
        try:
            # ลบไฟล์รวมเดิม
            if os.path.exists(Config.EXCEL_OUTPUT_PATH):
                os.remove(Config.EXCEL_OUTPUT_PATH)
                logging.info(f"🧹 Removed legacy file: {Config.EXCEL_OUTPUT_PATH}")
            # ลบ snapshot เดิมทั้งหมด
            for f in glob.glob(os.path.join(Config.OUTPUT_DIR, 'airbag_results_snapshot_*.xlsx')):
                try:
                    os.remove(f)
                    logging.info(f"🧹 Removed snapshot: {f}")
                except Exception as e:
                    logging.warning(f"⚠️ Failed to remove snapshot {f}: {e}")
        except Exception as e:
            logging.warning(f"⚠️ Cleanup failed: {e}")

    def process_existing_folders(self):
        """Process existing folders in the input directory."""
        try:
            existing_folders = [f for f in os.listdir(Config.INPUT_DIR) 
                              if os.path.isdir(os.path.join(Config.INPUT_DIR, f))]
            
            valid_folders = []
            for folder_name in existing_folders:
                # เพิ่มการเช็คว่าเคย scan แล้วหรือยัง
                if folder_name in self.scanned_folders:
                    continue
                    
                self.scanned_folders.add(folder_name)  # เพิ่มเข้าไปใน set
                
                parsed = parse_folder_name(folder_name)
                if parsed['valid']:
                    valid_folders.append(folder_name)
                # Invalid folders จะไม่แสดง warning ซ้ำๆ เพราะถูก skip แล้ว
                    
            if valid_folders:
                logging.info(f"Found {len(valid_folders)} existing folders to process")
                for folder_name in valid_folders:
                    folder_path = os.path.join(Config.INPUT_DIR, folder_name)
                    try:
                        results = self.processor.process_folder(folder_path)
                        if results and 'error' not in results:
                            saved_path = self.excel_manager.add_result(results)
                            self._move_folder_to_bin(folder_path)
                            logging.info(f"Added result for {folder_name} -> {saved_path}")
                        else:
                            error_msg = results.get('error', 'Unknown error') if results else 'No results returned'
                            logging.error(f"Failed to process folder {folder_name}: {error_msg}")
                    except Exception as e:
                        logging.error(f"Failed to process existing folder {folder_name}: {e}")
            else:
                logging.info("No new valid folders found for processing")
                
        except Exception as e:
            logging.error(f"Error scanning existing folders: {e}")

    def _move_folder_to_bin(self, folder_path):
        """Move processed folder to bin directory."""
        try:
            folder_name = os.path.basename(folder_path)
            destination = os.path.join(Config.BIN_DIR, folder_name)
            
            # Handle duplicate folder names by adding timestamp
            if os.path.exists(destination):
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                folder_name_with_ts = f"{folder_name}_{timestamp}"
                destination = os.path.join(Config.BIN_DIR, folder_name_with_ts)
            
            shutil.move(folder_path, destination)
            logging.info(f"📦 Moved processed folder to bin: {destination}")
            
        except Exception as e:
            logging.error(f"❌ Failed to move folder to bin: {e}")

    def start_monitoring(self):
        handler = FolderHandler(self.processor, self.excel_manager, self)
        if not os.path.isdir(Config.INPUT_DIR):
            logging.error("INPUT_DIR not accessible: %s", Config.INPUT_DIR)
        else:
            try:
                entries = [e for e in os.listdir(Config.INPUT_DIR)[:5]]
                logging.info("INPUT_DIR OK. First entries: %s", entries)
            except Exception as e:
                logging.error("Cannot list INPUT_DIR (%s): %s", Config.INPUT_DIR, e)
        self.observer.schedule(handler, Config.INPUT_DIR, recursive=False)
        self.observer.start()
        logging.info(f"👀 Monitoring directory for folders: {Config.INPUT_DIR}")
        logging.info(f"📊 Results will be saved per video in: {Config.OUTPUT_DIR}")
        logging.info(f"📦 Processed folders will be moved to: {Config.BIN_DIR}")
        
        # Reset warning counters ทุกๆ 1 ชั่วโมง
        last_reset_time = time.time()
        
        try:
            while True:
                time.sleep(1)
                
                # Reset warning counters ทุกๆ 3600 วินาที (1 ชั่วโมง)
                current_time = time.time()
                if current_time - last_reset_time > 3600:
                    global WARNING_COUNTERS
                    WARNING_COUNTERS = {key: 0 for key in WARNING_COUNTERS}
                    logging.info("🔄 Warning counters reset")
                    last_reset_time = current_time
                    
        except KeyboardInterrupt:
            logging.info("🛑 Stopping automation system...")
            self.observer.stop()
        self.observer.join()

    def run(self):
        logging.info("🚀 Starting Airbag Detection Automation System (Folder Mode)")
        logging.info(f"📁 Expected folder format: 'P703 DBL CAB [RT/CT/HT] [MODULE-SN]'")
        self.process_existing_folders()
        self.start_monitoring()



# ======= ENTRY POINT =======
if __name__ == "__main__":
    try:
        system = AirbagAutomationSystem()
        system.run()
    except KeyboardInterrupt:
        logging.info("👋 System stopped by user")
    except Exception as e:
        logging.error(f"💥 System crashed: {e}")
        sys.exit(1)
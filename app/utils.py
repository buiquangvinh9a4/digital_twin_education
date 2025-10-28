import os
import json
from datetime import datetime

STATUS_FILE = "data/simulations/status.json"
METRICS_LOG = "data/simulations/metrics_log.csv"

def read_status():
    try:
        if not os.path.exists(STATUS_FILE):
            return None
        with open(STATUS_FILE, "r") as f:
            data = json.load(f)
        return data
    except Exception:
        return None

def format_mtime(ts):
    try:
        return datetime.fromtimestamp(float(ts)).strftime("%H:%M:%S")
    except Exception:
        return "--:--:--"

def file_mtime(path):
    try:
        return os.path.getmtime(path)
    except Exception:
        return None



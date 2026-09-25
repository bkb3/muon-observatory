import cv2
import numpy as np
import time
import os
import csv
import subprocess
import math
from threading import Thread
from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
import uvicorn
import onnxruntime as ort
import datetime

cv2.setNumThreads(0)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["ORT_DISABLE_TELEMETRY"] = "1"

BASE_DIR = "/mnt/usb_data/radio_astronomy/muon"
MODEL_PATH = os.path.join(BASE_DIR, "muon_model_bce.onnx")

ONNX_SESSION = ort.InferenceSession(MODEL_PATH)
INPUT_NAME = ONNX_SESSION.get_inputs()[0].name


def find_working_camera():
    for index in range(5):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                print(f"Successfully found active camera at index {index}")
                return index
    raise RuntimeError("No active video source found!")


def get_camera_hardware_name(index):
    name_path = f"/sys/class/video4linux/video{index}/name"
    raw_name = ""
    if os.path.exists(name_path):
        try:
            with open(name_path, "r") as f:
                raw_name = f.read().strip()
        except Exception:
            pass
            
    if not raw_name:
        try:
            res = subprocess.run(["v4l2-ctl", f"-d/dev/video{index}", "--all"], capture_output=True, text=True)
            for line in res.stdout.splitlines():
                if "Card type" in line:
                    raw_name = line.split(":")[-1].strip()
                    break
        except Exception:
            pass
            
    if not raw_name:
        return f"Video Device {index}"

    parts = [p.strip() for p in raw_name.split(":") if p.strip()]
    unique_parts = []
    for p in parts:
        if not unique_parts or p.lower() != unique_parts[-1].lower():
            unique_parts.append(p)
            
    clean_name = " - ".join(unique_parts) if unique_parts else f"Video Device {index}"
    return clean_name


VIDEO_SOURCE = find_working_camera()
THRESHOLD_VALUE = 140
MIN_PIXELS = 4
MAX_PIXELS = 100
CALIBRATION_FRAMES = 100
PORT = 9003

TEMPLATE_FILE = os.path.join(BASE_DIR, "templates", "index.html")

stats = {
    "total_muons": 0,
    "today_muons": 0,
    "last_event": "None",
    "status": "Calibrating sensor...",
    "flux_rate": "0.0/hr",
    "mean_saturation": "0.0/255",
    "all_time_max_size": "0px",
    "all_time_max_bright": "0",
    "class_artefacts": "0%",
    "class_dots": "0%",
    "class_tracks": "0%",
    "class_worms": "0%"
}

historical_events_cache = []
hit_coordinate_history = {}
hot_pixel_mask = None

def get_date_paths(custom_date_str=None):
    if custom_date_str:
        t_struct = time.strptime(custom_date_str, "%Y-%m-%d")
        slash_path = time.strftime("data/%Y/%m/%d", t_struct)
        return slash_path, custom_date_str
    else:
        local_time = time.localtime()
        slash_path = time.strftime("data/%Y/%m/%d", local_time)
        hyphen_str = time.strftime("%Y-%m-%d", local_time)
        return slash_path, hyphen_str


def get_or_create_daily_csv(slash_path):
    day_dir = os.path.join(BASE_DIR, slash_path)
    os.makedirs(day_dir, exist_ok=True)
    csv_path = os.path.join(day_dir, "muon_log.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Pixel_Count", "Max_Brightness", "Image_Filename", "Width", "Height", "Particle_Type", "Probability"])
    return csv_path

HISTORICAL_CSV_PATH = os.path.join(BASE_DIR, "static", "historical_data.csv")
CSV_HEADER = ["Timestamp", "Pixel_Count", "Max_Brightness", "Image_Filename", "Width", "Height", "Particle_Type", "Probability"]

def append_to_historical_csv(row):
    """Appends a row to static/historical_data.csv ONLY if it is a muon track."""
    if len(row) <= 6:
        return
    
    p_type = row[6].lower().strip()
    if p_type in ['tracks', 'track', 'muon']:
        os.makedirs(os.path.dirname(HISTORICAL_CSV_PATH), exist_ok=True)
        file_exists = os.path.exists(HISTORICAL_CSV_PATH)
        with open(HISTORICAL_CSV_PATH, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(CSV_HEADER)
            writer.writerow(row)

def rebuild_historical_csv(all_rows):
    """Overwrites static/historical_data.csv filtering strictly for muon tracks."""
    muon_rows = [
        r for r in all_rows 
        if len(r) > 6 and r[6].lower().strip() in ['tracks', 'track', 'muon']
    ]
    os.makedirs(os.path.dirname(HISTORICAL_CSV_PATH), exist_ok=True)
    with open(HISTORICAL_CSV_PATH, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        writer.writerows(muon_rows)

def preload_historical_cache():
    global historical_events_cache
    cached_rows = []
    for root, _, files in os.walk(BASE_DIR):
        if "muon_log.csv" in files:
            try:
                with open(os.path.join(root, "muon_log.csv"), "r") as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
                    for row in reader:
                        if len(row) >= 4:
                            if len(row) < 7:
                                img_filename = row[3]
                                inferred_type = "tracks" if img_filename.startswith("muon") else "dots"
                                row.append(inferred_type)
                            
                            current_type = row[6].lower()
                            if current_type == "muon":
                                row[6] = "tracks"
                            elif current_type == "beta":
                                row[6] = "dots"
                            elif not current_type.endswith('s') and current_type in ['artefact', 'dot', 'track', 'worm']:
                                row[6] = current_type + 's'

                            if len(row) < 8:
                                row.append("100.0%")
                                
                            cached_rows.append(row)
            except Exception:
                pass
    historical_events_cache = cached_rows
    compute_advanced_statistics()
    # Export full history to static/historical_data.csv on boot
    rebuild_historical_csv(historical_events_cache)

def compute_advanced_statistics():
    global stats, historical_events_cache
    
    total_events = len(historical_events_cache)
    stats["total_muons"] = total_events
    if total_events == 0:
        return

    max_size, max_bright, total_brightness, today_count = 0, 0, 0, 0
    c_counts = {"artefacts": 0, "dots": 0, "tracks": 0, "worms": 0}
    
    _, current_hyphen_date = get_date_paths()  
    timestamps = []

    for r in historical_events_cache:
        try:
            ts = time.mktime(time.strptime(r[0].split('.')[0], "%Y-%m-%d %H:%M:%S"))
            timestamps.append(ts)
            
            if r[0].startswith(current_hyphen_date):
                today_count += 1
                
            size, bright = int(r[1]), int(r[2])
            total_brightness += bright
            
            if size > max_size: max_size = size
            if bright > max_bright: max_bright = bright
            
            p_type = r[6].lower() if len(r) > 6 else "dots"
            if not p_type.endswith('s'): p_type += 's'
            if p_type in c_counts:
                c_counts[p_type] += 1
        except Exception:
            pass

    stats["today_muons"] = today_count  
    stats["all_time_max_size"] = f"{max_size}px"
    stats["all_time_max_bright"] = str(max_bright)
    stats["mean_saturation"] = f"{round(total_brightness / total_events, 1)}/255"
    
    for k in c_counts:
        stats[f"class_{k}"] = f"{round((c_counts[k] / total_events) * 100, 1)}%"

    if len(timestamps) > 0:
        timestamps.sort()
        total_runtime_hours = max(1.0, (timestamps[-1] - timestamps[0]) / 3600.0)
        stats["flux_rate"] = f"{round(total_events / total_runtime_hours, 2)}/hr"


def disable_camera_auto_features():
    try:
        subprocess.run(["v4l2-ctl", f"-d/dev/video{VIDEO_SOURCE}", "-c", "exposure_auto=1"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["v4l2-ctl", f"-d/dev/video{VIDEO_SOURCE}", "-c", "gain_automatic=0"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


CLASSES = ['artefacts', 'dots', 'tracks', 'worms']
OPTIMAL_THRESHOLDS = {
    'artefacts': 0.49,
    'dots': 0.64,
    'tracks': 0.52,
    'worms': 0.20
}

def classify_particle_track(event_crop_gray):
    if event_crop_gray.shape != (60, 60):
        event_crop_gray = cv2.resize(event_crop_gray, (60, 60), interpolation=cv2.INTER_AREA)

    input_tensor = event_crop_gray.astype(np.float32) / 255.0
    input_tensor = np.expand_dims(input_tensor, axis=(0, 1))

    raw_outputs = ONNX_SESSION.run(None, {INPUT_NAME: input_tensor})
    logits = raw_outputs[0][0]

    # Convert raw logits to independent Sigmoid probabilities
    probs = 1 / (1 + np.exp(-logits))

    passed_classes = []
    for idx, name in enumerate(CLASSES):
        prob = float(probs[idx])
        # Check if the class probability passes its custom MCC threshold
        if prob >= OPTIMAL_THRESHOLDS[name]:
            passed_classes.append((idx, name, prob))

    if not passed_classes:
        # Fallback to artefacts if no class crosses its threshold
        # Using the base artefacts probability for reporting
        return "artefacts", float(probs[0])
    
    # Sort classes based on the margin of victory over their respective thresholds
    # Formula: margin = probability - threshold
    passed_classes.sort(key=lambda x: x[2] - OPTIMAL_THRESHOLDS[x[1]], reverse=True)
    
    winning_idx, winning_class, winning_prob = passed_classes[0]
    return winning_class, winning_prob



def particle_detector_loop():
    global stats, hot_pixel_mask, hit_coordinate_history, historical_events_cache
    disable_camera_auto_features()
    
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        stats["status"] = f"Error: Cannot open camera index {VIDEO_SOURCE}"
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15) 

    hot_pixel_accumulator = None
    frames_collected = 0
    while frames_collected < CALIBRATION_FRAMES:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if hot_pixel_accumulator is None:
            hot_pixel_accumulator = np.zeros_like(gray, dtype=np.uint32)
        _, thresh = cv2.threshold(gray, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
        hot_pixel_accumulator += thresh
        frames_collected += 1
        
    hot_pixel_mask = (hot_pixel_accumulator > (CALIBRATION_FRAMES * 0.1 * 255))
    cam_name = get_camera_hardware_name(VIDEO_SOURCE)
    stats["status"] = f"Active: {cam_name} (Index {VIDEO_SOURCE})"
    
    while True:
        loop_start = time.time()
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.5)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray[hot_pixel_mask] = 0
        
        if not np.any(gray > THRESHOLD_VALUE):
            time_spent = time.time() - loop_start
            delay = max(0.001, 0.066 - time_spent)
            time.sleep(delay)
            continue

        _, thresh = cv2.threshold(gray, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)

        num_labels, labels, stats_connected, centroids = cv2.connectedComponentsWithStats(thresh)
        img_h, img_w = gray.shape

        if num_labels <= 1:
            time_spent = time.time() - loop_start
            delay = max(0.001, 0.066 - time_spent)
            time.sleep(delay)
            continue

        # Find the largest connected component (ignoring background index 0)
        valid_labels = []
        for i in range(1, num_labels):
            pixel_area = stats_connected[i, cv2.CC_STAT_AREA]
            if MIN_PIXELS <= pixel_area <= MAX_PIXELS:
                valid_labels.append(i)

        if not valid_labels:
            time_spent = time.time() - loop_start
            delay = max(0.001, 0.066 - time_spent)
            time.sleep(delay)
            continue

        # Pick only the largest component to avoid multiple false triggers per frame
        # this is fine as muons are generally long tracks and two long tracks at once is 
        # unlikely
        i = max(valid_labels, key=lambda idx: stats_connected[idx, cv2.CC_STAT_AREA])
        
        pixel_area = stats_connected[i, cv2.CC_STAT_AREA]
        cx, cy = int(centroids[i][0]), int(centroids[i][1])
        coord_key = (cx, cy)
        hit_coordinate_history[coord_key] = hit_coordinate_history.get(coord_key, 0) + 1
        
        if hit_coordinate_history[coord_key] > 2:
            hot_pixel_mask[labels == i] = True  
            continue  

        slash_path, _ = get_date_paths()
        t_now = time.time()
        microseconds = int((t_now - int(t_now)) * 1000000)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_now)) + f".{microseconds:06d}"

        epoch_time = int(time.time())
        
        mask = (labels == i)
        max_val = int(np.max(gray[mask]))
        
        w = stats_connected[i, cv2.CC_STAT_WIDTH]
        h = stats_connected[i, cv2.CC_STAT_HEIGHT]

        track_max_dim = max(w, h)
        crop_half_size = max(30, (track_max_dim // 2) + 10)

        y1, y2 = cy - crop_half_size, cy + crop_half_size
        x1, x2 = cx - crop_half_size, cx + crop_half_size

        pad_top, pad_bottom = max(0, -y1), max(0, y2 - img_h)
        pad_left, pad_right = max(0, -x1), max(0, x2 - img_w)

        crop = gray[max(0, y1):min(img_h, y2), max(0, x1):min(img_w, x2)]
        crop = cv2.copyMakeBorder(crop, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)

        if crop.shape != (60, 60):
            crop = cv2.resize(crop, (60, 60), interpolation=cv2.INTER_AREA)

        particle_type, confidence = classify_particle_track(crop)
        
        crop_colored = cv2.applyColorMap(crop, cv2.COLORMAP_PLASMA)
        
        day_dir = os.path.join(BASE_DIR, slash_path)
        os.makedirs(day_dir, exist_ok=True)
        
        img_name = f"{particle_type}_{epoch_time}.png"
        img_path = os.path.join(day_dir, img_name)
        
        cv2.imwrite(img_path, crop_colored)

        csv_file = os.path.join(day_dir, "muon_log.csv")

        confidence_str = f"{round(confidence * 100, 1)}%"
        new_row = [timestamp, pixel_area, max_val, img_name, w, h, particle_type, confidence_str]
        csv_file = get_or_create_daily_csv(slash_path)
        
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(new_row)
        
        append_to_historical_csv(new_row)

        historical_events_cache.append(new_row)
        compute_advanced_statistics()
        stats["last_event"] = f"{timestamp} ({particle_type}, {pixel_area}px, peak: {max_val})"

        time_spent = time.time() - loop_start
        delay = max(0.001, 0.066 - time_spent)
        time.sleep(delay)


app = FastAPI()
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")


@app.get("/api/poisson")
def get_poisson_chart_data():
    """Computes Poisson distribution for ALL historical track events."""
    global historical_events_cache
    if not historical_events_cache:
        return {"observed_bins": [], "poisson_fit": [], "labels": ["0 hits"], "lambda": 0}

    hourly_bins = {}
    
    for r in historical_events_cache:
        try:
            p_type = r[6].lower().strip() if len(r) > 6 else "dots"
            if p_type != 'tracks':
                continue 
                
            t_str = r[0]
            hour_bucket = t_str[:13]
            hourly_bins[hour_bucket] = hourly_bins.get(hour_bucket, 0) + 1
        except Exception:
            pass

    counts = list(hourly_bins.values())
    if not counts:
        return {"observed_bins": [], "poisson_fit": [], "labels": ["0 hits"], "lambda": 0}
        
    lambda_val = np.mean(counts)
    max_hits = max(max(counts), 5)
    
    observed_distribution = [0] * (max_hits + 1)
    for c in counts:
        observed_distribution[c] += 1
        
    total_hours = len(counts)
    observed_pct = [round((x / total_hours) * 100, 1) for x in observed_distribution]
    
    theoretical_pct = []
    for k in range(max_hits + 1):
        p_k = (math.pow(lambda_val, k) * math.exp(-lambda_val)) / math.factorial(k)
        theoretical_pct.append(round(p_k * 100, 1))
        
    labels = [f"{k}" for k in range(max_hits + 1)]
    return {
        "observed_bins": observed_pct,
        "poisson_fit": theoretical_pct,
        "labels": labels,
        "lambda": round(lambda_val, 3)
    }


@app.get("/api/hourly")
def get_hourly_distribution(date: str = Query(None)):
    """
    Returns 24-hour distribution. If 'date' query parameter is omitted or empty,
    it returns today's historical track events.
    """
    global historical_events_cache
    hours_count = [0] * 24

    if not date:
        date = datetime.datetime.utcnow().strftime("%Y-%m-%d")
    
    for r in historical_events_cache:
        try:
            if date and not r[0].startswith(date):
                continue
                
            p_type = r[6].lower().strip() if len(r) > 6 else "dots"
            if p_type != 'tracks':
                continue 
                
            t_struct = time.strptime(r[0].split('.')[0], "%Y-%m-%d %H:%M:%S")
            hours_count[t_struct.tm_hour] += 1
        except Exception:
            pass
            
    return {"labels": [f"{h:02d}:00" for h in range(24)], "counts": hours_count}


@app.get("/api/intervals")
def get_waiting_intervals():
    """Computes inter-arrival waiting time distribution across ALL historical events."""
    global historical_events_cache
    timestamps = []
    
    for r in historical_events_cache:
        try:
            p_type = r[6].lower().strip() if len(r) > 6 else "dots"
            if p_type != 'tracks':
                continue
            timestamps.append(time.mktime(time.strptime(r[0].split('.')[0], "%Y-%m-%d %H:%M:%S")))
        except Exception:
            pass
            
    labels = ["0-10m", "10-20m", "20-30m", "30-40m", "40-50m", "50-60m"]
    bins = [0] * 6
    
    if len(timestamps) >= 2:
        timestamps.sort()
        for i in range(1, len(timestamps)):
            d_min = (timestamps[i] - timestamps[i-1]) / 60.0
            if d_min <= 10: bins[0] += 1
            elif d_min <= 20: bins[1] += 1
            elif d_min <= 30: bins[2] += 1
            elif d_min <= 40: bins[3] += 1
            elif d_min <= 50: bins[4] += 1
            elif d_min <= 60: bins[5] += 1

    total_intervals = sum(bins)
    observed_pcts = [0.0] * 6
    expected_pcts = []
    
    if total_intervals > 0 and len(timestamps) >= 2:
        observed_pcts = [round((b / total_intervals) * 100, 1) for b in bins]
        
        total_time_span = (timestamps[-1] - timestamps[0]) / 60.0
        if total_time_span > 0:
            lam = (len(timestamps) - 1) / total_time_span
            bin_width = 10.0
            
            for i in range(6):
                t1 = i * bin_width
                t2 = (i + 1) * bin_width
                prob = math.exp(-lam * t1) - math.exp(-lam * t2)
                expected_pcts.append(round(prob * 100, 1))
    
    if not expected_pcts:
        expected_pcts = [0.0] * 6

    return {
        "labels": labels, 
        "bins": observed_pcts, 
        "expected_fit": expected_pcts
    }


@app.get("/api/random")
def get_true_random_number():
    global historical_events_cache
    
    # Filter strictly for track/muon events
    track_events = [
        r for r in historical_events_cache 
        if len(r) > 6 and r[6].lower().strip() in ['tracks', 'track', 'muon']
    ]

    if not track_events:
        return {
            "status": "waiting",
            "message": "No valid muon track events available for RNG",
            "random_1_100": "---",
            "random_byte": "---",
            "entropy_pool_bits": 0
        }

    latest_event = track_events[-1]
    timestamp_str = latest_event[0]
    event_type = latest_event[6] if len(latest_event) > 6 else "tracks"
    
    try:
        parts = timestamp_str.split('.')
        if len(parts) == 2:
            latest_micros = int(parts[1])
        else:
            latest_micros = int(hash(timestamp_str) % 1000000)
    except Exception:
        latest_micros = int(time.time() * 1000000) % 1000000

    rand_100 = (latest_micros % 100) + 1
    rand_byte = latest_micros % 256
    
    # 8 bits per event (1 byte extracted via % 256)
    BITS_PER_EVENT = 8
    entropy_pool_bits = len(track_events) * BITS_PER_EVENT
    
    return {
        "status": "ready",
        "source_event_type": event_type,
        "random_1_100": rand_100,
        "random_byte": f"0x{rand_byte:02x}",
        "entropy_pool_bits": entropy_pool_bits
    }


@app.get("/", response_class=FileResponse)
def serve_unified_dashboard():
    return FileResponse("index.html")


@app.get("/api/stats_endpoint_mock_or_real")
def get_live_dashboard_stats():
    global stats
    return JSONResponse(content=stats)


@app.get("/api/table_rows_mock_or_real")
def get_live_dashboard_table_rows(date: str = Query(None), filter: str = Query(None)):
    _, current_hyphen = get_date_paths()
    selected_hyphen = date if date else current_hyphen
    selected_slash, _ = get_date_paths(selected_hyphen)
    
    target_day_dir = os.path.join(BASE_DIR, selected_slash)
    cards_html = ""
    rows_html = ""
    rows = []
    
    if os.path.exists(target_day_dir):
        csv_file = os.path.join(target_day_dir, "muon_log.csv")
        if os.path.exists(csv_file):
            try:
                with open(csv_file, mode='r') as f:
                    reader = csv.reader(f)
                    next(reader, None)
                    rows = list(reader)
                    
                    rows.sort(key=lambda x: x[0] if len(x) > 0 else "", reverse=True)
                    for r in rows:
                        if len(r) >= 4:
                            p_type = r[6].lower().strip() if len(r) > 6 else "dots"
                            if filter == "tracks" and p_type not in ["tracks", "track"]:
                                continue
                                
                            cat_name = r[6].capitalize() if len(r) > 6 else "Dots"
                            try:
                                raw_prob = float(r[7].replace('%', ''))
                            except (IndexError, ValueError):
                                raw_prob = 100.0
                            
                            prob_val = f"{raw_prob:.2f}%"
                            display_ts = r[0].split('.')[0]
                            
                            rows_html += f"<tr><td>{display_ts}</td><td>{r[1]} px</td><td>{r[2]}</td><td>{cat_name}</td><td><strong>{prob_val}</strong></td></tr>"
                            
                            img_filename = r[3]
                            img_src = f"data/images/{selected_hyphen}/{img_filename}"
                            cards_html += f"""
                            <div class="card" onclick="openZoom('{img_src}', '{img_filename}', '{prob_val}')">
                                <img src="{img_src}" alt="muon track">
                                <span>{img_filename.replace('muon_', '').replace('.png', '')}</span>
                            </div>
                            """
            except Exception:
                pass

    if not cards_html:
        cards_html = "<p style='grid-column: 1/-1; color: #888;'>No track image frames captured for this date.</p>"
    if not rows_html:
        rows_html = "<tr><td colspan='5' style='color:#888; text-align:center;'>No track log entries recorded today.</td></tr>"

    t_counts = {"artefacts": 0, "dots": 0, "tracks": 0, "worms": 0}
    for r in rows:
        if len(r) >= 7:
            p_type = r[6].lower().strip()
            if not p_type.endswith('s'): p_type += 's'
            if p_type in t_counts:
                t_counts[p_type] += 1

    breakdown_parts = []
    if t_counts["tracks"] > 0: breakdown_parts.append(f"{t_counts['tracks']} Track{'s' if t_counts['tracks']!=1 else ''}")
    if t_counts["dots"] > 0: breakdown_parts.append(f"{t_counts['dots']} Dot{'s' if t_counts['dots']!=1 else ''}")
    if t_counts["worms"] > 0: breakdown_parts.append(f"{t_counts['worms']} Worm{'s' if t_counts['worms']!=1 else ''}")
    if t_counts["artefacts"] > 0: breakdown_parts.append(f"{t_counts['artefacts']} Noise")
    
    breakdown_str = f"{len(rows)}"
    if breakdown_parts:
        breakdown_str += f" ({', '.join(breakdown_parts)})"

    return JSONResponse(content={
        "rows_html": rows_html, 
        "cards_html": cards_html,
        "today_breakdown": breakdown_str
    })


@app.get("/data/images/{date_hyphen}/{filename}")
def serve_image(date_hyphen: str, filename: str):
    slash_path, _ = get_date_paths(date_hyphen)
    file_path = os.path.join(BASE_DIR, slash_path, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return HTMLResponse("Target track snapshot file entry missing.", status_code=404)


@app.get("/log")
def get_log(date: str = Query(None)):
    _, current_hyphen = get_date_paths()
    selected_hyphen = date if date else current_hyphen
    selected_slash, _ = get_date_paths(selected_hyphen)
    
    csv_file = os.path.join(BASE_DIR, selected_slash, "muon_log.csv")
    if os.path.exists(csv_file):
        return FileResponse(csv_file, media_type="text/csv", filename=f"muon_log_{selected_hyphen}.csv")
    return HTMLResponse("No log files matched selected calendar grid parameters.", status_code=404)


if __name__ == "__main__":
    boot_slash, _ = get_date_paths()
    get_or_create_daily_csv(boot_slash)
    preload_historical_cache()

    detector_thread = Thread(target=particle_detector_loop, daemon=True)
    detector_thread.start()

    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning", workers=1)
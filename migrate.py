# sudo -E env PATH="$PATH" python migrate.py

import os
import csv
import cv2
import numpy as np
import onnxruntime as ort
import time

BASE_DIR = "/mnt/usb_data/radio_astronomy/muon"
HISTORICAL_CSV_PATH = os.path.join(BASE_DIR, "static", "historical_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "muon_model_bce.onnx")

ONNX_SESSION = ort.InferenceSession(MODEL_PATH)
INPUT_NAME = ONNX_SESSION.get_inputs()[0].name

CLASSES = ['artefacts', 'dots', 'tracks', 'worms']
OPTIMAL_THRESHOLDS = {
    'artefacts': 0.49,
    'dots': 0.64,
    'tracks': 0.52,
    'worms': 0.20
}

def classify_crop(crop_gray):
    if crop_gray.shape != (60, 60):
        crop_gray = cv2.resize(crop_gray, (60, 60), interpolation=cv2.INTER_AREA)

    input_tensor = crop_gray.astype(np.float32) / 255.0
    input_tensor = np.expand_dims(input_tensor, axis=(0, 1))

    raw_outputs = ONNX_SESSION.run(None, {INPUT_NAME: input_tensor})
    logits = raw_outputs[0][0]

    # Convert to independent sigmoid probabilities
    probs = 1 / (1 + np.exp(-logits))

    passed_classes = []
    for idx, name in enumerate(CLASSES):
        prob = float(probs[idx])
        if prob >= OPTIMAL_THRESHOLDS[name]:
            passed_classes.append((idx, name, prob))

    if not passed_classes:
        return "artefacts", float(probs[0])
    
    # Sort by margin over threshold
    passed_classes.sort(key=lambda x: x[2] - OPTIMAL_THRESHOLDS[x[1]], reverse=True)
    return passed_classes[0][1], passed_classes[0][2]


def process_directory(target_dir):
    print(f"Scanning directory hierarchy under: {target_dir}")

    for root, dirs, files in os.walk(target_dir):
        valid_files = []
        csv_path = os.path.join(root, "muon_log.csv")

        for f in files:
            if f.endswith(".csv"):
                continue
            if any(f.startswith(c) for c in CLASSES):
                valid_files.append(f)

        if not valid_files and not os.path.exists(csv_path):
            continue

        print(f"Processing day folder: {root}")

        existing_rows_map = {}
        if os.path.exists(csv_path):
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    if len(row) >= 4:
                        fn = row[3].replace(".png", "")
                        parts = fn.split("_")
                        epoch = parts[-1] if len(parts) > 1 else ""
                        if epoch:
                            existing_rows_map[epoch] = row

        processed_rows = []

        for img_file in list(valid_files):
            img_path = os.path.join(root, img_file)
            if not os.path.exists(img_path):
                continue

            colored_crop = cv2.imread(img_path)
            if colored_crop is None:
                continue
            gray_crop = cv2.cvtColor(colored_crop, cv2.COLOR_BGR2GRAY)

            new_type, conf = classify_crop(gray_crop)

            base_name = img_file.replace(".png", "")
            parts = base_name.split("_")
            epoch_part = parts[-1] if len(parts) > 1 else str(int(time.time()))
            
            new_img_name = f"{new_type}_{epoch_part}.png"
            new_img_path = os.path.join(root, new_img_name)

            if img_path != new_img_path:
                if os.path.exists(new_img_path):
                    os.remove(new_img_path)
                os.rename(img_path, new_img_path)

            matched_row = existing_rows_map.get(epoch_part)

            if matched_row:
                timestamp = matched_row[0]
                pixel_count = matched_row[1]
                max_val = matched_row[2]
                w_dim = matched_row[4] if len(matched_row) > 4 else gray_crop.shape[1]
                h_dim = matched_row[5] if len(matched_row) > 5 else gray_crop.shape[0]
            else:
                try:
                    epoch_time = int(epoch_part)
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(epoch_time)) + ".000000"
                except ValueError:
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ".000000"
                pixel_count = int(np.sum(gray_crop > 140))
                max_val = int(np.max(gray_crop))
                w_dim, h_dim = gray_crop.shape[1], gray_crop.shape[0]

            conf_str = f"{round(conf * 100, 1)}%"
            row = [timestamp, pixel_count, max_val, new_img_name, w_dim, h_dim, new_type, conf_str]
            processed_rows.append(row)

        if not processed_rows:
            continue

        seen_seconds = {}
        for row in processed_rows:
            ts_second = row[0].split('.')[0]
            p_type = row[6].lower().strip()
            try:
                pixel_count = int(row[1])
            except ValueError:
                pixel_count = 0

            if ts_second not in seen_seconds:
                seen_seconds[ts_second] = row
            else:
                existing_row = seen_seconds[ts_second]
                existing_type = existing_row[6].lower().strip()
                try:
                    existing_pixels = int(existing_row[1])
                except ValueError:
                    existing_pixels = 0

                if p_type in ['tracks', 'track', 'muon'] and existing_type not in ['tracks', 'track', 'muon']:
                    seen_seconds[ts_second] = row
                elif p_type not in ['tracks', 'track', 'muon'] and existing_type in ['tracks', 'track', 'muon']:
                    pass
                else:
                    if pixel_count > existing_pixels:
                        seen_seconds[ts_second] = row

        updated_rows = list(seen_seconds.values())

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Timestamp", "Pixel_Count", "Max_Brightness",
                "Image_Filename", "Width", "Height",
                "Particle_Type", "Probability"
            ])
            writer.writerows(updated_rows)

        print(f"-> Synced & Recovered CSV: {csv_path} ({len(updated_rows)} records)")

    print("Rebuilding global historical summary from all daily logs...")
    all_muon_historical_rows = []

    for root, _, files in os.walk(target_dir):
        if "muon_log.csv" in files:
            daily_csv_path = os.path.join(root, "muon_log.csv")
            with open(daily_csv_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    if len(row) > 6 and row[6].lower().strip() in ['tracks', 'track', 'muon']:
                        all_muon_historical_rows.append(row)

    all_muon_historical_rows.sort(key=lambda x: x[0])

    os.makedirs(os.path.dirname(HISTORICAL_CSV_PATH), exist_ok=True)
    with open(HISTORICAL_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Timestamp", "Pixel_Count", "Max_Brightness",
            "Image_Filename", "Width", "Height",
            "Particle_Type", "Probability"
        ])
        writer.writerows(all_muon_historical_rows)
        
    print(f"Rebuilt serverless static log with {len(all_muon_historical_rows)} muon tracks: {HISTORICAL_CSV_PATH}")


if __name__ == "__main__":
    search_path = os.path.join(BASE_DIR, "data")
    process_directory(search_path)
    print("Migration, Recovery, and Deduplication Complete!")

import csv
import json
import os
import time
import struct
import threading
from datetime import datetime
import cv2
import numpy as np
import pydobot
from flask import Flask, Response
from ultralytics import YOLO
from picamera2 import Picamera2
from gpiozero import Button

app = Flask(__name__)

# --- HARDWARE CONFIGURATION ---
DOBOT_PORT = '/dev/ttyACM0'
CONVEYOR_SPEED = 25.0

# Timing Variables
SETTLE_TIME_SEC = 1.0   # Belt runs 1s after IR trigger to center item under camera
TRAVEL_TIME_SEC = 8.0   # Belt runs 8.0s to deliver item to pickup point

# --- UPDATED ARM COORDINATES (X, Y, Z, R) ---
HOME_POS    = (137.13,   -3.12,  77.06,   -1.31)   # Position 1: Home
PICK_POS    = (138.91,   -3.16,  25.57,   -1.31)   # Position 2: Pickup on belt
BOX_PAPER   = (162.67, -134.23,  63.22,  -39.53)   # Position 3: Box 1 (Paper)
BOX_PLASTIC = (103.65, -157.52,  58.25,  -56.66)   # Position 4: Box 2 (Plastic)
BOX_GENERAL = (233.33,  -13.84,  56.68,   -3.41)   # Position 5: Box 3 (General)

HOVER_Z     = 110.0  # Safe vertical clearance height

# IR Sensor on GPIO 17
sensor_camera = Button(17, pull_up=True)

# --- ROUTING CONFIGURATION ---
# "classifier": route each item by the classifier with the confidence gate (matches the paper)
# "sequence"  : original demo mode, PAPER -> PLASTIC -> GENERAL -> repeat, classifier only logged
ROUTING_MODE = "classifier"
TARGET_SEQUENCE = ["PAPER", "PLASTIC", "GENERAL"]
sequence_index = 0

# Confidence gate (paper, Section III-D):
#   if predicted class is PAPER or PLASTIC and confidence p < TAU -> route to GENERAL
#   otherwise route to the predicted class.
# Values are read from gate_config.json (produced by the evaluation notebook) if present.
MODEL_PATH  = 'yolov8n-cls.pt'   # fallback: stock ImageNet model + keyword mapping
TEMPERATURE = 1.0                # temperature scaling, T = 1.0 means raw softmax
TAU         = 0.90               # gate threshold; tau* = k / (k + 1) for cost ratio k = L_c / L_r
RECYCLABLE  = ("PAPER", "PLASTIC")
GATE_CONFIG = 'gate_config.json'
if os.path.exists(GATE_CONFIG):
    with open(GATE_CONFIG) as f:
        _cfg = json.load(f)
    MODEL_PATH  = _cfg.get('model', MODEL_PATH)
    TEMPERATURE = float(_cfg.get('temperature', TEMPERATURE))
    TAU         = float(_cfg.get('threshold_tau', TAU))
    print(f"Loaded {GATE_CONFIG}: model={MODEL_PATH}, T={TEMPERATURE:.2f}, tau={TAU:.2f}")

# Per-item log for the evaluation protocol (paper, Section V-A).
# For a labelled run, set TRUE_LABEL before each item (e.g. via the /label/<class> URL);
# it is written to the log but never read by the routing logic.
LOG_FILE = 'sorting_log.csv'
true_label = ""

# 1. Load YOLO Model
print(f"Loading YOLO Model ({MODEL_PATH})...")
model = YOLO(MODEL_PATH)
FINE_TUNED = {n.lower() for n in model.names.values()} == {"general", "paper", "plastic"}
print("Classifier type:", "fine-tuned 3-class" if FINE_TUNED else "ImageNet + keyword mapping")

# 2. Initialize Dobot
try:
    device = pydobot.Dobot(port=DOBOT_PORT, verbose=False)
    print("Dobot Connected Successfully!")
    device.grip(False)
    device.move_to(*HOME_POS, wait=True)
except Exception as e:
    print(f"Dobot Connection Failed: {e}")
    device = None

# Conveyor Motor Constants
STEP = (360.0 / 1.8) * 10.0 * 16.0
MM = 3.1415926535898 * 36.0

def set_conveyor(speed_mms=25.0, enable=True):
    """Direct motor control for Stepper 1."""
    if not device:
        return
    try:
        on = 1 if enable and speed_mms > 0 else 0
        speed_val = int(speed_mms * STEP / MM) if on else 0
        msg = pydobot.message.Message()
        msg.id = 135
        msg.ctrl = 0x00
        msg.params = bytearray([0, on])
        msg.params.extend(struct.pack("<i", speed_val))
        device._send_command(msg)
    except Exception as e:
        print(f"Conveyor Error: {e}")

# Keyword dictionaries for classification matching
PAPER_KEYWORDS = [
    'paper', 'cardboard', 'carton', 'envelope', 'binder', 'book', 'notebook',
    'newspaper', 'towel', 'toilet_tissue', 'tissue', 'paper_towel', 'packet', 'menu'
]

PLASTIC_KEYWORDS = [
    'bottle', 'water_bottle', 'cup', 'plastic', 'pill_bottle', 'can', 'spray',
    'tub', 'bucket', 'shampoo', 'beaker', 'flask', 'vial', 'jug'
]

def classify_waste(predictions):
    for pred_name in predictions:
        clean_name = pred_name.lower().replace('-', '_').replace(' ', '_')
        if any(k in clean_name for k in PAPER_KEYWORDS):
            return "PAPER", pred_name
        if any(k in clean_name for k in PLASTIC_KEYWORDS):
            return "PLASTIC", pred_name
    return "GENERAL", predictions[0]

def classify(image):
    """Returns (predicted category, confidence p, raw model label)."""
    r = model(image, imgsz=224, verbose=False)[0]
    probs = np.clip(r.probs.data.cpu().numpy().astype(np.float64), 1e-12, 1.0)
    # temperature scaling: softmax(log p / T)
    z = np.log(probs) / TEMPERATURE
    p = np.exp(z - z.max())
    p /= p.sum()
    if FINE_TUNED:
        idx = int(p.argmax())
        name = r.names[idx]
        return name.upper(), float(p[idx]), name
    # ImageNet fallback: first top-5 label that matches a keyword list decides the category
    top5 = list(np.argsort(p)[::-1][:5])
    names = [r.names[i] for i in top5]
    category, raw = classify_waste(names)
    return category, float(p[top5[names.index(raw)]]), raw

def apply_gate(category, confidence):
    """Confidence gate. Returns (routed category, diverted flag)."""
    if category in RECYCLABLE and confidence < TAU:
        return "GENERAL", True
    return category, False

def log_item(row):
    new = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)

# 3. PiCamera2 Setup with BGR native format
print("Initializing PiCamera2...")
picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (640, 480), "format": "BGR888"})
picam2.configure(config)
picam2.set_controls({"AwbMode": 0})  # Auto White Balance
picam2.start()
time.sleep(2.0)

latest_frame = None
latest_jpeg = None
lock = threading.Lock()

current_label = ("Waiting for item..." if ROUTING_MODE == "classifier"
                 else f"Waiting for {TARGET_SEQUENCE[sequence_index]}...")
is_processing = False

def camera_worker():
    global latest_frame, latest_jpeg
    while True:
        frame = picam2.capture_array()

        h, w, _ = frame.shape
        if ROUTING_MODE == "classifier":
            header = f"Mode: classifier + gate (tau={TAU:.2f}, T={TEMPERATURE:.2f})"
        else:
            header = f"Next Target: [{TARGET_SEQUENCE[sequence_index]}]"

        # Draw overlays in standard BGR (Yellow: 0, 255, 255 | Green: 0, 255, 0)
        cv2.rectangle(frame, (w//4, h//4), (3*w//4, 3*h//4), (0, 255, 255), 2)
        cv2.putText(frame, header, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Status: {current_label}", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

        if ret:
            with lock:
                latest_frame = frame.copy()
                latest_jpeg = buffer.tobytes()
        time.sleep(0.02)

cam_thread = threading.Thread(target=camera_worker, daemon=True)
cam_thread.start()

def execute_pick_and_drop(category):
    """Executes pick and drops directly to the assigned box."""
    if category == "PAPER":
        target_pos = BOX_PAPER
        box_name = "Paper Box (Box 1)"
    elif category == "PLASTIC":
        target_pos = BOX_PLASTIC
        box_name = "Plastic Box (Box 2)"
    else:
        target_pos = BOX_GENERAL
        box_name = "General Box (Box 3)"

    print(f"\n[DOBOT] Delivering [{category}] to {box_name} at {target_pos}...")

    if device:
        # 1. Move to pickup position
        device.grip(False)
        device.move_to(PICK_POS[0], PICK_POS[1], HOVER_Z, PICK_POS[3], wait=True)
        device.move_to(*PICK_POS, wait=True)

        # 2. Grip object
        device.grip(True)
        time.sleep(0.5)

        # 3. Lift to clearance height
        device.move_to(PICK_POS[0], PICK_POS[1], HOVER_Z, PICK_POS[3], wait=True)

        # 4. Move above assigned box and release
        device.move_to(target_pos[0], target_pos[1], HOVER_Z, target_pos[3], wait=True)
        device.move_to(*target_pos, wait=True)
        device.grip(False)
        time.sleep(0.5)

        # 5. Ascend and return to home safely
        device.move_to(target_pos[0], target_pos[1], HOVER_Z, target_pos[3], wait=True)
        device.move_to(*HOME_POS, wait=True)
        print(f"[DOBOT] Object dropped in {box_name}. Arm returned HOME.")

def sorting_logic_worker():
    global current_label, is_processing, sequence_index, true_label

    set_conveyor(CONVEYOR_SPEED, enable=True)

    while True:
        if sensor_camera.is_pressed and not is_processing:
            is_processing = True
            t_start = time.time()
            print(f"\n--> [IR TRIGGER] Object detected. Centering under camera ({SETTLE_TIME_SEC}s)...")

            # 1. Move 1s to center item
            time.sleep(SETTLE_TIME_SEC)

            # 2. Stop conveyor for YOLO scan
            set_conveyor(0, enable=False)
            current_label = "Scanning with YOLO..."
            time.sleep(0.5)

            # 3. Grab snapshot and run inference
            with lock:
                snap = latest_frame.copy() if latest_frame is not None else None

            predicted, conf, raw_pred, infer_ms = "GENERAL", 0.0, "no_frame", 0.0
            if snap is not None:
                h, w, _ = snap.shape
                crop = snap[h//4:3*h//4, w//4:3*w//4]
                t_inf = time.time()
                predicted, conf, raw_pred = classify(crop)
                infer_ms = (time.time() - t_inf) * 1000

            # 4. Decide the target box
            if ROUTING_MODE == "classifier":
                current_target, diverted = apply_gate(predicted, conf)
            else:
                current_target, diverted = TARGET_SEQUENCE[sequence_index], False
            note = " (DIVERTED by gate)" if diverted else ""
            print(f"--> [YOLO SCAN] {raw_pred} -> {predicted} (p={conf:.2f}, {infer_ms:.0f} ms) "
                  f"| Routed: [{current_target}]{note}")
            current_label = f"{predicted} p={conf:.2f} -> [{current_target}]{note}"
            time.sleep(1.0)

            # 5. Transport item to Dobot pickup point
            print(f"--> Transporting [{current_target}] to pickup zone ({TRAVEL_TIME_SEC}s)...")
            set_conveyor(CONVEYOR_SPEED, enable=True)
            sensor_camera.wait_for_release()
            time.sleep(TRAVEL_TIME_SEC)

            # 6. Stop belt at pickup point
            set_conveyor(0, enable=False)
            current_label = f"Dobot Sorting [{current_target}]..."

            # 7. Execute pick and place
            execute_pick_and_drop(current_target)

            # 8. Log the cycle (evaluation protocol, paper Section V-A)
            log_item({
                'timestamp': datetime.now().isoformat(timespec='seconds'),
                'mode': ROUTING_MODE, 'true_class': true_label,
                'predicted_class': predicted, 'raw_label': raw_pred,
                'confidence': round(conf, 4), 'tau': TAU, 'temperature': TEMPERATURE,
                'routed_class': current_target, 'diverted': int(diverted),
                'inference_ms': round(infer_ms, 1),
                'cycle_time_s': round(time.time() - t_start, 2),
            })
            true_label = ""

            # 9. Advance demo sequence (sequence mode only)
            if ROUTING_MODE == "sequence":
                sequence_index = (sequence_index + 1) % len(TARGET_SEQUENCE)
            print("--> Cycle complete!")

            # 10. Resume belt
            current_label = "Ready for next item (Belt Running...)"
            set_conveyor(CONVEYOR_SPEED, enable=True)
            is_processing = False

        time.sleep(0.05)

logic_thread = threading.Thread(target=sorting_logic_worker, daemon=True)
logic_thread.start()

def generate_stream():
    while True:
        with lock:
            frame_data = latest_jpeg
        if frame_data is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
        time.sleep(0.03)

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>EcoNexus Sorting Line</title>
        <style>
            body { background: #0f172a; color: white; text-align: center; font-family: sans-serif; padding-top: 30px; }
            img { width: 640px; border-radius: 10px; border: 3px solid #38bdf8; }
            .info { margin-top: 15px; font-size: 18px; color: #38bdf8; font-weight: bold; }
        </style>
    </head>
    <body>
        <h2>EcoNexus Multi-Box Automated Sorting</h2>
        <img src="/video_feed">
        <div class="info">Box 1: Paper | Box 2: Plastic | Box 3: General</div>
    </body>
    </html>
    '''

@app.route('/label/<cls>')
def set_label(cls):
    """Operator records the true class of the NEXT item (logged only, never used for routing)."""
    global true_label
    cls = cls.upper()
    if cls not in ("PAPER", "PLASTIC", "GENERAL"):
        return "Use /label/paper, /label/plastic or /label/general", 400
    true_label = cls
    return f"True class of next item recorded as {cls}"

@app.route('/video_feed')
def video_feed():
    return Response(generate_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=8000, threaded=True)
    finally:
        set_conveyor(0, enable=False)
        if device:
            device.close()
        picam2.stop()

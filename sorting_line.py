import time
import struct
import threading
import cv2
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

# 3-Step Sequence: PAPER -> PLASTIC -> GENERAL -> Repeat
TARGET_SEQUENCE = ["PAPER", "PLASTIC", "GENERAL"]
sequence_index = 0

# 1. Load YOLO Model
print("Loading YOLO Model...")
model = YOLO('yolov8n-cls.pt')

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

current_label = f"Waiting for {TARGET_SEQUENCE[sequence_index]}..."
is_processing = False

def camera_worker():
    global latest_frame, latest_jpeg
    while True:
        frame = picam2.capture_array()

        h, w, _ = frame.shape
        expected = TARGET_SEQUENCE[sequence_index]

        # Draw overlays in standard BGR (Yellow: 0, 255, 255 | Green: 0, 255, 0)
        cv2.rectangle(frame, (w//4, h//4), (3*w//4, 3*h//4), (0, 255, 255), 2)
        cv2.putText(frame, f"Next Target: [{expected}]", (20, 40),
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
    global current_label, is_processing, sequence_index

    set_conveyor(CONVEYOR_SPEED, enable=True)

    while True:
        if sensor_camera.is_pressed and not is_processing:
            is_processing = True
            current_target = TARGET_SEQUENCE[sequence_index]

            print(f"\n--> [IR TRIGGER] Object detected. Centering under camera ({SETTLE_TIME_SEC}s)...")

            # 1. Move 1s to center item
            time.sleep(SETTLE_TIME_SEC)

            # 2. Stop conveyor for YOLO scan
            set_conveyor(0, enable=False)
            current_label = f"Scanning [{current_target}] with YOLO..."
            time.sleep(0.5)

            # 3. Grab snapshot and run inference
            with lock:
                snap = latest_frame.copy() if latest_frame is not None else None

            if snap is not None:
                h, w, _ = snap.shape
                crop = snap[h//4:3*h//4, w//4:3*w//4]
                results = model(crop, imgsz=224, verbose=False)
                top5_ids = results[0].probs.top5
                top_names = [results[0].names[idx] for idx in top5_ids]
                _, raw_pred = classify_waste(top_names)
                conf = float(results[0].probs.top1conf) * 100
                print(f"--> [YOLO SCAN] Detected: {raw_pred} ({conf:.1f}%) | Assigned: [{current_target}]")

            current_label = f"Identified: [{current_target}]"
            time.sleep(1.0)

            # 4. Transport item to Dobot pickup point (8.0s)
            print(f"--> Transporting [{current_target}] to pickup zone ({TRAVEL_TIME_SEC}s)...")
            set_conveyor(CONVEYOR_SPEED, enable=True)
            sensor_camera.wait_for_release()
            time.sleep(TRAVEL_TIME_SEC)

            # 5. Stop belt at pickup point
            set_conveyor(0, enable=False)
            current_label = f"Dobot Sorting [{current_target}]..."

            # 6. Execute pick and place
            execute_pick_and_drop(current_target)

            # 7. Advance sequence
            sequence_index = (sequence_index + 1) % len(TARGET_SEQUENCE)
            next_target = TARGET_SEQUENCE[sequence_index]
            print(f"--> Cycle complete! Next target: [{next_target}]")

            # 8. Resume belt
            current_label = f"Ready for {next_target} (Belt Running...)"
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

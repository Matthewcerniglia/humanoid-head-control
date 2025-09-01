import cv2
import serial
import serial.tools.list_ports
import time
import threading
import numpy as np
import sys
import io
import base64
import qrcode
from PIL import Image
from flask import Flask, render_template, Response, jsonify, request
import pyaudio
from ultralytics import YOLOWorld
import logging
import queue
import os

# Suppress Ultralytics logs
logging.getLogger('ultralytics').setLevel(logging.CRITICAL)
os.environ["YOLO_VERBOSE"] = "False"

app = Flask(__name__)

# --- Detection Mode ---
detection_mode = "face"  # Default: "off", "face", "object"
mode_lock = threading.Lock()
auto_mode = "manual"  # Face selection mode: "manual", "largest", "closest", "center", "speaking", "priority"

# --- Camera Configuration ---
camera = None
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_X_HALF_RES = CAMERA_WIDTH / 2.0
CAMERA_Y_HALF_RES = CAMERA_HEIGHT / 2.0

# --- YOLO-World Configuration ---
yolo_model = None
yolo_classes = ["coaster"]  # Default object detection prompt
yolo_prompt_queue = queue.Queue()
yolo_last_boxes = []
yolo_last_names = {}
yolo_last_prompt = "coaster"
FRAME_SKIP = 5
yolo_frame_count = 0

# --- Auto Selection Configuration ---
last_auto_switch_time = 0.0
face_stability_tracker = {}
speaking_detection_history = {}
AUTO_SWITCH_COOLDOWN = 1.0  # Cooldown for subsequent selections
FORCE_INITIAL_SELECTION = True  # Flag to force selection on mode switch
FACE_STABILITY_THRESHOLD = 0.5

def initialize_yolo_model():
    global yolo_model
    model_path = "yolov8s-worldv2.pt"
    if not os.path.exists(model_path):
        print(f"ERROR: YOLO model file {model_path} not found.", file=sys.stderr)
        return False
    try:
        yolo_model = YOLOWorld(model_path)
        yolo_model.set_classes(yolo_classes)
        print("YOLO-World model loaded successfully.")
        return True
    except Exception as e:
        print(f"ERROR: Failed to load YOLO model: {e}", file=sys.stderr)
        return False

initialize_yolo_model()

def initialize_camera():
    global camera, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_X_HALF_RES, CAMERA_Y_HALF_RES
    for index in [0, 1, 2]:
        camera = cv2.VideoCapture(index)
        if camera.isOpened():
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            CAMERA_WIDTH = 640
            CAMERA_HEIGHT = 480
            CAMERA_X_HALF_RES = CAMERA_WIDTH / 2.0
            CAMERA_Y_HALF_RES = CAMERA_HEIGHT / 2.0
            print(f"Camera opened at index {index}. Resolution: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
            return True
    print("ERROR: Could not open any camera.")
    return False

if not initialize_camera():
    sys.exit(1)

# --- Serial Connection ---
ser = None

def initialize_serial():
    global ser
    target_port = '/dev/cu.usbmodemF412FA6370882'
    try:
        ser = serial.Serial(target_port, 115200, timeout=1)
        time.sleep(2)
        print(f"Serial port {target_port} opened successfully.")
        return True
    except serial.SerialException as e:
        print(f"Failed to open {target_port}: {e}")
        print("WARNING: Continuing without serial communication.")
        return False

initialize_serial()

# --- Face Detection ---
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if face_classifier.empty():
    print("CRITICAL ERROR: Could not load Haar cascade classifier.")
    sys.exit(1)

# --- Audio Setup ---
p = pyaudio.PyAudio()
audio_stream = None
current_mouth_angle = 70
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
AUDIO_THRESHOLD = 35
MIN_SERVO_ANGLE = 70
MAX_SERVO_ANGLE = 110
RMS_MIN_INPUT = 50
RMS_MAX_INPUT = 500
RMS_SENSITIVITY = 50

def find_audio_device():
    input_device_index = -1
    print("Available audio input devices:")
    for i in range(p.get_device_count()):
        dev_info = p.get_device_info_by_index(i)
        print(f"  {i}: {dev_info['name']} (Input Channels: {dev_info['maxInputChannels']})")
        if "blackhole" in dev_info['name'].lower() and dev_info['maxInputChannels'] > 0:
            input_device_index = i
            print(f"  -> Found BlackHole at index {i}")
            break
    return input_device_index

audio_device_index = find_audio_device()

# --- Global Variables ---
next_face_id = 0
tracked_faces = {}
IOU_THRESHOLD = 0.2  # Lowered for better tracking
TRACKING_TIMEOUT_S = 1.5
face_tracking_lock = threading.Lock()
current_faces_info = []
selected_face_id_for_arduino = -1
last_known_x_dist = 0.0
last_known_y_dist = 0.0
last_detection_time = time.time()
detection_timeout_s = 5.0  # Retain position longer
vision_and_eyes_active = True
audio_tracking_active = False
last_system_active_state = False
smoothing_alpha = 0.2
current_rms = 0.0
force_selection = False

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    return interArea / float(boxAArea + boxBArea - interArea)

def calculate_face_area(box):
    w, h = box[2], box[3]
    return w * h

def calculate_distance_from_center(box, screen_center):
    x, y, w, h = box
    face_center = (x + w / 2, y + h / 2)
    return ((face_center[0] - screen_center[0]) ** 2 + (face_center[1] - screen_center[1]) ** 2) ** 0.5

def calculate_face_score(face_id, box, screen_center, current_rms):
    area = calculate_face_area(box)
    distance = calculate_distance_from_center(box, screen_center)
    center_bonus = max(0, 160 - distance) / 160
    area_score = area / (CAMERA_WIDTH * CAMERA_HEIGHT)
    speaking_score = 0
    current_time = time.time()
    if current_rms > 50:
        recent_speaking = len([t for t in speaking_detection_history.get(face_id, []) if current_time - t < 3.0])
        speaking_score = recent_speaking * 0.1
    return center_bonus + area_score + speaking_score

def auto_select_face(tracked_faces, screen_center, current_rms=0):
    global last_auto_switch_time, face_stability_tracker, speaking_detection_history, selected_face_id_for_arduino, force_selection
    if not tracked_faces or auto_mode == "manual":
        print(f"auto_select_face: No faces or manual mode (tracked_faces={len(tracked_faces)}, auto_mode={auto_mode})")
        return None
    current_time = time.time()
    for face_id, face_data in tracked_faces.items():
        if face_id not in face_stability_tracker:
            face_stability_tracker[face_id] = current_time
            print(f"New face {face_id} added to stability tracker at {current_time}")
        if current_rms > 100:
            speaking_detection_history.setdefault(face_id, []).append(current_time)
            speaking_detection_history[face_id] = [t for t in speaking_detection_history[face_id]
                                                  if current_time - t < 10.0]
            print(f"Face {face_id}: Updated speaking history, RMS={current_rms}")
    active_face_ids = set(tracked_faces.keys())
    face_stability_tracker = {fid: timestamp for fid, timestamp in face_stability_tracker.items()
                              if fid in active_face_ids}
    speaking_detection_history = {fid: history for fid, history in speaking_detection_history.items()
                                  if fid in active_face_ids}
    if not force_selection and current_time - last_auto_switch_time < AUTO_SWITCH_COOLDOWN:
        print(f"auto_select_face: Cooldown active, last switch at {last_auto_switch_time}, current_time={current_time}")
        return None
    best_face_id = -1
    best_score = -1
    if auto_mode == "largest":
        largest_area = 0
        for face_id, face_data in tracked_faces.items():
            area = calculate_face_area(face_data['box'])
            if area > largest_area:
                largest_area = area
                best_face_id = face_id
                best_score = area / (CAMERA_WIDTH * CAMERA_HEIGHT)
        print(f"auto_select_face: largest mode, best_face_id={best_face_id}, area={largest_area}, score={best_score}")
    elif auto_mode == "closest":
        min_distance = float('inf')
        for face_id, face_data in tracked_faces.items():
            distance = calculate_distance_from_center(face_data['box'], screen_center)
            if distance < min_distance:
                min_distance = distance
                best_face_id = face_id
                best_score = max(0, 160 - distance) / 160
        print(f"auto_select_face: closest mode, best_face_id={best_face_id}, distance={min_distance}, score={best_score}")
    elif auto_mode == "center":
        candidates = []
        for face_id, face_data in tracked_faces.items():
            distance = calculate_distance_from_center(face_data['box'], screen_center)
            candidates.append((face_id, distance))
        if candidates:
            best_face_id = min(candidates, key=lambda x: x[1])[0]
            best_score = max(0, 160 - min(candidates, key=lambda x: x[1])[1]) / 160
            print(f"auto_select_face: center mode, best_face_id={best_face_id}, distance={min(candidates, key=lambda x: x[1])[1]}, score={best_score}")
    elif auto_mode == "speaking":
        if current_rms > 50:
            for face_id, face_data in tracked_faces.items():
                center_bonus = max(0, 160 - calculate_distance_from_center(face_data['box'], screen_center)) / 160
                recent_speaking = len([t for t in speaking_detection_history.get(face_id, [])
                                       if current_time - t < 3.0])
                speaking_score = center_bonus + recent_speaking * 0.1
                if speaking_score > best_score:
                    best_score = speaking_score
                    best_face_id = face_id
            print(f"auto_select_face: speaking mode, best_face_id={best_face_id}, score={best_score}, RMS={current_rms}")
        else:
            largest_area = 0
            for face_id, face_data in tracked_faces.items():
                area = calculate_face_area(face_data['box'])
                if area > largest_area:
                    largest_area = area
                    best_face_id = face_id
                    best_score = area / (CAMERA_WIDTH * CAMERA_HEIGHT)
            print(f"auto_select_face: speaking mode (no audio), best_face_id={best_face_id}, area={largest_area}, score={best_score}")
    elif auto_mode == "priority":
        for face_id, face_data in tracked_faces.items():
            if current_time - face_stability_tracker.get(face_id, current_time) >= FACE_STABILITY_THRESHOLD:
                score = calculate_face_score(face_id, face_data['box'], screen_center, current_rms)
                if score > best_score:
                    best_score = score
                    best_face_id = face_id
        print(f"auto_select_face: priority mode, best_face_id={best_face_id}, score={best_score}")
    if best_face_id != -1:
        last_auto_switch_time = current_time
        force_selection = False
        print(f"auto_select_face: Selected face {best_face_id} using {auto_mode} mode, score={best_score}, force_selection={force_selection}")
        return best_face_id
    print(f"auto_select_face: No valid face selected, best_face_id={best_face_id}, current_selected={selected_face_id_for_arduino}")
    return None

def audio_callback(in_data, frame_count, time_info, status):
    global current_mouth_angle, current_rms
    with face_tracking_lock:
        sensitivity = RMS_SENSITIVITY
    if not audio_tracking_active or not in_data or len(in_data) == 0:
        current_mouth_angle = MIN_SERVO_ANGLE
        current_rms = 0.0
        return in_data, pyaudio.paContinue
    audio_data = np.frombuffer(in_data, dtype=np.int16)
    if audio_data.size == 0:
        current_mouth_angle = MIN_SERVO_ANGLE
        current_rms = 0.0
        return in_data, pyaudio.paContinue
    current_rms = np.sqrt(np.mean(audio_data.astype(np.float64) ** 2) + 1e-9)
    rms_min = RMS_MIN_INPUT * (1 - sensitivity / 200)
    rms_max = RMS_MAX_INPUT * (1 + sensitivity / 50)
    clamped_rms = np.clip(current_rms, rms_min, rms_max)
    target_angle = np.interp(clamped_rms, [rms_min, rms_max], [MIN_SERVO_ANGLE, MAX_SERVO_ANGLE])
    current_mouth_angle = int(np.round(np.clip(target_angle, MIN_SERVO_ANGLE, MAX_SERVO_ANGLE)))
    return in_data, pyaudio.paContinue

def start_audio_stream():
    global audio_stream
    if audio_device_index == -1:
        print("No suitable audio device found. Audio tracking disabled.")
        return False
    try:
        audio_stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=audio_device_index,
            frames_per_buffer=CHUNK,
            stream_callback=audio_callback
        )
        audio_stream.start_stream()
        print("Audio stream started successfully.")
        return True
    except Exception as e:
        print(f"Failed to start audio stream: {e}")
        return False

def stop_audio_stream():
    global audio_stream
    if audio_stream and audio_stream.is_active():
        audio_stream.stop_stream()
        audio_stream.close()
        audio_stream = None
        print("Audio stream stopped.")

def detect_track_and_draw(vid, draw_boxes_enabled):
    global next_face_id, tracked_faces, selected_face_id_for_arduino
    global last_known_x_dist, last_known_y_dist, last_detection_time
    global current_faces_info, vision_and_eyes_active, yolo_last_boxes
    global yolo_last_names, yolo_last_prompt, yolo_frame_count
    screen_center = (CAMERA_WIDTH // 2, CAMERA_HEIGHT // 2)

    with mode_lock:
        current_mode = detection_mode
        current_auto_mode = auto_mode

    print(f"detect_track_and_draw: mode={current_mode}, auto_mode={current_auto_mode}, vision_active={vision_and_eyes_active}, tracked_faces={len(tracked_faces)}, selected_id={selected_face_id_for_arduino}")

    # Clear data and skip processing if off or vision disabled
    if current_mode == "off" or not vision_and_eyes_active:
        with face_tracking_lock:
            current_faces_info = []
            tracked_faces.clear()
            selected_face_id_for_arduino = -1
        yolo_last_boxes = []
        print("detect_track_and_draw: Mode off or vision disabled, clearing data")
        return vid

    # Initialize output variables
    x_distance_arduino = 0.0
    y_distance_arduino = 0.0
    boxes_to_draw = []
    labels_to_draw = []

    if current_mode == "face":
        # Face detection
        if vision_and_eyes_active:
            gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
            current_detections = face_classifier.detectMultiScale(
                gray_image,
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(30, 30)
            )
            print(f"detect_track_and_draw: Detected {len(current_detections)} faces: {[(x, y, w, h) for (x, y, w, h) in current_detections]}")
            with face_tracking_lock:
                # Update tracked faces
                matched_current_detection_indices = set()
                for i, (x, y, w, h) in enumerate(current_detections):
                    matched_id = None
                    best_iou = 0.0
                    for face_id, face_data in list(tracked_faces.items()):
                        current_iou = iou([x, y, w, h], face_data['box'])
                        if current_iou > IOU_THRESHOLD and current_iou > best_iou:
                            best_iou = current_iou
                            matched_id = face_id
                    if matched_id is not None:
                        tracked_faces[matched_id]['box'] = [x, y, w, h]
                        tracked_faces[matched_id]['last_seen'] = time.time()
                        matched_current_detection_indices.add(i)
                    else:
                        tracked_faces[next_face_id] = {'box': [x, y, w, h], 'last_seen': time.time()}
                        print(f"New face assigned ID {next_face_id}")
                        next_face_id += 1

                # Remove stale faces
                stale_ids = [fid for fid, data in tracked_faces.items() if time.time() - data['last_seen'] > TRACKING_TIMEOUT_S]
                for fid in stale_ids:
                    del tracked_faces[fid]
                    if selected_face_id_for_arduino == fid:
                        selected_face_id_for_arduino = -1
                        print(f"Removed stale face {fid}, reset selected_face_id_for_arduino")

                # Prepare face info for UI
                current_faces_info = []
                target_face_for_arduino_box = None
                found_selected_face = False
                sorted_tracked_faces = sorted(tracked_faces.items(), key=lambda x: x[0])

                for idx, (face_id, face_data) in enumerate(sorted_tracked_faces):
                    box = face_data['box']
                    current_faces_info.append({
                        'id': int(face_id),
                        'x': int(box[0]),
                        'y': int(box[1]),
                        'w': int(box[2]),
                        'h': int(box[3])
                    })
                    if face_id == selected_face_id_for_arduino:
                        target_face_for_arduino_box = box
                        found_selected_face = True
                    if draw_boxes_enabled:
                        # Only draw the selected face's box if a face is selected
                        if selected_face_id_for_arduino != -1 and face_id == selected_face_id_for_arduino:
                            color = (0, 255, 0)  # Green for selected
                            print(f"Drawing selected face {face_id}: Color={color}, Box={box}, Selected ID={selected_face_id_for_arduino}")
                            boxes_to_draw.append((box[0], box[1], box[0] + box[2], box[1] + box[3], color))
                            labels_to_draw.append((f"Face {idx + 1}", box[0], max(box[1] - 10, 10), color))
                        elif selected_face_id_for_arduino == -1:
                            color = (255, 0, 0)  # Blue for all faces when none selected
                            print(f"Drawing face {face_id}: Color={color}, Box={box}, Selected ID={selected_face_id_for_arduino}")
                            boxes_to_draw.append((box[0], box[1], box[0] + box[2], box[1] + box[3], color))
                            labels_to_draw.append((f"Face {idx + 1}", box[0], max(box[1] - 10, 10), color))

                # Auto-select face if not in manual mode or forced selection
                if current_auto_mode != "manual" and tracked_faces:
                    new_face_id = auto_select_face(tracked_faces, screen_center, current_rms)
                    if new_face_id is not None:
                        selected_face_id_for_arduino = new_face_id
                        print(f"Auto-selected face ID: {selected_face_id_for_arduino}")
                        # Update drawing to reflect new selection
                        if draw_boxes_enabled:
                            boxes_to_draw = []
                            labels_to_draw = []
                            for idx, (face_id, face_data) in enumerate(sorted_tracked_faces):
                                box = face_data['box']
                                if selected_face_id_for_arduino != -1 and face_id == selected_face_id_for_arduino:
                                    color = (0, 255, 0)  # Green for selected
                                    print(f"Re-drawing selected face {face_id}: Color={color}, Box={box}, Selected ID={selected_face_id_for_arduino}")
                                    boxes_to_draw.append((box[0], box[1], box[0] + box[2], box[1] + box[3], color))
                                    labels_to_draw.append((f"Face {idx + 1}", box[0], max(box[1] - 10, 10), color))
                                elif selected_face_id_for_arduino == -1:
                                    color = (255, 0, 0)  # Blue for all faces when none selected
                                    print(f"Re-drawing face {face_id}: Color={color}, Box={box}, Selected ID={selected_face_id_for_arduino}")
                                    boxes_to_draw.append((box[0], box[1], box[0] + box[2], box[1] + box[3], color))
                                    labels_to_draw.append((f"Face {idx + 1}", box[0], max(box[1] - 10, 10), color))

                # Calculate distances for Arduino
                if selected_face_id_for_arduino != -1 and target_face_for_arduino_box is not None:
                    tx, ty, tw, th = target_face_for_arduino_box
                    face_center = (tx + tw // 2, ty + th // 2)
                    x_distance_arduino = face_center[0] - screen_center[0]
                    y_distance_arduino = face_center[1] - screen_center[1]
                    last_known_x_dist = x_distance_arduino
                    last_known_y_dist = y_distance_arduino
                    last_detection_time = time.time()
                    print(f"Selected face {selected_face_id_for_arduino}: Center=({face_center[0]}, {face_center[1]}), Distances=(x={x_distance_arduino}, y={y_distance_arduino})")
                elif selected_face_id_for_arduino != -1 and (time.time() - last_detection_time) < detection_timeout_s and selected_face_id_for_arduino in tracked_faces:
                    x_distance_arduino = last_known_x_dist
                    y_distance_arduino = last_known_y_dist
                    print(f"Using last known distances for face {selected_face_id_for_arduino}: x={x_distance_arduino}, y={y_distance_arduino}")
                else:
                    if selected_face_id_for_arduino != -1:
                        print(f"Resetting selected_face_id_for_arduino: No valid face found, selected_id={selected_face_id_for_arduino}, found_selected_face={found_selected_face}, in_tracked_faces={selected_face_id_for_arduino in tracked_faces}, timeout_exceeded={time.time() - last_detection_time >= detection_timeout_s}")
                        selected_face_id_for_arduino = -1
                    x_distance_arduino = 0.0
                    y_distance_arduino = 0.0
        else:
            # Clear face data if vision is disabled
            with face_tracking_lock:
                current_faces_info = []
                selected_face_id_for_arduino = -1
                tracked_faces.clear()
            print("detect_track_and_draw: Vision disabled, cleared face data")
    elif current_mode == "object":
        with face_tracking_lock:
            current_faces_info = []
            selected_face_id_for_arduino = -1
            tracked_faces.clear()
        if yolo_model is None:
            return vid
        with mode_lock:
            yolo_frame_count += 1
            try:
                new_prompt = yolo_prompt_queue.get_nowait()
                yolo_classes.clear()
                yolo_classes.append(new_prompt)
                yolo_model.set_classes(yolo_classes)
                yolo_last_prompt = new_prompt
                yolo_last_boxes = []
            except queue.Empty:
                pass
            if yolo_frame_count % FRAME_SKIP == 0:
                results = yolo_model.predict(vid, stream=True, verbose=False)
                yolo_last_boxes = []
                for result in results:
                    if isinstance(result.names, dict):
                        yolo_last_names = result.names
                    else:
                        yolo_last_names = {i: str(name) for i, name in enumerate(result.names)} if result.names else {0: "Unknown"}
                    for box in result.boxes:
                        if box.conf[0] > 0.4:
                            yolo_last_boxes.append({
                                'xyxy': box.xyxy[0].cpu().numpy(),
                                'cls': int(box.cls[0]),
                                'conf': float(box.conf[0])
                            })
            for box in yolo_last_boxes:
                x1, y1, x2, y2 = map(int, box['xyxy'])
                cls = box['cls']
                class_name = yolo_last_names.get(cls, "Unknown")
                conf = box['conf']
                if draw_boxes_enabled:
                    boxes_to_draw.append((x1, y1, x2, y2, (0, 255, 0)))
                    labels_to_draw.append((f"{class_name} {conf:.2f}", x1, max(y1 - 10, 10), (0, 255, 0)))

    if draw_boxes_enabled:
        for (x1, y1, x2, y2, color) in boxes_to_draw:
            cv2.rectangle(vid, (x1, y1), (x2, y2), color, 2)
        for (text, x, y, color) in labels_to_draw:
            cv2.putText(vid, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    return vid

def generate_frames():
    global last_system_active_state, last_known_x_dist, last_known_y_dist
    global selected_face_id_for_arduino, vision_and_eyes_active, audio_tracking_active
    global smoothing_alpha, current_mouth_angle, current_rms
    draw_boxes_enabled = True
    while True:
        if not camera.isOpened():
            print("Camera disconnected. Attempting to reinitialize...")
            if not initialize_camera():
                time.sleep(1)
                continue
        success, frame = camera.read()
        if not success:
            print("Failed to read frame from camera.")
            time.sleep(0.1)
            continue
        with mode_lock:
            current_mode = detection_mode
            status_text = f"Mode: {current_mode.capitalize()} | Vision: {'ON' if vision_and_eyes_active else 'OFF'} | Audio: {'ON' if audio_tracking_active else 'OFF'}"
            if current_mode == "object":
                status_text += f" | Detecting: {yolo_last_prompt}"
            elif current_mode == "face" and audio_tracking_active:
                with face_tracking_lock:
                    status_text += f" | Sensitivity: {RMS_SENSITIVITY} | RMS: {current_rms:.1f} | Mouth: {current_mouth_angle}°"
        if current_mode != "off":
            frame = detect_track_and_draw(frame, draw_boxes_enabled)
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if ser and ser.is_open:
            try:
                system_active = (current_mode == "face" and vision_and_eyes_active) or (current_mode == "face" and audio_tracking_active)
                if system_active != last_system_active_state:
                    command = "SYSTEM_ON\n" if system_active else "SYSTEM_OFF\n"
                    ser.write(command.encode('utf-8'))
                    print(f"Sent to Arduino: {command.strip()}")
                    last_system_active_state = system_active
                if system_active and current_mode == "face":
                    num_faces = 1 if (vision_and_eyes_active and selected_face_id_for_arduino != -1) else 0
                    eye_x = last_known_x_dist if vision_and_eyes_active else 0.0
                    eye_y = last_known_y_dist if vision_and_eyes_active else 0.0
                    mouth_angle = current_mouth_angle if audio_tracking_active else MIN_SERVO_ANGLE
                    command = f"COMBINED,{num_faces},{eye_x:.1f},{eye_y:.1f},{mouth_angle},{smoothing_alpha:.2f}\n"
                    ser.write(command.encode('utf-8'))
                    print(f"Sent to Arduino: {command.strip()}")
                if ser.in_waiting > 0:
                    response = ser.readline().decode('utf-8').strip()
                    if response:
                        print(f"Arduino: {response}")
            except serial.SerialException as e:
                print(f"Serial communication error: {e}")
                if ser:
                    ser.close()
                initialize_serial()
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ret:
            print("Failed to encode frame.")
            continue
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('camera.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_face_data')
def get_face_data():
    with face_tracking_lock:
        with mode_lock:
            return jsonify({
                'faces': current_faces_info,
                'selected_face_id': selected_face_id_for_arduino,
                'vision_active': vision_and_eyes_active,
                'audio_active': audio_tracking_active,
                'current_rms': float(current_rms),
                'mouth_angle': int(current_mouth_angle),
                'mode': detection_mode,
                'auto_mode': auto_mode,
                'object_prompt': yolo_last_prompt if detection_mode == "object" else "",
                'sensitivity': RMS_SENSITIVITY
            })

@app.route('/set_selected_face', methods=['POST'])
def set_selected_face():
    global selected_face_id_for_arduino, force_selection
    try:
        data = request.get_json()
        face_id = int(data.get('face_id'))
        with face_tracking_lock:
            with mode_lock:
                if detection_mode == "face" and (face_id == -1 or face_id in tracked_faces):
                    selected_face_id_for_arduino = face_id
                    force_selection = False
                    print(f"Set selected face ID: {selected_face_id_for_arduino}")
                    return jsonify({'status': 'success', 'selected_id': selected_face_id_for_arduino})
                else:
                    return jsonify({'status': 'error', 'message': 'Invalid face ID or not in face mode'}), 400
    except Exception as e:
        print(f"ERROR: set_selected_face: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/toggle_vision_eyes', methods=['POST'])
def toggle_vision_eyes():
    global vision_and_eyes_active
    try:
        data = request.get_json()
        with face_tracking_lock:
            vision_and_eyes_active = bool(data.get('active'))
        print(f"Vision state set to: {vision_and_eyes_active}")
        return jsonify({'status': 'success', 'vision_active': vision_and_eyes_active})
    except Exception as e:
        print(f"ERROR: toggle_vision_eyes: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/toggle_audio_tracking', methods=['POST'])
def toggle_audio_tracking():
    global audio_tracking_active
    try:
        data = request.get_json()
        new_state = bool(data.get('active'))
        with face_tracking_lock:
            if new_state and not audio_tracking_active:
                if start_audio_stream():
                    audio_tracking_active = True
                    print("Audio tracking started.")
                else:
                    return jsonify({'status': 'error', 'message': 'Failed to start audio stream'}), 500
            elif not new_state and audio_tracking_active:
                stop_audio_stream()
                audio_tracking_active = False
                print("Audio tracking stopped.")
            else:
                audio_tracking_active = new_state
        return jsonify({'status': 'success', 'audio_active': audio_tracking_active})
    except Exception as e:
        print(f"ERROR: toggle_audio_tracking: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/set_smoothing', methods=['POST'])
def set_smoothing():
    global smoothing_alpha
    try:
        data = request.get_json()
        alpha = float(data.get('alpha'))
        if 0 <= alpha <= 1:
            with face_tracking_lock:
                smoothing_alpha = alpha
            print(f"Smoothing alpha set to: {smoothing_alpha:.2f}")
            return jsonify({'status': 'success', 'alpha': smoothing_alpha})
        else:
            return jsonify({'status': 'error', 'message': 'Alpha must be between 0 and 1'}), 400
    except Exception as e:
        print(f"ERROR: set_smoothing: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/set_auto_mode', methods=['POST'])
def set_auto_mode():
    global auto_mode, selected_face_id_for_arduino, force_selection
    try:
        data = request.get_json()
        mode = data.get('mode')
        valid_modes = ["manual", "largest", "closest", "center", "speaking", "priority"]
        if mode in valid_modes:
            with mode_lock:
                with face_tracking_lock:
                    if detection_mode == "face":
                        auto_mode = mode
                        if mode != "manual":
                            selected_face_id_for_arduino = -1
                            force_selection = True  # Force immediate selection
                            print(f"Forcing initial face selection for mode {mode}")
                        else:
                            force_selection = False
                        print(f"Set auto mode: {mode}, Selected ID={selected_face_id_for_arduino}, Force selection={force_selection}")
                        return jsonify({'status': 'success', 'mode': auto_mode})
                    else:
                        return jsonify({'status': 'error', 'message': 'Not in face detection mode'}), 400
        else:
            return jsonify({'status': 'error', 'message': 'Invalid auto mode'}), 400
    except Exception as e:
        print(f"ERROR: set_auto_mode: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/set_audio_sensitivity', methods=['POST'])
def set_audio_sensitivity():
    global RMS_SENSITIVITY
    try:
        data = request.get_json()
        sensitivity = float(data.get('sensitivity'))
        if 0 <= sensitivity <= 100:
            with face_tracking_lock:
                RMS_SENSITIVITY = sensitivity
            print(f"Audio sensitivity set to: {RMS_SENSITIVITY}")
            return jsonify({'status': 'success', 'sensitivity': RMS_SENSITIVITY})
        else:
            return jsonify({'status': 'error', 'message': 'Sensitivity must be between 0 and 1'}), 400
    except Exception as e:
        print(f"ERROR: set_audio_sensitivity: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/generate_qr_code')
def generate_qr_code():
    app_host = request.host_url
    qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_L, box_size=10, border=4)
    qr.add_data(app_host)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white").convert('RGB')
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return jsonify({'qr_image': f'data:image/png;base64,{img_str}', 'url': app_host})

@app.route('/update_object_prompt', methods=['POST'])
def update_object_prompt():
    try:
        data = request.get_json()
        prompt = data.get('prompt').strip()
        if prompt:
            with mode_lock:
                if detection_mode == "object":
                    while not yolo_prompt_queue.empty():
                        yolo_prompt_queue.get_nowait()
                    yolo_prompt_queue.put(prompt)
                    return jsonify({'status': 'success', 'prompt': prompt})
                else:
                    return jsonify({'status': 'error', 'message': 'Not in object detection mode'}), 400
        else:
            return jsonify({'status': 'error', 'message': 'Prompt cannot be empty'}), 400
    except Exception as e:
        print(f"ERROR: update_object_prompt: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/set_detection_mode', methods=['POST'])
def set_detection_mode():
    global detection_mode, selected_face_id_for_arduino, tracked_faces, current_faces_info, audio_tracking_active, force_selection
    try:
        data = request.get_json()
        mode = data.get('mode')
        if mode in ["off", "face", "object"]:
            with mode_lock:
                with face_tracking_lock:
                    detection_mode = mode
                    if mode != "face":
                        tracked_faces.clear()
                        current_faces_info = []
                        selected_face_id_for_arduino = -1
                        force_selection = False
                        if audio_tracking_active:
                            stop_audio_stream()
                            audio_tracking_active = False
                    if mode == "off":
                        yolo_last_boxes = []
                        vision_and_eyes_active = False
                        audio_tracking_active = False
                    elif mode == "face":
                        selected_face_id_for_arduino = -1
                        force_selection = auto_mode != "manual"
                        print(f"Cleared tracked faces on switch to face mode, force_selection={force_selection}")
                    print(f"Detection mode set to: {mode}")
                    return jsonify({'status': 'success', 'mode': detection_mode})
        else:
            return jsonify({'status': 'error', 'message': 'Invalid detection mode'}), 400
    except Exception as e:
        print(f"ERROR: set_detection_mode: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
    except Exception as e:
        print(f"ERROR: Flask app failed: {e}")
    finally:
        stop_audio_stream()
        if p:
            p.terminate()
        if camera and camera.isOpened():
            camera.release()
            print("Camera released.")
        if ser and ser.is_open:
            ser.close()
            print("Serial port closed.")
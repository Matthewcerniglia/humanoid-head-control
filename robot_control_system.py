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

app = Flask(__name__)

# --- Camera Configuration ---
camera = 1
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_X_HALF_RES = CAMERA_WIDTH / 2.0
CAMERA_Y_HALF_RES = CAMERA_HEIGHT / 2.0

# --- Audio Configuration ---
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
AUDIO_THRESHOLD = 35
MIN_SERVO_ANGLE = 70  # Mouth closed
MAX_SERVO_ANGLE = 110  # Mouth open
RMS_MIN_INPUT = 50
RMS_MAX_INPUT = 500


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
    target_port = '/dev/cu.usbmodemF412FA6370882'  # Adjust as needed
    try:
        ser = serial.Serial(target_port, 115200, timeout=1)
        time.sleep(2)  # Allow Arduino to reset
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
current_mouth_angle = MIN_SERVO_ANGLE


# Find BlackHole audio device
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
IOU_THRESHOLD = 0.2
TRACKING_TIMEOUT_S = 2.0
face_tracking_lock = threading.Lock()
current_faces_info = []
selected_face_id_for_arduino = -1
last_known_x_dist = 0.0
last_known_y_dist = 0.0
last_detection_time = time.time()
detection_timeout_s = 0.5
vision_and_eyes_active = True
audio_tracking_active = False
last_system_active_state = False
smoothing_alpha = 0.2
current_rms = 0.0


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    return interArea / float(boxAArea + boxBArea - interArea)


def audio_callback(in_data, frame_count, time_info, status):
    global current_mouth_angle, current_rms

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

    # Map RMS to servo angle
    clamped_rms = np.clip(current_rms, RMS_MIN_INPUT, RMS_MAX_INPUT)
    target_angle = np.interp(clamped_rms, [RMS_MIN_INPUT, RMS_MAX_INPUT], [MIN_SERVO_ANGLE, MAX_SERVO_ANGLE])
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


def detect_track_and_draw_face(vid, draw_boxes_enabled):
    global next_face_id, tracked_faces, selected_face_id_for_arduino
    global last_known_x_dist, last_known_y_dist, last_detection_time
    global current_faces_info, vision_and_eyes_active
    screen_center = (CAMERA_WIDTH // 2, CAMERA_HEIGHT // 2)

    if not vision_and_eyes_active:
        current_faces_info = []
        return vid

    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    current_detections = face_classifier.detectMultiScale(
        gray_image,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(30, 30)
    )

    with face_tracking_lock:
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
                next_face_id += 1

        stale_ids = [fid for fid, data in tracked_faces.items() if time.time() - data['last_seen'] > TRACKING_TIMEOUT_S]
        for fid in stale_ids:
            del tracked_faces[fid]
            if selected_face_id_for_arduino == fid:
                selected_face_id_for_arduino = -1

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
                if selected_face_id_for_arduino == -1 or face_id == selected_face_id_for_arduino:
                    color = (0, 255, 0) if face_id == selected_face_id_for_arduino else (255, 0, 0)
                    cv2.rectangle(vid, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), color, 2)
                    cv2.putText(vid, f"Face {idx + 1}", (box[0], box[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        x_distance_arduino = 0.0
        y_distance_arduino = 0.0
        if selected_face_id_for_arduino != -1 and target_face_for_arduino_box is not None:
            tx, ty, tw, th = target_face_for_arduino_box
            face_center = (tx + tw // 2, ty + th // 2)
            x_distance_arduino = face_center[0] - screen_center[0]
            y_distance_arduino = face_center[1] - screen_center[1]
            last_known_x_dist = x_distance_arduino
            last_known_y_dist = y_distance_arduino
            last_detection_time = time.time()
        elif selected_face_id_for_arduino != -1 and (time.time() - last_detection_time) < detection_timeout_s:
            x_distance_arduino = last_known_x_dist
            y_distance_arduino = last_known_y_dist
        else:
            if not found_selected_face:
                selected_face_id_for_arduino = -1
            x_distance_arduino = 0.0
            y_distance_arduino = 0.0

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

        frame = detect_track_and_draw_face(frame, draw_boxes_enabled)

        # Add status overlay
        status_text = f"Vision: {'ON' if vision_and_eyes_active else 'OFF'} | Audio: {'ON' if audio_tracking_active else 'OFF'}"
        if audio_tracking_active:
            status_text += f" | RMS: {current_rms:.1f} | Mouth: {current_mouth_angle}°"

        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Send data to Arduino
        if ser and ser.is_open:
            try:
                system_active = vision_and_eyes_active or audio_tracking_active

                # Send system state change if needed
                if system_active != last_system_active_state:
                    command = "SYSTEM_ON\n" if system_active else "SYSTEM_OFF\n"
                    ser.write(command.encode('utf-8'))
                    print(f"Sent to Arduino: {command.strip()}")
                    last_system_active_state = system_active

                if system_active:
                    # Send combined data: vision + audio
                    num_faces = 1 if (vision_and_eyes_active and selected_face_id_for_arduino != -1) else 0
                    eye_x = last_known_x_dist if vision_and_eyes_active else 0.0
                    eye_y = last_known_y_dist if vision_and_eyes_active else 0.0
                    mouth_angle = current_mouth_angle if audio_tracking_active else MIN_SERVO_ANGLE

                    # Format: COMBINED,num_faces,eye_x,eye_y,mouth_angle,smoothing_alpha
                    command = f"COMBINED,{num_faces},{eye_x:.1f},{eye_y:.1f},{mouth_angle},{smoothing_alpha:.2f}\n"
                    ser.write(command.encode('utf-8'))

                # Read Arduino responses
                if ser.in_waiting > 0:
                    response = ser.readline().decode('utf-8').strip()
                    if response:  # Only print non-empty responses
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
        return jsonify({
            'faces': current_faces_info,
            'selected_face_id': selected_face_id_for_arduino,
            'vision_active': vision_and_eyes_active,
            'audio_active': audio_tracking_active,
            'current_rms': float(current_rms),
            'mouth_angle': int(current_mouth_angle)
        })


@app.route('/set_selected_face', methods=['POST'])
def set_selected_face():
    global selected_face_id_for_arduino
    try:
        data = request.get_json()
        face_id = int(data.get('face_id'))
        with face_tracking_lock:
            if face_id == -1 or face_id in tracked_faces:
                selected_face_id_for_arduino = face_id
                return jsonify({'status': 'success', 'selected_id': selected_face_id_for_arduino})
            else:
                return jsonify({'status': 'error', 'message': 'Invalid face ID'}), 400
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
                # Start audio tracking
                if start_audio_stream():
                    audio_tracking_active = True
                    print("Audio tracking started.")
                else:
                    return jsonify({'status': 'error', 'message': 'Failed to start audio stream'}), 500
            elif not new_state and audio_tracking_active:
                # Stop audio tracking
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


@app.route('/generate_qr_code')
def generate_qr_code():
    app_host = request.host_url
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(app_host)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white").convert('RGB')
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return jsonify({'qr_image': f'data:image/png;base64,{img_str}', 'url': app_host})


if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
    except Exception as e:
        print(f"ERROR: Flask app failed: {e}")
    finally:
        # Cleanup
        stop_audio_stream()
        if p:
            p.terminate()
        if camera and camera.isOpened():
            camera.release()
            print("Camera released.")
        if ser and ser.is_open:
            ser.close()
            print("Serial port closed.")
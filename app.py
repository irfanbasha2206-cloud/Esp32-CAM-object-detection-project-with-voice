from flask import Flask, Response, render_template_string
import cv2
import urllib.request
import numpy as np
import multiprocessing
import threading
import time
import speech_recognition as sr
from ultralytics import YOLO

app = Flask(__name__)
# Shared memory dictionary (Now works perfectly with Threads)
app_state = {"latest_frame": None, "detected_objects_data": []}

# ... [Imports and Setup remains same as previous] ...

# ── 1. MODIFIED VOICE LISTENER (Interactive Mode Only) ──
def listen_voice_command(speech_q):
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    print("\n[Jarvis Ear] Waiting for trigger word 'Jarvis'...")
    
    while True:
        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            try:
                audio = recognizer.listen(source, timeout=None, phrase_time_limit=3)
                command = recognizer.recognize_google(audio).lower()
                
                # ONLY INTERACT IF TRIGGER WORD IS USED
                if "jarvis" in command:
                    print(f"[User Triggered]: {command}")
                    objects = app_state.get("detected_objects_data", [])
                    
                    if "what" in command or "see" in command:
                        if objects:
                            labels = [obj['name'] for obj in objects]
                            speech_q.put(f"Sir, I currently see {', '.join(labels)} in front of you.")
                        else:
                            speech_q.put("I do not see any objects in your path right now.")
            except:
                pass

# ── 2. MODIFIED CAMERA FEED (Silent Safety Mode Only) ──
def process_camera_feed():
    # ... [Same stream setup] ...
    
    # SILENT MODE: ONLY SPEAK IF DANGER DETECTED
    while True:
        # ... [Same frame processing logic] ...
        
        if is_too_close and is_centered:
            danger_detected = True
            # This is the ONLY auto-speak condition
            danger_msg = f"Alert! {label} is extremely close. Please move."
        

# ── 1. Text-To-Speech Worker (MUST BE A PROCESS ON WINDOWS) ──
def speech_worker(queue):
    import pyttsx3
    while True:
        text = queue.get()
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 160)
            engine.say(text)
            engine.runAndWait()
            del engine
        except Exception as e:
            print(f"Audio Error: {e}")

speech_queue = multiprocessing.Queue()

# ── 2. Speech-To-Text Worker (THREAD) ──
def listen_voice_command(speech_q):
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    print("\n[Jarvis Ear] Microphone listening system started...")
    
    while True:
        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            try:
                audio = recognizer.listen(source, timeout=None, phrase_time_limit=3)
                command = recognizer.recognize_google(audio).lower()
                print(f"[User Said]: {command}")
                
                if "what" in command or "see" in command or "front" in command or "jarvis" in command:
                    # Now the Mic thread can correctly read what the Camera thread sees!
                    objects = app_state.get("detected_objects_data", [])
                    if objects:
                        labels = [obj['name'] for obj in objects]
                        speech_q.put(f"In front of you, I see {', '.join(labels)}")
                    else:
                        speech_q.put("The path ahead looks completely clear.")
            except sr.UnknownValueError:
                pass
            except Exception as e:
                pass # Keeps loop alive silently

# ── 3. AI Camera Feed (THREAD) ──
model = YOLO("yolov8n.pt")
ESP32_URL = "http://10.11.69.128/stream" # Double check your IP!

def process_camera_feed():
    print(f"[AI Brain] Opening stream: {ESP32_URL}")
    try:
        stream = urllib.request.urlopen(ESP32_URL, timeout=5)
    except Exception as e:
        print(f"Connection Failed: {e}")
        return

    bytes_data = b''
    safety_cooldown = 0

    while True:
        try:
            bytes_data += stream.read(2048)
            a = bytes_data.find(b'\xff\xd8')
            b = bytes_data.find(b'\xff\xd9')
            
            if a != -1 and b != -1:
                jpg = bytes_data[a:b+2]
                bytes_data = bytes_data[b+2:]
                
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    h, w, _ = frame.shape
                    results = model(frame, conf=0.45, verbose=False)
                    
                    current_detections = []
                    danger_detected = False
                    danger_msg = ""

                    for result in results:
                        for box in result.boxes:
                            cls_id = int(box.cls[0])
                            label = model.names[cls_id]
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            
                            obj_width = x2 - x1
                            obj_height = y2 - y1
                            obj_center_x = x1 + (obj_width // 2)
                            
                            is_too_close = (obj_height / h) > 0.60
                            is_centered = (w * 0.3) < obj_center_x < (w * 0.7)

                            current_detections.append({"name": label, "box": [x1, y1, x2, y2]})
                            
                            if is_too_close and is_centered:
                                danger_detected = True
                                danger_msg = f"Watch out! A {label} is directly close ahead!"
                            
                            color = (0, 0, 255) if is_too_close else (0, 255, 0)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(frame, f"{label} (Alert)", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # Update shared state
                    app_state["detected_objects_data"] = current_detections
                    app_state["latest_frame"] = frame
                    
                    if danger_detected and safety_cooldown > 20:
                        speech_queue.put(danger_msg)
                        safety_cooldown = 0
                    
                    safety_cooldown += 1
                    
        except Exception as e:
            print(f"Processing Failure: {e}")
            break

# ── 4. Web Stream Wrapper ──
def generate_mjpeg():
    while True:
        frame = app_state.get("latest_frame")
        if frame is not None:
            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        # THIS 0.05s DELAY FIXES THE BLACK SCREEN INFINITE LOOP CRASH!
        time.sleep(0.05) 

@app.route('/')
def index():
    # Added some text so the screen isn't totally blank before the video loads
    return render_template_string("""
    <html>
        <body style="background:#111; color:#00ff00; text-align:center; font-family:Arial; padding-top:20px;">
            <h2>Jarvis Smart Stick Feed</h2>
            <img src="/video" width="640" style="border: 3px solid #00ff00; border-radius: 8px;">
        </body>
    </html>
    """)

@app.route('/video')
def video():
    return Response(generate_mjpeg(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # 1. Start Speaker (Must be a separate Process)
    p_speech = multiprocessing.Process(target=speech_worker, args=(speech_queue,))
    p_speech.daemon = True
    p_speech.start()

    # 2. Start AI Camera (Thread - Shares Memory)
    t_cam = threading.Thread(target=process_camera_feed)
    t_cam.daemon = True
    t_cam.start()

    # 3. Start Mic Listener (Thread - Shares Memory)
    t_mic = threading.Thread(target=listen_voice_command, args=(speech_queue,))
    t_mic.daemon = True
    t_mic.start()

    # Threaded=True prevents Flask from blocking itself
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
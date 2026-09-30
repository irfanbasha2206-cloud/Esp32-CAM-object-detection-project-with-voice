"""
JARVIS SMART STICK — Assistive Vision System
==============================================
Two operating modes:

  INTERACTIVE MODE  -> Jarvis stays quiet and waits. Say "Jarvis" plus a
                        question and it answers conversationally, based on
                        what the camera can currently see (counts, left /
                        center / right position, how close things are).

  DETECTIVE MODE     -> Jarvis needs no trigger word. It calls out things as
                        soon as they enter view ("A person just appeared on
                        your right"), and gives a short recap if the scene
                        has been unchanged for a while.

Safety alerts (something big and dead-ahead) ALWAYS speak immediately in
either mode — safety overrides everything else.
"""

from flask import Flask, Response, render_template_string, jsonify, request
import cv2
import re
import urllib.request
import numpy as np
import multiprocessing
import threading
import time
import datetime
import speech_recognition as sr
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
ESP32_URL = "http://10.11.69.128/stream"   
MODEL_PATH = "yolov8n.pt"
DANGER_HEIGHT_RATIO = 0.60                 
DANGER_COOLDOWN_SEC = 6
NEW_OBJECT_COOLDOWN_SEC = 5                
DETECTIVE_RECAP_INTERVAL = 15              
MAX_LOG_LINES = 60

app = Flask(__name__)

# ─────────────────────────────────────────────────────────────
# SHARED STATE
# ─────────────────────────────────────────────────────────────
app_state = {
    "mode": "interactive",
    "language": "en",              
    "latest_frame": None,
    "detected_objects_data": [],   
    "listening": False,
    "speaking": False,
    "log": [],                     
    "conversation": [],            
    "started_at": time.time(),
}
state_lock = threading.Lock()
speech_queue = multiprocessing.Queue()


def log_event(text, kind="system"):
    entry = {"time": datetime.datetime.now().strftime("%H:%M:%S"), "text": text, "type": kind}
    with state_lock:
        app_state["log"].append(entry)
        if len(app_state["log"]) > MAX_LOG_LINES:
            app_state["log"] = app_state["log"][-MAX_LOG_LINES:]


def log_conversation(role, text):
    with state_lock:
        app_state["conversation"].append({
            "role": role, "text": text,
            "time": datetime.datetime.now().strftime("%H:%M:%S"),
        })
        if len(app_state["conversation"]) > MAX_LOG_LINES:
            app_state["conversation"] = app_state["conversation"][-MAX_LOG_LINES:]


def get_language():
    with state_lock:
        return app_state["language"]


def speak(text, kind="system", as_conversation=False, lang=None):
    if lang is None:
        lang = get_language()
    speech_queue.put((text, lang))
    if as_conversation:
        log_conversation("jarvis", text)
    else:
        log_event(text, kind)


# ─────────────────────────────────────────────────────────────
# SCENE DESCRIPTION HELPERS
# ─────────────────────────────────────────────────────────────
POSITION_PHRASE = {"left": "on your left", "center": "straight ahead", "right": "on your right"}


def _plural(name, count):
    if count == 1:
        return name
    return name + "es" if name.endswith(("s", "x", "ch", "sh")) else name + "s"


def summarize_objects(objects):
    groups = {}
    for o in objects:
        g = groups.setdefault(o["name"], {"count": 0, "positions": set(), "danger": False})
        g["count"] += 1
        g["positions"].add(o["position"])
        g["danger"] = g["danger"] or o["danger"]
    return groups


def build_scene_sentence(objects):
    if not objects:
        return "The path ahead looks clear. I don't see anything right now."

    groups = summarize_objects(objects)
    clauses = []
    for name, g in groups.items():
        pos_text = " and ".join(POSITION_PHRASE[p] for p in sorted(g["positions"]))
        if g["count"] == 1:
            clause = f"a {name} {pos_text}"
        else:
            clause = f"{g['count']} {_plural(name, g['count'])} {pos_text}"
        if g["danger"]:
            clause += " — quite close, be careful"
        clauses.append(clause)

    if len(clauses) == 1:
        body = clauses[0]
    else:
        body = ", ".join(clauses[:-1]) + " and " + clauses[-1]
    return f"I can see {body}."


def build_change_sentence(new_labels_positions):
    parts = []
    for name, pos, danger in new_labels_positions:
        p = f"a {name} just appeared {POSITION_PHRASE[pos]}"
        if danger:
            p += " — it's close, watch out"
        parts.append(p)
    return ". ".join(parts) + "."


OBJECT_TA = {
    "person": "நபர்", "car": "கார்", "bicycle": "சைக்கிள்", "motorcycle": "மோட்டார் சைக்கிள்",
    "bus": "பேருந்து", "truck": "லாரி", "chair": "நாற்காலி", "table": "மேசை", "dog": "நாய்",
    "cat": "பூனை", "bottle": "பாட்டில்", "laptop": "லேப்டாப்", "cell phone": "மொபைல் போன்",
    "backpack": "பை", "book": "புத்தகம்", "tv": "டிவி", "sofa": "சோஃபா", "bed": "படுக்கை",
    "bench": "பெஞ்ச்", "umbrella": "குடை", "handbag": "கைப்பை", "suitcase": "பயணப்பை",
    "traffic light": "சிக்னல் லைட்", "stop sign": "நிறுத்து சைன்போர்டு", "bowl": "கிண்ணம்",
    "cup": "கப்", "potted plant": "செடி", "refrigerator": "குளிர்சாதனப் பெட்டி", "bird": "பறவை",
}
POSITION_PHRASE_TA = {"left": "உங்க இடது பக்கம்", "center": "நேரா முன்னாடி", "right": "உங்க வலது பக்கம்"}


def _ta_name(name):
    return OBJECT_TA.get(name, name)


def build_scene_sentence_ta(objects):
    if not objects:
        return "முன்னாடி பாதை கிளியரா இருக்கு. இப்போ ஒன்னும் தெரியல."

    groups = summarize_objects(objects)
    clauses = []
    for name, g in groups.items():
        pos_text = " மற்றும் ".join(POSITION_PHRASE_TA[p] for p in sorted(g["positions"]))
        tname = _ta_name(name)
        clause = f"{pos_text} {'ஒரு ' if g['count'] == 1 else str(g['count']) + ' '}{tname}"
        if g["danger"]:
            clause += " — ரொம்ப கிட்ட இருக்கு, கவனமா இருங்க"
        clauses.append(clause)
    return "எனக்கு தெரியுது, " + ", ".join(clauses) + "."


def build_change_sentence_ta(new_labels_positions):
    parts = []
    for name, pos, danger in new_labels_positions:
        p = f"{POSITION_PHRASE_TA[pos]} ஒரு {_ta_name(name)} இப்போதான் தெரியுது"
        if danger:
            p += " — கிட்ட இருக்கு, கவனம்"
        parts.append(p)
    return ". ".join(parts) + "."


def get_scene_sentence(objects, language):
    return build_scene_sentence_ta(objects) if language == "ta" else build_scene_sentence(objects)


def get_change_sentence(items, language):
    return build_change_sentence_ta(items) if language == "ta" else build_change_sentence(items)


def get_danger_message(label, language):
    if language == "ta":
        return f"கவனம்! ஒரு {_ta_name(label)} நேரா முன்னாடி ரொம்ப கிட்ட இருக்கு."
    return f"Watch out! A {label} is directly ahead and very close."


MODE_MSG = {
    ("detective", "en"): "Detective mode engaged. I'll call things out as I spot them.",
    ("detective", "ta"): "டிடெக்டிவ் மோட் ஆன் ஆச்சு. நான் என்ன பார்க்கிறேன்னு உடனே சொல்றேன்.",
    ("interactive", "en"): "Interactive mode engaged. Say Jarvis, then ask me anything.",
    ("interactive", "ta"): "இன்டராக்டிவ் மோட் ஆன் ஆச்சு. ஜார்விஸ் சொல்லிட்டு எதுவும் கேளுங்க.",
}
LANGUAGE_MSG = {
    "ta": "சரி, இப்போ நான் தமிழ்ல பேசுவேன்.",
    "en": "Okay, I'll speak in English now.",
}


# ─────────────────────────────────────────────────────────────
# VOICE COMMAND ROUTING
# ─────────────────────────────────────────────────────────────
TA_TO_OBJECT = {v: k for k, v in OBJECT_TA.items()}


def handle_interactive_command(command, language="en"):
    with state_lock:
        objects = list(app_state["detected_objects_data"])
    if language == "ta":
        return _handle_command_ta(command, objects)
    return _handle_command_en(command, objects)


def _handle_command_en(command, objects):
    if re.search(r"\b(hi|hello|hey|you there|you awake)\b", command):
        return "Yes, I'm here and watching the path ahead."

    if re.search(r"\bwho are you\b|\bwhat are you\b", command):
        return "I'm Jarvis, your smart stick's eyes. Ask me what I see."

    side_match = re.search(r"\b(left|right)\b", command)
    if side_match and re.search(r"\b(what|anything|see)\b", command):
        side = side_match.group(1)
        side_objs = [o for o in objects if o["position"] == side]
        if side_objs:
            names = sorted(set(o["name"] for o in side_objs))
            return f"On your {side}, I see {', '.join(names)}."
        return f"Nothing on your {side} right now."

    count_match = re.search(r"how many (\w+)", command)
    if count_match:
        target = count_match.group(1).rstrip("s")
        matches = [o for o in objects if o["name"].startswith(target)]
        if matches:
            return f"I count {len(matches)} {_plural(target, len(matches))} in view."
        return f"I don't see any {target} right now."

    if re.search(r"\bis (there|anyone|anybody)\b|\banyone (there|around)\b", command):
        if objects:
            names = sorted(set(o["name"] for o in objects))
            return f"Yes — I can see {', '.join(names)}."
        return "No, the area looks empty right now."

    if re.search(r"\b(safe|clear|obstacle)\b", command):
        danger_objs = [o for o in objects if o["danger"]]
        if danger_objs:
            names = sorted(set(o["name"] for o in danger_objs))
            return f"Careful — {', '.join(names)} very close ahead."
        return "It looks clear. No close obstacles detected."

    if re.search(r"\b(what|see|front|ahead|describe|around|scene)\b", command):
        return build_scene_sentence(objects)

    return "I heard you, but I'm not sure what you're asking. Try 'what do you see' or 'is it safe'."


def _handle_command_ta(command, objects):
    if re.search(r"(வணக்கம்|இருக்கீங்களா|இருக்கியா)", command):
        return "ஆமா, நான் இருக்கேன். முன்னாடி பாதையை கவனிச்சிட்டு இருக்கேன்."

    if re.search(r"(யாரு நீ|நீ யாரு|என்ன நீ)", command):
        return "நான் ஜார்விஸ், உங்க ஸ்மார்ட் ஸ்டிக்-ன் கண்கள். நான் என்ன பாக்குறேன்னு கேளுங்க."

    side = "left" if "இடது" in command else "right" if "வலது" in command else None
    if side and re.search(r"(என்ன|தெரியுது|இருக்கு)", command):
        side_word = "இடது" if side == "left" else "வலது"
        side_objs = [o for o in objects if o["position"] == side]
        if side_objs:
            names = sorted(set(o["name"] for o in side_objs))
            return f"உங்க {side_word} பக்கம், {', '.join(_ta_name(n) for n in names)} தெரியுது."
        return f"உங்க {side_word} பக்கம் இப்போ ஒன்னும் இல்ல."

    count_match = re.search(r"எத்தனை (\S+)", command)
    if count_match:
        target_ta = count_match.group(1)
        target_en = TA_TO_OBJECT.get(target_ta, target_ta)
        matches = [o for o in objects if o["name"].startswith(target_en)]
        if matches:
            return f"நான் {len(matches)} {target_ta} பாக்குறேன்."
        return f"{target_ta} எதும் தெரியல."

    if re.search(r"(யாராவது இருக்கா|எவனாவது இருக்கா|\bஇருக்கா\b)", command):
        if objects:
            names = sorted(set(o["name"] for o in objects))
            return f"ஆமா — {', '.join(_ta_name(n) for n in names)} தெரியுது."
        return "இல்ல, இந்த இடம் காலியா இருக்கு."

    if re.search(r"(பாதுகாப்பு|பாதுகாப்பா|கிளியரா)", command):
        danger_objs = [o for o in objects if o["danger"]]
        if danger_objs:
            names = sorted(set(o["name"] for o in danger_objs))
            return f"கவனம் — {', '.join(_ta_name(n) for n in names)} ரொம்ப கிட்ட இருக்கு முன்னாடி."
        return "பாதை கிளியரா இருக்கு. கிட்ட எந்த பொருளும் இல்ல."

    if re.search(r"(என்ன|தெரியுது|முன்னாடி|பாக்கிற|சுத்து)", command):
        return build_scene_sentence_ta(objects)

    return "நான் கேட்டேன், ஆனா என்ன கேக்குறீங்கனு எனக்கு புரியல. 'என்ன தெரியுது' அல்லது 'பாதுகாப்பா' கேளுங்க."


# ─────────────────────────────────────────────────────────────
# WORKER 1 — Text-to-Speech (Pygame Engine Fix)
# ─────────────────────────────────────────────────────────────
def _speak_pyttsx3(pyttsx3_module, text):
    engine = pyttsx3_module.init()
    engine.setProperty('rate', 165)
    engine.say(text)
    engine.runAndWait()
    del engine


def _speak_gtts(text, lang_code="ta"):
    import tempfile
    import os
    from gtts import gTTS
    from pygame import mixer

    tmp_path = None
    try:
        tts = gTTS(text=text, lang=lang_code)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            tmp_path = f.name
        tts.save(tmp_path)
        
        # Fixed execution pipeline via Pygame Mixer to bypass OS file-locks
        mixer.init()
        mixer.music.load(tmp_path)
        mixer.music.play()
        while mixer.music.get_busy():
            time.sleep(0.05)
        mixer.music.unload()
        mixer.quit()
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def speech_worker(queue):
    import pyttsx3
    while True:
        text, lang = queue.get()
        with state_lock:
            app_state["speaking"] = True
        try:
            if lang == "ta":
                try:
                    _speak_gtts(text, "ta")
                except Exception as e:
                    print(f"[gTTS engine error, falling back to OS Voice Engine] {e}")
                    _speak_pyttsx3(pyttsx3, text)
            else:
                _speak_pyttsx3(pyttsx3, text)
        except Exception as e:
            print(f"[Speech Worker Drop Error] {e}")
        finally:
            with state_lock:
                app_state["speaking"] = False


# ─────────────────────────────────────────────────────────────
# WORKER 2 — Speech-to-Text Listener (Echo Filter Fix)
# ─────────────────────────────────────────────────────────────
def _has_tamil_script(text):
    return bool(text) and any('\u0B80' <= ch <= '\u0BFF' for ch in text)


def recognize_bilingual(recognizer, audio):
    text_en, text_ta = None, None
    try:
        text_en = recognizer.recognize_google(audio, language="en-IN")
    except Exception:
        pass
    try:
        text_ta = recognizer.recognize_google(audio, language="ta-IN")
    except Exception:
        pass

    if text_ta and _has_tamil_script(text_ta):
        # Explicit context filtering to drop phonetic false positives generated by English inputs
        tamil_keywords = ["வணக்கம்", "இருக்க", "யாரு", "நீ", "என்ன", "இடது", "வலது", "எத்தனை", "யாராவது", "பாதுகாப்", "கிளியர்", "முன்னாடி", "பார்க்க", "சுத்து", "ஜார்விஸ்"]
        if any(kw in text_ta for kw in tamil_keywords):
            return text_ta, "ta"
        
    if text_en:
        return text_en.lower(), "en"
    return None, None


def listen_voice_command():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    log_event("Microphone online. Listening for Tamil or English commands.", "system")

    while True:
        with state_lock:
            app_state["listening"] = True
        try:
            with mic as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = recognizer.listen(source, timeout=None, phrase_time_limit=5)

            with state_lock:
                app_state["listening"] = False

            # Anti-Feedback Loop Lockout Check
            with state_lock:
                was_speaking = app_state["speaking"]
            if was_speaking:
                continue

            command, detected_lang = recognize_bilingual(recognizer, audio)
            if command is None:
                continue
            print(f"[Heard, {detected_lang}]: {command}")

            mode_words = command
            if "detective" in mode_words and ("mode" in mode_words or "switch" in mode_words):
                with state_lock:
                    app_state["mode"] = "detective"
                    app_state["language"] = detected_lang
                speak(MODE_MSG[("detective", detected_lang)], "system", lang=detected_lang)
                continue

            if "interactive" in mode_words and ("mode" in mode_words or "switch" in mode_words):
                with state_lock:
                    app_state["mode"] = "interactive"
                    app_state["language"] = detected_lang
                speak(MODE_MSG[("interactive", detected_lang)], "system", lang=detected_lang)
                continue

            triggered = "jarvis" in command or "ஜார்விஸ்" in command
            if triggered:
                log_conversation("user", command)
                with state_lock:
                    app_state["language"] = detected_lang   
                reply = handle_interactive_command(command, detected_lang)
                speak(reply, as_conversation=True, lang=detected_lang)

        except sr.WaitTimeoutError:
            pass
        except Exception as e:
            print(f"[Mic Module Interruption] {e}")
            time.sleep(1)
        finally:
            with state_lock:
                app_state["listening"] = False


# ─────────────────────────────────────────────────────────────
# WORKER 3 — Camera Engine + Computer Vision
# ─────────────────────────────────────────────────────────────
model = YOLO(MODEL_PATH)


def _position_of(center_x, w):
    ratio = center_x / w
    if ratio < 0.4:
        return "left"
    if ratio > 0.6:
        return "right"
    return "center"


def process_camera_feed():
    print(f"[Camera] Connecting to {ESP32_URL}")
    try:
        stream = urllib.request.urlopen(ESP32_URL, timeout=5)
    except Exception as e:
        log_event(f"Could not connect to ESP32-CAM: {e}", "system")
        print(f"[Camera] Connection failed: {e}")
        return

    log_event("Camera feed online.", "system")

    bytes_data = b''
    last_danger_time = 0.0
    last_recap_time = 0.0
    last_seen_labels = set()          
    last_announced_at = {}            

    while True:
        try:
            bytes_data += stream.read(2048)
            a = bytes_data.find(b'\xff\xd8')
            b = bytes_data.find(b'\xff\xd9')
            if a == -1 or b == -1:
                continue

            jpg = bytes_data[a:b + 2]
            bytes_data = bytes_data[b + 2:]

            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue

            h, w, _ = frame.shape
            results = model(frame, conf=0.45, verbose=False)

            current_detections = []
            danger_detected = False
            danger_label = None

            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    label = model.names[cls_id]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    obj_height = y2 - y1
                    center_x = x1 + (x2 - x1) // 2
                    position = _position_of(center_x, w)

                    is_too_close = (obj_height / h) > DANGER_HEIGHT_RATIO
                    is_danger = is_too_close and position == "center"

                    current_detections.append({
                        "name": label, "box": [x1, y1, x2, y2],
                        "danger": is_danger, "position": position,
                    })

                    if is_danger:
                        danger_detected, danger_label = True, label

                    color = (60, 60, 235) if is_danger else (90, 200, 90)
                    tag = f"{label} (CLOSE)" if is_danger else label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, tag, (x1, max(0, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            with state_lock:
                app_state["detected_objects_data"] = current_detections
                app_state["latest_frame"] = frame
                current_mode = app_state["mode"]

            now = time.time()
            current_labels = {d["name"] for d in current_detections}
            current_language = get_language()

            if danger_detected and (now - last_danger_time) > DANGER_COOLDOWN_SEC:
                speak(get_danger_message(danger_label, current_language), "danger", lang=current_language)
                last_danger_time = now

            elif current_mode == "detective":
                new_labels = current_labels - last_seen_labels
                to_announce = []
                for d in current_detections:
                    if d["name"] in new_labels:
                        last_time = last_announced_at.get(d["name"], 0)
                        if now - last_time > NEW_OBJECT_COOLDOWN_SEC:
                            to_announce.append((d["name"], d["position"], d["danger"]))
                            last_announced_at[d["name"]] = now

                if to_announce:
                    seen = set()
                    unique = []
                    for item in to_announce:
                        if item[0] not in seen:
                            unique.append(item)
                            seen.add(item[0])
                    speak(get_change_sentence(unique, current_language), "detective", lang=current_language)
                    last_recap_time = now
                elif current_detections and (now - last_recap_time) > DETECTIVE_RECAP_INTERVAL:
                    recap_prefix = "இன்னும் கண்காணிக்கிறேன்: " if current_language == "ta" else "Still tracking: "
                    speak(recap_prefix + get_scene_sentence(current_detections, current_language),
                          "detective", lang=current_language)
                    last_recap_time = now

            last_seen_labels = current_labels

        except Exception as e:
            print(f"[CV Frame Drop Exception] {e}")
            time.sleep(0.5)


# ─────────────────────────────────────────────────────────────
# FLASK WEB INTERFACE ENDPOINTS
# ─────────────────────────────────────────────────────────────
def generate_mjpeg():
    while True:
        with state_lock:
            frame = app_state["latest_frame"]
        if frame is not None:
            ok, buffer = cv2.imencode('.jpg', frame)
            if ok:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.05)


@app.route('/video')
def video():
    return Response(generate_mjpeg(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/state')
def api_state():
    with state_lock:
        uptime = int(time.time() - app_state["started_at"])
        return jsonify({
            "mode": app_state["mode"],
            "language": app_state["language"],
            "listening": app_state["listening"],
            "speaking": app_state["speaking"],
            "objects": app_state["detected_objects_data"],
            "log": app_state["log"][-20:],
            "conversation": app_state["conversation"][-20:],
            "uptime": uptime,
        })


@app.route('/api/mode', methods=['POST'])
def api_mode():
    new_mode = request.json.get("mode") if request.is_json else None
    if new_mode not in ("interactive", "detective"):
        return jsonify({"ok": False, "error": "Invalid operation mode parameter"}), 400
    with state_lock:
        app_state["mode"] = new_mode
        current_lang = app_state["language"]
    speak(MODE_MSG[(new_mode, current_lang)], "system", lang=current_lang)
    return jsonify({"ok": True, "mode": new_mode})


@app.route('/api/language', methods=['POST'])
def api_language():
    new_lang = request.json.get("language") if request.is_json else None
    if new_lang not in ("en", "ta"):
        return jsonify({"ok": False, "error": "Language mapping standard unsupported"}), 400
    with state_lock:
        app_state["language"] = new_lang
    speak(LANGUAGE_MSG[new_lang], "system", lang=new_lang)
    return jsonify({"ok": True, "language": new_lang})


@app.route('/')
def index():
    return render_template_string(PAGE_HTML)


# ─────────────────────────────────────────────────────────────
# FRONTEND RENDERING DOM ENGINE (Realignment Applied)
# ─────────────────────────────────────────────────────────────
PAGE_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>JARVIS · Vision Console</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root{
    --bg:#0c0b0a;
    --panel:#151311;
    --panel-2:#1b1815;
    --line:rgba(255,182,72,0.14);
    --cream:#f4ecdb;
    --amber:#ffb648;
    --amber-dim:#a67833;
    --red:#e2453a;
    --teal:#49b6a8;
    --muted:#8a8175;
  }
  *{box-sizing:border-box;}
  body{
    margin:0; min-height:100vh; background:var(--bg); color:var(--cream);
    font-family:'Inter',sans-serif; padding:26px 26px 40px;
  }
  h1,h2{font-family:'Space Grotesk',sans-serif; margin:0;}
  .topbar{display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:14px; margin-bottom:22px;}
  .topbar h1{font-size:1.5rem; letter-spacing:0.3px;}
  .topbar h1 span{color:var(--amber);}
  .mode-pill{
    display:flex; gap:6px; background:var(--panel); border:1px solid var(--line);
    border-radius:999px; padding:4px;
  }
  .mode-pill button{
    font-family:'IBM Plex Mono',monospace; font-size:0.72rem; letter-spacing:1px; text-transform:uppercase;
    border:none; border-radius:999px; padding:9px 16px; background:transparent; color:var(--muted); cursor:pointer;
    transition:0.15s;
  }
  .mode-pill button.active-interactive{background:var(--teal); color:#08211d;}
  .mode-pill button.active-detective{background:var(--red); color:#2a0a08;}

  .grid{display:grid; grid-template-columns: 1.15fr 0.85fr; gap:20px;}
  @media(max-width:960px){ .grid{grid-template-columns:1fr;} }

  .panel{
    background:linear-gradient(180deg, rgba(255,255,255,0.015), transparent), var(--panel);
    border:1px solid var(--line); border-radius:14px; padding:18px;
  }
  .panel-fullwidth{
    margin-top:20px;
    background:linear-gradient(180deg, rgba(255,255,255,0.015), transparent), var(--panel);
    border:1px solid var(--line); border-radius:14px; padding:18px;
    width:100%;
  }
  .panel + .panel{margin-top:20px;}
  .panel h2, .panel-fullwidth h2{font-size:0.72rem; letter-spacing:2px; text-transform:uppercase; color:var(--amber-dim); margin-bottom:14px;}

  .sonar-wrap{position:relative; display:flex; justify-content:center; padding:6px 0 0;}
  .sonar-wrap svg{width:100%; max-width:420px; height:auto;}
  .sweep{ transform-origin:210px 230px; animation:sweep 3.2s linear infinite; }
  @keyframes sweep{ from{transform:rotate(0deg);} to{transform:rotate(-180deg);} }
  .ring{ fill:none; stroke:var(--line); stroke-width:1; }
  .blip{ transition: cx 0.4s ease, cy 0.4s ease; }
  .blip.danger circle{ animation:blipPulse 0.9s ease-in-out infinite; }
  @keyframes blipPulse{ 0%,100%{opacity:1;} 50%{opacity:0.35;} }
  .blip-label{ font-family:'IBM Plex Mono',monospace; font-size:9px; fill:var(--cream); }

  .status-strip{display:flex; justify-content:center; gap:22px; margin-top:10px; font-family:'IBM Plex Mono',monospace; font-size:0.7rem; color:var(--muted);}
  .status-strip .dot{width:7px; height:7px; border-radius:50%; display:inline-block; margin-right:6px; background:#3a352c;}
  .status-strip .on{background:var(--amber); box-shadow:0 0 6px var(--amber);}

  .lens{
    border-radius:16px; overflow:hidden; border:2px solid var(--amber-dim);
    position:relative; background:#000;
  }
  .lens img{width:100%; display:block;}
  .lens .rec{
    position:absolute; top:10px; left:12px; font-family:'IBM Plex Mono',monospace; font-size:0.68rem;
    background:rgba(0,0,0,0.55); padding:4px 8px; border-radius:6px; color:var(--red);
  }
  .lens .rec .dot{width:6px;height:6px;border-radius:50%;background:var(--red);display:inline-block;margin-right:5px;animation:blipPulse 1.2s infinite;}

  .ledger{display:flex; flex-wrap:wrap; gap:8px;}
  .chip{
    font-family:'IBM Plex Mono',monospace; font-size:0.74rem; padding:6px 11px; border-radius:8px;
    background:var(--panel-2); border:1px solid var(--line); color:var(--cream);
  }
  .chip .pos{color:var(--muted); margin-left:5px;}
  .chip.danger{border-color:var(--red); color:#ffb2ac;}
  .empty{color:var(--muted); font-family:'IBM Plex Mono',monospace; font-size:0.78rem;}

  .chat{height:230px; overflow-y:auto; display:flex; flex-direction:column; gap:10px; padding-right:4px;}
  .chat::-webkit-scrollbar{width:6px;}
  .chat::-webkit-scrollbar-thumb{background:var(--line); border-radius:4px;}
  .bubble{max-width:82%; padding:9px 13px; border-radius:12px; font-size:0.85rem; line-height:1.4;}
  .bubble.user{align-self:flex-end; background:var(--teal); color:#08211d; border-bottom-right-radius:3px;}
  .bubble.jarvis{align-self:flex-start; background:var(--panel-2); border:1px solid var(--line); border-bottom-left-radius:3px;}
  .bubble .who{display:block; font-family:'IBM Plex Mono',monospace; font-size:0.62rem; opacity:0.65; margin-bottom:3px; text-transform:uppercase; letter-spacing:1px;}

  /* Extended console spacing rules */
  .console{height:180px; overflow-y:auto; font-family:'IBM Plex Mono',monospace; font-size:0.82rem; line-height:1.6;}
  .console::-webkit-scrollbar{width:6px;}
  .console::-webkit-scrollbar-thumb{background:var(--line); border-radius:4px;}
  .console .row{margin-bottom:6px; display:flex; align-items:flex-start;}
  .console .t{color:var(--amber-dim); margin-right:14px; flex-shrink: 0;}
  .console .row.danger .m{color:#ff8f86; font-weight:600;}
  .console .row.detective .m{color:var(--cream);}
  .console .row.system .m{color:var(--muted); font-style:italic;}

  .uptime{margin-top:14px; text-align:right; font-family:'IBM Plex Mono',monospace; font-size:0.7rem; color:var(--muted);}
</style>
</head>
<body>

  <div class="topbar">
    <h1>JARVIS <span>· Vision Console</span></h1>
    <div style="display:flex; gap:12px; align-items:center; flex-wrap:wrap;">
      <div class="mode-pill">
        <button id="btnInteractive" onclick="setMode('interactive')">Interactive</button>
        <button id="btnDetective" onclick="setMode('detective')">Detective</button>
      </div>
      <div class="mode-pill" title="Auto-detected from your last question">
        <button id="btnLangEn" onclick="setLanguage('en')">EN</button>
        <button id="btnLangTa" onclick="setLanguage('ta')">தமிழ்</button>
      </div>
      <span id="autoLangNote" style="font-family:'IBM Plex Mono',monospace; font-size:0.68rem; color:var(--muted);">
        auto-detects Tamil / English per question
      </span>
    </div>
  </div>

  <div class="grid">
    <!-- LEFT COLUMN Layout Config -->
    <div>
      <!-- Camera Lens Frame: Positioned Upper Left -->
      <div class="panel">
        <h2>Camera Lens</h2>
        <div class="lens">
          <img src="/video" alt="ESP32-CAM live feed">
          <div class="rec"><span class="dot"></span>LIVE</div>
        </div>
        <div class="uptime" id="uptime">uptime 00:00:00</div>
      </div>

      <!-- Field Scan Radar: Shifted Down Directly Below Lens -->
      <div class="panel">
        <h2>Field Scan</h2>
        <div class="sonar-wrap">
          <svg viewBox="0 0 420 250">
            <path class="ring" d="M 20 230 A 190 190 0 0 1 400 230" />
            <path class="ring" d="M 75 230 A 135 135 0 0 1 345 230" />
            <path class="ring" d="M 130 230 A 80 80 0 0 1 310 230" />
            <line x1="20" y1="230" x2="400" y2="230" stroke="var(--line)" stroke-width="1"/>
            <g class="sweep">
              <line x1="210" y1="230" x2="210" y2="40" stroke="var(--amber)" stroke-width="2" opacity="0.55"/>
            </g>
            <g id="blips"></g>
          </svg>
        </div>
        <div class="status-strip">
          <span><span class="dot" id="dotMic"></span>Mic</span>
          <span><span class="dot" id="dotSpeak"></span>Speaking</span>
          <span><span class="dot on"></span>Camera</span>
        </div>
      </div>
    </div>

    <!-- RIGHT COLUMN Layout Config -->
    <div>
      <div class="panel">
        <h2>Objects In View</h2>
        <div class="ledger" id="ledger"><span class="empty">scanning…</span></div>
      </div>

      <div class="panel">
        <h2 id="convoTitle">Talk to Jarvis</h2>
        <div class="chat" id="chat"></div>
      </div>
    </div>
  </div>

  <!-- WIDE HORIZONTAL VIEW: Activity Logs decoupled from side columns -->
  <div class="panel-fullwidth">
    <h2>Activity Log</h2>
    <div class="console" id="console"></div>
  </div>

<script>
const POS_ANGLE = { left: 155, center: 90, right: 25 }; 
const cx = 210, cy = 230;

function polar(angleDeg, r){
  const rad = (angleDeg * Math.PI) / 180;
  return { x: cx - r * Math.cos(rad), y: cy - r * Math.sin(rad) };
}

function setMode(mode){
  fetch('/api/mode', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({mode})});
}

function setLanguage(language){
  fetch('/api/language', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({language})});
}

function fmtUptime(sec){
  const h = String(Math.floor(sec/3600)).padStart(2,'0');
  const m = String(Math.floor((sec%3600)/60)).padStart(2,'0');
  const s = String(sec%60).padStart(2,'0');
  return `${h}:${m}:${s}`;
}

async function poll(){
  try{
    const res = await fetch('/api/state');
    const data = await res.json();

    document.getElementById('btnInteractive').className = data.mode==='interactive' ? 'active-interactive' : '';
    document.getElementById('btnDetective').className = data.mode==='detective' ? 'active-detective' : '';
    document.getElementById('btnLangEn').className = data.language==='en' ? 'active-interactive' : '';
    document.getElementById('btnLangTa').className = data.language==='ta' ? 'active-detective' : '';
    document.getElementById('convoTitle').textContent = data.mode === 'interactive' ? 'Talk to Jarvis' : 'Jarvis Is Watching';

    document.getElementById('dotMic').className = 'dot' + (data.listening ? ' on' : '');
    document.getElementById('dotSpeak').className = 'dot' + (data.speaking ? ' on' : '');
    document.getElementById('uptime').textContent = 'uptime ' + fmtUptime(data.uptime);

    const blipsEl = document.getElementById('blips');
    const grouped = {};
    data.objects.forEach(o => {
      const key = o.position;
      grouped[key] = grouped[key] || [];
      grouped[key].push(o);
    });
    let svg = '';
    Object.entries(grouped).forEach(([pos, objs]) => {
      const angle = POS_ANGLE[pos] ?? 90;
      objs.forEach((o, i) => {
        const r = 60 + (i * 30) % 120;
        const {x, y} = polar(angle + (i * 6 - 6), r);
        const color = o.danger ? 'var(--red)' : 'var(--amber)';
        svg += `<g class="blip${o.danger ? ' danger' : ''}">
          <circle cx="${x}" cy="${y}" r="6" fill="${color}"></circle>
          <text class="blip-label" x="${x+9}" y="${y+3}">${o.name}</text>
        </g>`;
      });
    });
    blipsEl.innerHTML = svg;

    const ledgerEl = document.getElementById('ledger');
    if(data.objects.length === 0){
      ledgerEl.innerHTML = '<span class="empty">nothing in view right now…</span>';
    } else {
      const seen = {};
      data.objects.forEach(o => { seen[o.name+'|'+o.position] = o; });
      ledgerEl.innerHTML = Object.values(seen).map(o =>
        `<span class="chip${o.danger?' danger':''}">${o.name}<span class="pos">${o.position}</span></span>`
      ).join('');
    }

    const chatEl = document.getElementById('chat');
    chatEl.innerHTML = data.conversation.map(c =>
      `<div class="bubble ${c.role}"><span class="who">${c.role === 'user' ? 'You' : 'Jarvis'} · ${c.time}</span>${c.text}</div>`
    ).join('') || '<span class="empty">say "Jarvis, what do you see?" to start a conversation</span>';
    chatEl.scrollTop = chatEl.scrollHeight;

    const logEl = document.getElementById('console');
    logEl.innerHTML = data.log.map(l =>
      `<div class="row ${l.type}"><span class="t">[${l.time}]</span><span class="m">${l.text}</span></div>`
    ).join('');
    logEl.scrollTop = logEl.scrollHeight;

  } catch(e){ }
}

setInterval(poll, 1000);
poll();
</script>
</body>
</html>
"""


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    p_speech = multiprocessing.Process(target=speech_worker, args=(speech_queue,))
    p_speech.daemon = True
    p_speech.start()

    t_cam = threading.Thread(target=process_camera_feed, daemon=True)
    t_cam.start()

    t_mic = threading.Thread(target=listen_voice_command, daemon=True)
    t_mic.start()

    print("\nJarvis Smart Stick is live -> http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
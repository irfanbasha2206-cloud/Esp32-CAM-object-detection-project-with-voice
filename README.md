# 🦯 JARVIS SMART STICK — Assistive Vision System

JARVIS Smart Stick is an edge-connected, assistive computer vision and conversational audio platform designed to aid visually impaired individuals. It processes live video streams from an ESP32-CAM module using YOLOv8 to detect objects, assess collision hazards, calculate spatial positions (left, center, right), and communicate environmental feedback in real time.

The system features bilingual speech interaction (English and Tamil), collision warnings that supersede routine speech, and a low-latency Flask telemetry HUD with a radar display.

---

## ⚡ Core Features

- **Dual Interaction Modes**:
  - **Interactive Mode**: Jarvis remains quiet until triggered (`"Jarvis..."` / `"ஜார்விஸ்..."`). Users can inquire about counts, obstacle locations, and scene clearance.
  - **Detective Mode**: Autonomous proactive narration calling out entities the moment they enter the frame, followed by periodic tracking recaps.
- **Immediate Collision Override**: Any large object detected straight ahead exceeding the safety height threshold (`DANGER_HEIGHT_RATIO = 0.60`) instantly triggers an emergency audio warning in either mode.
- **Bilingual STT/TTS Support**:
  - Speech-to-Text: Auto-detects and distinguishes between English (`en-IN`) and Tamil (`ta-IN`) commands via Google Speech Recognition with phonetic false-positive filtering.
  - Text-to-Speech: Dual-engine pipeline routing English to offline `pyttsx3` and Tamil to `gTTS` with temporary audio playback using `pygame.mixer`.
- **Live Visual Web Console**:
  - MJPEG video feed with real-time detection bounding boxes.
  - SVG radar display rendering obstacle positions across polar coordinates.
  - Live activity log and transcript feed.
  - REST endpoints for dynamic mode switching (`/api/mode`) and language toggling (`/api/language`).

---

## 🏗️ System Architecture

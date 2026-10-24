"""
╔══════════════════════════════════════════════════════════════════╗
║       REAL-TIME EMOTION DETECTION  —  Python 3.10.11             ║
║                                                                  ║
║  Stack:                                                          ║
║   • MediaPipe   → face mesh + 468 landmarks                      ║
║   • DeepFace    → 7-emotion deep learning classifier             ║
║   • OpenCV      → webcam capture & rendering                     ║
║   • Threading   → non-blocking inference for smooth FPS          ║
╚══════════════════════════════════════════════════════════════════╝

Detects 7 emotions per face:
    😠 Angry  🤢 Disgust  😨 Fear  😊 Happy
    😐 Neutral  😢 Sad  😲 Surprise

Requirements  (Python 3.10.11):
    pip install opencv-python mediapipe deepface tf-keras

    Note: On first run DeepFace auto-downloads the facial emotion model
    (~80 MB).  Internet connection required once.

Run in VS Code terminal:
    python emotion_detection.py

Controls:
    Q  → quit
    B  → toggle emotion bar chart
    M  → toggle MediaPipe mesh overlay
    F  → toggle FPS display
    P  → pause / resume inference
"""

from __future__ import annotations

import threading
import time
import queue
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
from deepface import DeepFace


# ══════════════════════════════════════════════════════════════════════════════
#  Configuration
# ══════════════════════════════════════════════════════════════════════════════
CAMERA_INDEX = 0           # change to 1/2 if wrong camera opens
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
# run DeepFace every N frames (balances speed/accuracy)
INFERENCE_EVERY_N = 4
FACE_PAD_FRAC = 0.20        # expand detected face crop by this fraction
SMOOTHING_FRAMES = 5           # temporal smoothing window for probabilities

# ── Emotion metadata ──────────────────────────────────────────────────────────
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

EMOTION_DISPLAY = {
    "angry":    "Angry",
    "disgust":  "Disgust",
    "fear":     "Fear",
    "happy":    "Happy",
    "neutral":  "Neutral",
    "sad":      "Sad",
    "surprise": "Surprise",
}

EMOTION_EMOJI = {
    "angry":    "😠",
    "disgust":  "🤢",
    "fear":     "😨",
    "happy":    "😊",
    "neutral":  "😐",
    "sad":      "😢",
    "surprise": "😲",
}

# BGR colours per emotion
EMOTION_COLORS = {
    "angry":    (40,  40, 220),
    "disgust":  (40, 160,  40),
    "fear":     (160,  40, 220),
    "happy":    (50, 220,  50),
    "neutral":  (180, 180, 180),
    "sad":      (220, 100,  40),
    "surprise": (50, 200, 255),
}

# ── UI constants ──────────────────────────────────────────────────────────────
BAR_MAX_W = 150
BAR_H = 15
BAR_GAP = 8
PANEL_W = 215
TOPBAR_H = 32


# ══════════════════════════════════════════════════════════════════════════════
#  MediaPipe face mesh setup
# ══════════════════════════════════════════════════════════════════════════════
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_draw_styles = mp.solutions.drawing_styles

MESH_DRAWING_SPEC = mp_drawing.DrawingSpec(
    color=(80, 120, 80), thickness=1, circle_radius=1
)


# ══════════════════════════════════════════════════════════════════════════════
#  DeepFace inference worker (runs in background thread)
# ══════════════════════════════════════════════════════════════════════════════
class EmotionWorker:
    """
    Runs DeepFace.analyze() in a background thread so the main loop is
    never blocked.  The latest result is always available via .result.
    """

    def __init__(self) -> None:
        self._frame_q: queue.Queue = queue.Queue(maxsize=1)
        self._result_lock = threading.Lock()
        self._latest: list[dict] = []
        self._running = True

        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def submit(self, frame: np.ndarray) -> None:
        """Submit a frame for analysis (drops frame if worker is busy)."""
        try:
            self._frame_q.put_nowait(frame.copy())
        except queue.Full:
            pass

    @property
    def result(self) -> list[dict]:
        with self._result_lock:
            return list(self._latest)

    def stop(self) -> None:
        self._running = False

    def _worker(self) -> None:
        while self._running:
            try:
                frame = self._frame_q.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                analyses = DeepFace.analyze(
                    frame,
                    actions=["emotion"],
                    enforce_detection=False,
                    detector_backend="opencv",
                    silent=True,
                )
                # DeepFace returns a list (one dict per face)
                if isinstance(analyses, dict):
                    analyses = [analyses]

                with self._result_lock:
                    self._latest = analyses

            except Exception:
                # No face or analysis error — clear results
                with self._result_lock:
                    self._latest = []


# ══════════════════════════════════════════════════════════════════════════════
#  Emotion probability smoother (reduces flicker)
# ══════════════════════════════════════════════════════════════════════════════
class EmotionSmoother:
    def __init__(self, window: int = SMOOTHING_FRAMES) -> None:
        self._history: dict[str, deque] = {
            e: deque(maxlen=window) for e in EMOTIONS
        }

    def update(self, probs: dict[str, float]) -> dict[str, float]:
        for emo in EMOTIONS:
            self._history[emo].append(probs.get(emo, 0.0))
        return {emo: float(np.mean(self._history[emo])) for emo in EMOTIONS}

    def smoothed(self) -> dict[str, float]:
        return {
            emo: float(np.mean(buf)) if buf else 0.0
            for emo, buf in self._history.items()
        }


# ══════════════════════════════════════════════════════════════════════════════
#  Rendering helpers
# ══════════════════════════════════════════════════════════════════════════════
def _softmax_dict(raw: dict[str, float]) -> dict[str, float]:
    vals = np.array([raw.get(e, 0.0) for e in EMOTIONS], dtype=np.float64)
    e = np.exp(vals - vals.max())
    s = e / e.sum()
    return dict(zip(EMOTIONS, s.tolist()))


def draw_face_box(frame: np.ndarray, region: dict,
                  dominant: str, conf: float) -> None:
    """Draw bounding box + emotion pill for one face."""
    x = region.get("x", 0)
    y = region.get("y", 0)
    w = region.get("w", 0)
    h = region.get("h", 0)
    col = EMOTION_COLORS[dominant]

    # Face rectangle
    cv2.rectangle(frame, (x, y), (x + w, y + h), col, 2, cv2.LINE_AA)

    # Corner accents
    corner = 18
    thick = 3
    for (px, py), (dx, dy) in [
        ((x, y),         (1, 1)),
        ((x + w, y),     (-1, 1)),
        ((x, y + h),     (1, -1)),
        ((x + w, y + h), (-1, -1)),
    ]:
        cv2.line(frame, (px, py), (px + dx * corner, py),
                 col, thick, cv2.LINE_AA)
        cv2.line(frame, (px, py), (px, py + dy * corner),
                 col, thick, cv2.LINE_AA)

    # Label pill
    label = f"{EMOTION_DISPLAY[dominant]}  {conf * 100:.0f}%"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 1)
    px1 = x
    py1 = max(0, y - th - 14)
    px2 = x + tw + 14
    py2 = y

    cv2.rectangle(frame, (px1, py1), (px2, py2), col, -1, cv2.LINE_AA)
    cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, label, (px1 + 7, py2 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 1, cv2.LINE_AA)


def draw_bar_chart(frame: np.ndarray, probs: dict[str, float],
                   dominant: str, face_region: dict) -> None:
    """Draw a probability bar chart anchored near the face."""
    H, W = frame.shape[:2]
    row_h = BAR_H + BAR_GAP
    ph = len(EMOTIONS) * row_h + 28

    # Anchor to right of face box, fall back to left if not enough room
    fx2 = face_region.get("x", 0) + face_region.get("w", 0)
    fy = face_region.get("y", 0)
    cx = fx2 + 10
    if cx + PANEL_W > W:
        cx = max(0, face_region.get("x", 0) - PANEL_W - 10)
    cy = fy

    # Dark background panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (cx - 6, cy - 6),
                  (cx + PANEL_W, cy + ph), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    # Thin top border
    cv2.line(frame, (cx - 6, cy - 6), (cx + PANEL_W, cy - 6),
             (80, 80, 80), 1, cv2.LINE_AA)

    sorted_emos = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)

    for i, (emo, prob) in enumerate(sorted_emos):
        y = cy + 18 + i * row_h
        col = EMOTION_COLORS[emo]
        bw = max(2, int(prob * BAR_MAX_W))
        is_dom = (emo == dominant)

        # Highlight dominant row
        if is_dom:
            cv2.rectangle(frame, (cx - 6, y - 14),
                          (cx + PANEL_W - 6, y + BAR_H - 4),
                          (40, 40, 40), -1)

        # Bar background track
        cv2.rectangle(frame, (cx, y - 2), (cx + BAR_MAX_W, y + BAR_H - 6),
                      (45, 45, 45), -1)
        # Filled bar
        cv2.rectangle(frame, (cx, y - 2), (cx + bw, y + BAR_H - 6),
                      col, -1, cv2.LINE_AA)

        # Emotion label
        lbl_col = (255, 255, 255) if is_dom else (160, 160, 160)
        cv2.putText(frame, EMOTION_DISPLAY[emo],
                    (cx + BAR_MAX_W + 8, y + BAR_H - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, lbl_col, 1, cv2.LINE_AA)

        # Percentage
        pct = f"{prob * 100:.0f}%"
        cv2.putText(frame, pct, (cx + bw + 2, y + BAR_H - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, col, 1, cv2.LINE_AA)


def draw_topbar(frame: np.ndarray, fps: float, show_fps: bool,
                n_faces: int, paused: bool) -> None:
    H, W = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (W, TOPBAR_H), (20, 20, 20), -1)

    status = "PAUSED" if paused else f"Faces: {n_faces}"
    info = (
        f"Emotion Detector (Python 3.10)  |  "
        f"{status}  |  B=bars  M=mesh  F=fps  P=pause  Q=quit"
    )
    cv2.putText(frame, info, (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (140, 140, 140), 1, cv2.LINE_AA)

    if show_fps:
        fps_txt = f"FPS {fps:.1f}"
        (tw, _), _ = cv2.getTextSize(fps_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
        col = (80, 220, 80) if fps >= 20 else (
            80, 180, 220) if fps >= 10 else (80, 80, 220)
        cv2.putText(frame, fps_txt, (W - tw - 12, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, col, 1, cv2.LINE_AA)


def draw_no_face_msg(frame: np.ndarray) -> None:
    H, W = frame.shape[:2]
    msg = "No face detected — move closer or improve lighting"
    (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.60, 1)
    x = (W - tw) // 2
    y = H // 2
    cv2.rectangle(frame, (x - 12, y - th - 8), (x + tw + 12, y + 8),
                  (30, 30, 30), -1)
    cv2.putText(frame, msg, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (120, 120, 120), 1, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
#  Main application
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    print("\n╔══════════════════════════════════════════════════════╗")
    print("║   Real-Time Emotion Detection  —  Python 3.10.11      ║")
    print("╚══════════════════════════════════════════════════════╝\n")
    print("Libraries  : OpenCV + MediaPipe + DeepFace")
    print("Emotions   : Angry · Disgust · Fear · Happy · Neutral · Sad · Surprise")
    print("Controls   : B=bars  M=mesh  F=fps  P=pause  Q=quit")
    print("\nNote: DeepFace will auto-download the emotion model (~80 MB)")
    print("      on first run — this may take a minute.\n")

    # ── Open webcam ────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera (index {CAMERA_INDEX}).")
        print("Try changing CAMERA_INDEX to 1 at the top of the script.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    # ── Initialise components ──────────────────────────────────────────────
    worker = EmotionWorker()
    smoother = EmotionSmoother(SMOOTHING_FRAMES)

    # Per-face smoothers stored by face index
    face_smoothers: dict[int, EmotionSmoother] = {}

    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=4,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # ── State flags ───────────────────────────────────────────────────────
    show_bars = True
    show_mesh = True
    show_fps = True
    paused = False

    fps = 0.0
    t_prev = time.perf_counter()
    frame_count = 0
    last_results: list[dict] = []

    print("Camera open. Running …\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to grab frame.")
            break

        frame = cv2.flip(frame, 1)      # mirror so it feels natural
        frame_count += 1

        # ── MediaPipe face mesh ────────────────────────────────────────────
        if show_mesh:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mesh_results = face_mesh.process(rgb)
            if mesh_results.multi_face_landmarks:
                for face_lms in mesh_results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_lms,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_draw_styles
                        .get_default_face_mesh_tesselation_style(),
                    )
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_lms,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_draw_styles
                        .get_default_face_mesh_contours_style(),
                    )

        # ── Submit frame to DeepFace worker ───────────────────────────────
        if not paused and frame_count % INFERENCE_EVERY_N == 0:
            worker.submit(frame)

        # ── Fetch latest inference result ─────────────────────────────────
        if not paused:
            results = worker.result
            if results:
                last_results = results

        # ── Render per-face overlays ──────────────────────────────────────
        for i, analysis in enumerate(last_results):
            region = analysis.get("region", {})
            raw_em = analysis.get("emotion", {})

            if not raw_em or not region:
                continue

            # Normalise to probabilities
            probs = _softmax_dict(raw_em)

            # Per-face temporal smoothing
            if i not in face_smoothers:
                face_smoothers[i] = EmotionSmoother(SMOOTHING_FRAMES)
            probs = face_smoothers[i].update(probs)

            dominant = max(probs, key=probs.__getitem__)
            conf = probs[dominant]

            draw_face_box(frame, region, dominant, conf)
            if show_bars:
                draw_bar_chart(frame, probs, dominant, region)

        if not last_results:
            draw_no_face_msg(frame)

        # ── Top bar & FPS ─────────────────────────────────────────────────
        t_now = time.perf_counter()
        fps = 0.92 * fps + 0.08 / max(t_now - t_prev, 1e-6)
        t_prev = t_now

        draw_topbar(frame, fps, show_fps, len(last_results), paused)

        cv2.imshow("Emotion Detection", frame)

        # ── Key handling ──────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("b"):
            show_bars = not show_bars
        elif key == ord("m"):
            show_mesh = not show_mesh
        elif key == ord("f"):
            show_fps = not show_fps
        elif key == ord("p"):
            paused = not paused
            print("Paused." if paused else "Resumed.")

    # ── Cleanup ───────────────────────────────────────────────────────────
    worker.stop()
    face_mesh.close()
    cap.release()
    cv2.destroyAllWindows()
    print("\nStopped.")


if __name__ == "__main__":
    main()

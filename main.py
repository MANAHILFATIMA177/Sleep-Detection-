import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import winsound  # Windows only

# ─── MediaPipe setup ──────────────────────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# ─── Defaults (overridden by sliders) ─────────────────────────────────────────
DEFAULT_EAR_THRESHOLD = 0.25
DEFAULT_SLEEP_TIME    = 3.0      # seconds eyes must be closed to trigger alert
ALARM_FREQ            = 1000     # Hz
ALARM_DURATION        = 600      # ms per beep
ALARM_INTERVAL        = 1.5      # seconds between repeated beeps


# ─── Helpers ──────────────────────────────────────────────────────────────────
def eye_aspect_ratio(landmarks, eye_indices, w, h):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
    p1, p2, p3, p4, p5, p6 = pts
    v1 = np.linalg.norm(np.array(p2) - np.array(p6))
    v2 = np.linalg.norm(np.array(p3) - np.array(p5))
    hz = np.linalg.norm(np.array(p1) - np.array(p4))
    return (v1 + v2) / (2.0 * hz) if hz != 0 else 0.0


def draw_rounded_rect(img, pt1, pt2, color, radius=10, thickness=2):
    """Draw a rounded rectangle on a numpy image."""
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius),  90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius),   0, 0, 90, color, thickness)


# ─── App ──────────────────────────────────────────────────────────────────────
class SleepDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sleep Detection System")
        self.root.configure(bg="#0f1117")
        self.root.resizable(False, False)

        # State
        self.closed_start   = None
        self.alarm_active   = False
        self.last_alarm_t   = 0.0
        self.blink_count    = 0
        self.alert_count    = 0
        self.session_start  = time.time()
        self._prev_ear_open = True   # for blink detection

        self._build_ui()

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.status_var.set("⚠ Camera not found")
        else:
            self._update_frame()

    # ── UI construction ───────────────────────────────────────────────────────
    def _build_ui(self):
        BG      = "#0f1117"
        PANEL   = "#1a1d27"
        ACCENT  = "#4f8ef7"
        FG      = "#e8eaf0"
        MUTED   = "#6b7280"

        # ── Top: video + overlay panel ──────────────────────────────────────
        top = tk.Frame(self.root, bg=BG)
        top.pack(padx=16, pady=(16, 8))

        self.video_label = tk.Label(top, bg=BG, relief="flat", bd=0)
        self.video_label.pack(side="left")

        side = tk.Frame(top, bg=PANEL, width=220, padx=14, pady=14)
        side.pack(side="left", fill="y", padx=(12, 0))
        side.pack_propagate(False)

        tk.Label(side, text="SLEEP GUARD", bg=PANEL, fg=ACCENT,
                 font=("Courier New", 13, "bold")).pack(anchor="w", pady=(0, 12))

        # Status badge
        self.status_var = tk.StringVar(value="● AWAKE")
        self.status_lbl = tk.Label(side, textvariable=self.status_var,
                                   bg="#1e3a2f", fg="#4ade80",
                                   font=("Courier New", 11, "bold"),
                                   padx=8, pady=4, relief="flat")
        self.status_lbl.pack(fill="x", pady=(0, 14))

        # EAR bar
        tk.Label(side, text="EAR Level", bg=PANEL, fg=MUTED,
                 font=("Courier New", 8)).pack(anchor="w")
        self.ear_bar = ttk.Progressbar(side, length=192, maximum=100,
                                       mode="determinate")
        self.ear_bar.pack(fill="x", pady=(2, 10))

        # Eye-closed duration bar
        tk.Label(side, text="Eye-closed Duration", bg=PANEL, fg=MUTED,
                 font=("Courier New", 8)).pack(anchor="w")
        self.closed_bar = ttk.Progressbar(side, length=192, maximum=100,
                                          mode="determinate")
        self.closed_bar.pack(fill="x", pady=(2, 14))

        # Stats
        stats_frame = tk.Frame(side, bg=PANEL)
        stats_frame.pack(fill="x")

        def stat_row(label, var):
            f = tk.Frame(stats_frame, bg=PANEL)
            f.pack(fill="x", pady=2)
            tk.Label(f, text=label, bg=PANEL, fg=MUTED,
                     font=("Courier New", 8), width=12, anchor="w").pack(side="left")
            lbl = tk.Label(f, textvariable=var, bg=PANEL, fg=FG,
                           font=("Courier New", 9, "bold"), anchor="e")
            lbl.pack(side="right")
            return lbl

        self.ear_var     = tk.StringVar(value="0.00")
        self.blink_var   = tk.StringVar(value="0")
        self.alert_var   = tk.StringVar(value="0")
        self.session_var = tk.StringVar(value="00:00")

        stat_row("EAR",       self.ear_var)
        stat_row("Blinks",    self.blink_var)
        stat_row("Alerts",    self.alert_var)
        stat_row("Session",   self.session_var)

        # ── Bottom: controls ─────────────────────────────────────────────────
        ctrl = tk.Frame(self.root, bg=PANEL, padx=14, pady=10)
        ctrl.pack(fill="x", padx=16, pady=(0, 16))

        def slider_row(parent, label, from_, to, default, resolution=0.01):
            f = tk.Frame(parent, bg=PANEL)
            f.pack(side="left", padx=16)
            tk.Label(f, text=label, bg=PANEL, fg=MUTED,
                     font=("Courier New", 8)).pack(anchor="w")
            var = tk.DoubleVar(value=default)
            s = tk.Scale(f, variable=var, from_=from_, to=to,
                         resolution=resolution, orient="horizontal",
                         bg=PANEL, fg=FG, troughcolor="#2d3148",
                         highlightthickness=0, bd=0, length=160,
                         font=("Courier New", 8))
            s.pack()
            return var

        self.ear_thresh_var  = slider_row(ctrl, "EAR Threshold",   0.10, 0.45, DEFAULT_EAR_THRESHOLD)
        self.sleep_time_var  = slider_row(ctrl, "Alert Delay (s)",  1.0, 10.0, DEFAULT_SLEEP_TIME, 0.5)

        # Mute toggle
        self.muted = tk.BooleanVar(value=False)
        tk.Checkbutton(ctrl, text="Mute alarm", variable=self.muted,
                       bg=PANEL, fg=FG, selectcolor="#2d3148",
                       activebackground=PANEL, activeforeground=FG,
                       font=("Courier New", 9)).pack(side="left", padx=24, pady=4)

    # ── Alarm (threaded so UI stays responsive) ───────────────────────────────
    def _trigger_alarm(self):
        if self.muted.get():
            return
        now = time.time()
        if now - self.last_alarm_t >= ALARM_INTERVAL:
            self.last_alarm_t = now
            threading.Thread(
                target=winsound.Beep,
                args=(ALARM_FREQ, ALARM_DURATION),
                daemon=True
            ).start()

    # ── Main update loop ──────────────────────────────────────────────────────
    def _update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.root.after(30, self._update_frame)
            return

        h, w = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        status   = "AWAKE"
        color    = (74, 222, 128)   # green (RGB for PIL)
        cv_color = (74, 222, 128)   # same in OpenCV (we'll draw on the RGB frame)
        ear      = 0.0
        elapsed  = 0.0

        ear_threshold = self.ear_thresh_var.get()
        sleep_time    = self.sleep_time_var.get()

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark

            x_coords = [int(l.x * w) for l in lm]
            y_coords = [int(l.y * h) for l in lm]
            xmin, xmax = min(x_coords), max(x_coords)
            ymin, ymax = min(y_coords), max(y_coords)

            left_ear  = eye_aspect_ratio(lm, LEFT_EYE,  w, h)
            right_ear = eye_aspect_ratio(lm, RIGHT_EYE, w, h)
            ear       = (left_ear + right_ear) / 2.0

            # Blink detection
            if ear >= ear_threshold and not self._prev_ear_open:
                self.blink_count += 1
            self._prev_ear_open = ear >= ear_threshold

            if ear < ear_threshold:
                if self.closed_start is None:
                    self.closed_start = time.time()
                elapsed = time.time() - self.closed_start

                if elapsed >= sleep_time:
                    status   = "SLEEPING"
                    cv_color = (239, 68, 68)   # red
                    if not self.alarm_active:
                        self.alarm_active = True
                        self.alert_count += 1
                    self._trigger_alarm()
                else:
                    cv_color = (251, 191, 36)  # amber — eyes closing
                    status   = "DROWSY"
                    self.alarm_active = False
            else:
                self.closed_start = None
                self.alarm_active = False

            # Draw face box
            pad = 10
            draw_rounded_rect(rgb,
                              (max(xmin - pad, 0), max(ymin - pad, 0)),
                              (min(xmax + pad, w), min(ymax + pad, h)),
                              cv_color, radius=12, thickness=2)

            # EAR label on frame
            cv2.putText(rgb, f"EAR {ear:.2f}", (xmin, max(ymin - pad - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, cv_color, 1, cv2.LINE_AA)

        # Overlay dark top-bar
        overlay = rgb.copy()
        cv2.rectangle(overlay, (0, 0), (w, 32), (15, 17, 23), -1)
        cv2.addWeighted(overlay, 0.75, rgb, 0.25, 0, rgb)
        session_secs = int(time.time() - self.session_start)
        cv2.putText(rgb, f"Session {session_secs//60:02d}:{session_secs%60:02d}",
                    (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (107, 114, 128), 1, cv2.LINE_AA)
        cv2.putText(rgb, "SLEEP GUARD", (w // 2 - 55, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (79, 142, 247), 1, cv2.LINE_AA)

        # ── Update Tkinter widgets ────────────────────────────────────────────
        self.ear_var.set(f"{ear:.3f}")
        self.blink_var.set(str(self.blink_count))
        self.alert_var.set(str(self.alert_count))
        mins, secs = divmod(session_secs, 60)
        self.session_var.set(f"{mins:02d}:{secs:02d}")

        ear_pct = min(int(ear / 0.45 * 100), 100)
        self.ear_bar["value"] = ear_pct

        closed_pct = min(int(elapsed / sleep_time * 100), 100) if sleep_time > 0 else 0
        self.closed_bar["value"] = closed_pct

        if status == "AWAKE":
            self.status_var.set("● AWAKE")
            self.status_lbl.configure(bg="#1e3a2f", fg="#4ade80")
        elif status == "DROWSY":
            self.status_var.set("◐ DROWSY")
            self.status_lbl.configure(bg="#3b2a10", fg="#fbbf24")
        else:
            self.status_var.set("✕ SLEEPING")
            self.status_lbl.configure(bg="#3b1010", fg="#ef4444")

        # ── Display frame ─────────────────────────────────────────────────────
        img    = Image.fromarray(rgb)
        imgtk  = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.root.after(10, self._update_frame)

    def on_close(self):
        self.cap.release()
        self.root.destroy()


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app  = SleepDetectorApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
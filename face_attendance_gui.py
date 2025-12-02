"""
GUI Application using Tkinter
Simple interface for face recognition attendance system
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import time
import platform
from datetime import datetime
from typing import Optional, Dict, List
from pathlib import Path
import cv2
from PIL import Image, ImageTk

from face_attendance_main import Config

# Optional custom alarm path (Windows WAV). Replace the string below if you want a different file.
ALARM_SOUND_PATH = r"E:\mohamed\projects\attendance_system-main\alarm.wav"

class AttendanceGUI:
    """GUI for Face Recognition Attendance System"""
    
    def __init__(self, face_recognizer, dataset_manager, attendance_logger,
                 face_detector, face_embedder):
        """
        Args:
            face_recognizer: FaceRecognizer instance
            dataset_manager: DatasetManager instance
            attendance_logger: AttendanceLogger instance
            face_detector: FaceDetector instance
            face_embedder: FaceEmbedder instance
        """
        self.face_recognizer = face_recognizer
        self.dataset_manager = dataset_manager
        self.attendance_logger = attendance_logger
        self.face_detector = face_detector
        self.face_embedder = face_embedder
        
        # Camera state
        self.camera_running = False
        self.cap = None
        self.frame_interval_ms = 150  # balance between smoothness and CPU load
        self.last_output_frame = None
        self.processing_frame = False
        self.pending_frame = None
        # Simple single-person tracker state
        self.tracker = None
        self.tracked_name: Optional[str] = None
        self.tracked_last_seen = 0.0
        self.tracking_grace_seconds = 1.0  # how long we tolerate lost tracking
        self.required_known_duration = 2.0  # seconds a person must stay verified before tracking
        self._tracking_candidate_name: Optional[str] = None
        self._tracking_candidate_since = 0.0
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("Face Recognition Attendance System")
        self.root.geometry("1000x700")
        self.setup_styles()
        
        # Add person fields
        self.rank_entry = None
        self.position_entry = None
        self.permission_var = tk.StringVar(value="Yes")
        self.permission_combo = None
        
        # Dataset management UI state
        self.dataset_window = None
        self.person_listbox = None
        self.person_details_text = None
        self.dataset_status_var = tk.StringVar(value="Dataset: 0 people")
        self.person_info_text = None
        self.person_select_var = tk.StringVar()
        self._last_displayed_infos: Optional[List[str]] = None
        self.last_alarm_time = 0.0
        self.alarm_cooldown = 5.0  # seconds between alarm sounds
        default_alarm = Path(__file__).parent / "alarm.wav"
        custom_alarm = Path(ALARM_SOUND_PATH) if ALARM_SOUND_PATH else None
        self.alarm_sound_path: Optional[str] = None
        for candidate in (custom_alarm, default_alarm):
            if candidate and candidate.exists():
                self.alarm_sound_path = str(candidate)
                break
        self._alarm_lock = threading.Lock()
        
        self.setup_ui()
        self.refresh_person_list()
    
    def setup_styles(self):
        """Apply a light modern theme while keeping widget dimensions unchanged."""
        bg = "#f5f7fb"
        primary = "#3A7AF0"
        text = "#1f2a3d"
        self.root.configure(bg=bg)
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background=bg)
        style.configure("TLabel", background=bg, foreground=text)
        style.configure("TLabelframe", background=bg, foreground=text, padding=6)
        style.configure("TLabelframe.Label", background=bg, foreground=text, font=("Arial", 10, "bold"))
        style.configure(
            "TButton",
            background=primary,
            foreground="white",
            padding=(4, 3)
        )
        style.map(
            "TButton",
            background=[("active", "#2f6bd6"), ("pressed", "#285bb6")],
            foreground=[("disabled", "#cbd3e1")]
        )
        style.configure(
            "TCombobox",
            fieldbackground="white",
            background="white",
            foreground=text,
            bordercolor="#cbd3e1",
            arrowcolor=primary
        )
        style.map("TCombobox", fieldbackground=[("readonly", "white")])
    
    def setup_ui(self):
        """Setup user interface"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Left panel - Controls
        self.setup_control_panel(main_frame)
        
        # Right panel - Video/Info
        self.setup_display_panel(main_frame)
    
    def setup_control_panel(self, parent):
        """Setup control panel"""
        container = ttk.Frame(parent)
        container.grid(row=0, column=0, rowspan=2, sticky=(tk.N, tk.S), padx=(0, 10))
        canvas = tk.Canvas(container, highlightthickness=0, width=280)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(container, orient='vertical', command=canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.configure(yscrollcommand=scrollbar.set)
        inner = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=inner, anchor='nw')
        def _on_config(event):
            canvas.configure(scrollregion=canvas.bbox('all'))
        inner.bind("<Configure>", _on_config)

        control_frame = ttk.LabelFrame(inner, text="Controls", padding="10")
        control_frame.pack(fill=tk.BOTH, expand=True)
        
        # Camera controls
        ttk.Label(control_frame, text="Camera Controls", font=('Arial', 10, 'bold')).pack(pady=5)
        
        self.camera_btn = ttk.Button(control_frame, text="Start Camera", 
                                     command=self.toggle_camera)
        self.camera_btn.pack(fill=tk.X, pady=5)
        
        # Add person
        ttk.Separator(control_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        ttk.Label(control_frame, text="Add New Person", font=('Arial', 10, 'bold')).pack(pady=5)
        
        ttk.Label(control_frame, text="Name:").pack()
        self.name_entry = ttk.Entry(control_frame)
        self.name_entry.pack(fill=tk.X, pady=5)
        
        ttk.Label(control_frame, text="Rank:").pack()
        self.rank_entry = ttk.Entry(control_frame)
        self.rank_entry.pack(fill=tk.X, pady=5)
        
        ttk.Label(control_frame, text="Position:").pack()
        self.position_entry = ttk.Entry(control_frame)
        self.position_entry.pack(fill=tk.X, pady=5)
        
        permission_row = ttk.Frame(control_frame)
        permission_row.pack(fill=tk.X, pady=5)
        ttk.Label(permission_row, text="Has Permission:").pack(side=tk.LEFT, padx=(0, 6))
        self.permission_combo = ttk.Combobox(
            permission_row,
            textvariable=self.permission_var,
            values=["Yes", "No"],
            state="readonly"
        )
        self.permission_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.permission_combo.current(0)
        
        ttk.Button(control_frame, text="Add Person", 
                  command=self.add_person).pack(fill=tk.X, pady=5)
        
        # Dataset controls
        ttk.Separator(control_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        ttk.Label(control_frame, text="Dataset Management", font=('Arial', 10, 'bold')).pack(pady=5)
        ttk.Label(control_frame, textvariable=self.dataset_status_var).pack(pady=2)
        ttk.Button(control_frame, text="Open Dataset Manager",
                   command=self.open_dataset_manager).pack(fill=tk.X, pady=5)
        # Attendance
        ttk.Separator(control_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        ttk.Button(control_frame, text="Show Today's Attendance", 
                  command=self.show_attendance).pack(fill=tk.X, pady=5)
        ttk.Button(control_frame, text="View Full Attendance",
                  command=self.show_all_attendance).pack(fill=tk.X, pady=5)
        ttk.Button(control_frame, text="Export Attendance", 
                  command=self.export_attendance).pack(fill=tk.X, pady=5)
        
        # Manual attendance (moved to bottom)
        ttk.Separator(control_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        ttk.Label(control_frame, text="Manual attendance field", font=('Arial', 10, 'bold')).pack(pady=(5, 0))
        person_select_row = ttk.Frame(control_frame)
        person_select_row.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(person_select_row, text="Select Person:").pack(side=tk.LEFT, padx=(0, 6))
        self.person_combo = ttk.Combobox(
            person_select_row,
            textvariable=self.person_select_var,
            state="readonly"
        )
        self.person_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        ttk.Button(control_frame, text="Log Selected Attendance",
                   command=self.log_selected_attendance).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="Remove Selected Attendance",
                   command=self.remove_selected_attendance).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="Clear Attendance Log",
                   command=self.clear_attendance_log).pack(fill=tk.X, pady=2)
    
    def setup_display_panel(self, parent):
        """Setup display panel"""
        display_frame = ttk.Frame(parent)
        display_frame.grid(row=0, column=1, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        display_frame.rowconfigure(0, weight=3)
        display_frame.rowconfigure(1, weight=0)
        display_frame.rowconfigure(2, weight=1)
        display_frame.columnconfigure(0, weight=1)
        
        # Video display
        video_frame = ttk.LabelFrame(display_frame, text="Camera Feed", padding="5")
        video_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        self.video_label = ttk.Label(video_frame)
        self.video_label.pack(expand=True, fill=tk.BOTH)
        
        # Threshold control placed between camera and log
        threshold_frame = ttk.LabelFrame(display_frame, text="Threshold", padding="5")
        threshold_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        ttk.Label(threshold_frame, text="Adjust recognition threshold").pack(anchor=tk.W)
        self.threshold_var = tk.DoubleVar(value=0.6)
        threshold_scale = ttk.Scale(
            threshold_frame,
            from_=0.3,
            to=0.9,
            variable=self.threshold_var,
            command=self.update_threshold
        )
        threshold_scale.pack(fill=tk.X, pady=5)
        self.threshold_label = ttk.Label(threshold_frame, text="0.60")
        self.threshold_label.pack(anchor=tk.E)
        
        # Log display
        log_frame = ttk.LabelFrame(display_frame, text="System Log", padding="5")
        log_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 8))
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=5, wrap=tk.WORD)
        self.log_text.pack(expand=True, fill=tk.BOTH)
        
        self.person_info_text = None
    
    def log(self, message):
        """Add message to log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)

    def _trigger_alarm(self):
        """Play an alarm sound when an unauthorized person is seen."""
        now = time.time()
        with self._alarm_lock:
            if now - self.last_alarm_time < self.alarm_cooldown:
                return
            self.last_alarm_time = now
        threading.Thread(target=self._play_alarm_sound, daemon=True).start()

    def _play_alarm_sound(self):
        """
        Non-blocking alarm sound with a safe fallback.
        If a WAV file path is set (Windows), it will be played; otherwise, a simple beep is used.
        """
        try:
            if platform.system() == "Windows":
                import winsound
                if self.alarm_sound_path:
                    winsound.PlaySound(self.alarm_sound_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
                else:
                    pattern = [(900, 250), (1200, 350), (900, 250)]
                    for freq, dur in pattern:
                        winsound.Beep(freq, dur)
            else:
                print("\a", end="", flush=True)
        except Exception as exc:
            self.root.after(0, lambda: self.log(f"Alarm sound failed: {exc}"))

    def _create_tracker(self):
        """Try to create an OpenCV tracker (handles legacy namespace)."""
        candidates = [
            ("legacy", "TrackerCSRT_create"),
            ("", "TrackerCSRT_create"),
            ("legacy", "TrackerKCF_create"),
            ("", "TrackerKCF_create"),
        ]
        for namespace, name in candidates:
            parent = getattr(cv2, namespace, cv2) if namespace else cv2
            creator = getattr(parent, name, None)
            if creator:
                try:
                    return creator()
                except Exception:
                    continue
        return None

    def _start_tracking(self, frame, bbox, name):
        """Initialize tracking for a verified person after the hold duration."""
        tracker = self._create_tracker()
        if not tracker:
            self.log("Tracking unavailable: no supported OpenCV tracker found.")
            return
        ok = tracker.init(frame, tuple(map(float, bbox)))
        if not ok:
            self.log("Failed to initialize tracker.")
            return
        self.tracker = tracker
        self.tracked_name = name
        self.tracked_last_seen = time.time()
        self._tracking_candidate_name = None
        self._tracking_candidate_since = 0.0
        self.log(f"Tracking locked on {name} (hold {self.required_known_duration:.1f}s reached)")

    def _clear_tracking(self):
        """Reset tracking state when the person leaves or tracking is lost."""
        self.tracker = None
        self.tracked_name = None
        self.tracked_last_seen = 0.0
        self._tracking_candidate_name = None
        self._tracking_candidate_since = 0.0

    def _result_has_permission(self, result: Dict) -> bool:
        """Return True only for people with permission explicitly set to Yes."""
        info = result.get('info')
        if not info and result.get('name'):
            info = self.dataset_manager.get_person_info(result['name'])
        return bool(info and info.get('has_permission') is True)

    @staticmethod
    def _bbox_iou(box_a, box_b) -> float:
        """Intersection over Union for (x, y, w, h) boxes."""
        ax, ay, aw, ah = box_a
        bx, by, bw, bh = box_b
        a_x2, a_y2 = ax + aw, ay + ah
        b_x2, b_y2 = bx + bw, by + bh
        inter_x1, inter_y1 = max(ax, bx), max(ay, by)
        inter_x2, inter_y2 = min(a_x2, b_x2), min(a_y2, b_y2)
        inter_w, inter_h = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        area_a = aw * ah
        area_b = bw * bh
        union = area_a + area_b - inter_area
        return inter_area / union if union > 0 else 0.0
    
    def update_live_person_info(self, infos: Optional[List[Dict]]):
        """Details pane removed."""
        return
    
    def toggle_camera(self):
        """Start/stop camera"""
        if not self.camera_running:
            self.start_camera()
        else:
            self.stop_camera()
    
    def start_camera(self):
        """Start camera feed"""
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(Config.CAMERA_ID)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Failed to open camera")
            return
        
        self.camera_running = True
        self.camera_btn.config(text="Stop Camera")
        self.log("Camera started")
        self.last_output_frame = None
        self.processing_frame = False
        
        # Start update loop
        self.update_camera()
    
    def stop_camera(self):
        """Stop camera feed"""
        self.camera_running = False
        self.processing_frame = False
        self._clear_tracking()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.camera_btn.config(text="Start Camera")
        self.video_label.config(image='')
        self.video_label.image = None
        self.log("Camera stopped")
    
    def update_camera(self):
        """Update camera frame"""
        if not self.camera_running or not self.cap:
            return
        
        ret, frame = self.cap.read()
        if ret:
            self.pending_frame = frame.copy()
            if not self.processing_frame:
                thread = threading.Thread(target=self._process_frame, args=(self.pending_frame.copy(),), daemon=True)
                self.processing_frame = True
                thread.start()
            
            display_frame = self.last_output_frame if self.last_output_frame is not None else frame
            frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = img.resize((640, 480), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image=img)
            self.video_label.config(image=photo)
            self.video_label.image = photo
        
        self.root.after(self.frame_interval_ms, self.update_camera)

    def _process_frame(self, frame):
        """Recognize faces in a background thread"""
        try:
            now = time.time()
            tracking_box = None

            if self.tracker and self.tracked_name:
                ok, bbox = self.tracker.update(frame)
                if ok:
                    self.tracked_last_seen = now
                    tracking_box = bbox
                elif now - self.tracked_last_seen > self.tracking_grace_seconds:
                    self._clear_tracking()

            results = self.face_recognizer.recognize_from_frame(frame)
            tracked_detection = None
            if self.tracked_name:
                tracked_detection = next((r for r in results if r.get('name') == self.tracked_name), None)

            if not self.tracked_name:
                candidate = next(
                    (r for r in results if r['verified'] and self._result_has_permission(r)),
                    None
                )
                if candidate:
                    if self._tracking_candidate_name == candidate['name']:
                        if now - self._tracking_candidate_since >= self.required_known_duration:
                            self._start_tracking(frame, candidate['bbox'], candidate['name'])
                            tracking_box = candidate['bbox']
                    else:
                        self._tracking_candidate_name = candidate['name']
                        self._tracking_candidate_since = now
                else:
                    self._tracking_candidate_name = None
                    self._tracking_candidate_since = 0.0
            else:
                # If we lost the tracker but still detect the person, re-anchor; if drifted, snap back.
                if tracked_detection:
                    det_box = tracked_detection['bbox']
                    if tracking_box:
                        iou = self._bbox_iou(tracking_box, det_box)
                        if iou < 0.1:
                            self._start_tracking(frame, det_box, self.tracked_name)
                            tracking_box = det_box
                    else:
                        self._start_tracking(frame, det_box, self.tracked_name)
                        tracking_box = det_box
                elif now - self.tracked_last_seen > self.tracking_grace_seconds:
                    self._clear_tracking()

            filtered_results = [
                r for r in results
                if not (self.tracked_name and r.get('name') == self.tracked_name)
            ]

            log_messages = []
            unauthorized_detected = False
            for result in filtered_results:
                if result['verified']:
                    success = self.attendance_logger.log_attendance(
                        result['name'],
                        result['similarity'],
                        result['face']
                    )
                    if success:
                        log_messages.append(f"✓ {result['name']} - Attendance logged")
            detected_infos = []
            for result in filtered_results:
                if result['verified']:
                    info = result.get('info')
                    if not info and result['name']:
                        info = self.dataset_manager.get_person_info(result['name'])
                    if info:
                        detected_infos.append(info)
                        if info.get('has_permission') is False:
                            unauthorized_detected = True
            if unauthorized_detected:
                self._trigger_alarm()
            self.root.after(
                0,
                lambda infos=detected_infos, msgs=log_messages, alert=unauthorized_detected:
                    self._apply_recognition_updates(infos, msgs, alert)
            )
            output_frame = self.face_recognizer.draw_results(frame, filtered_results)
            if tracking_box and self.tracked_name:
                x, y, w, h = [int(v) for v in tracking_box]
                info = self.dataset_manager.get_person_info(self.tracked_name) or {}
                color = (0, 255, 0)  # green
                label = f"{self.tracked_name}"
                rank = info.get('rank') or "N/A"
                position = info.get('position') or "N/A"
                perm_str = "Yes" if info.get('has_permission') else "No"
                extra_text_lines = [
                    f"{label} | Rank: {rank}",
                    f"Position: {position}",
                    f"Perm: {perm_str}",
                ]
                cv2.rectangle(output_frame, (x, y), (x + w, y + h), color, 2)
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                top_y = max(0, y - 25)
                cv2.rectangle(output_frame, (x, top_y), (x + label_size[0], y), color, -1)
                cv2.putText(
                    output_frame,
                    label,
                    (x, y - 8 if y - 8 > 0 else y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    2
                )
                if extra_text_lines:
                    text_y = y + h + 25
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    thickness = 1
                    sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in extra_text_lines]
                    max_width = max(w for w, _ in sizes)
                    line_height = max(hh for _, hh in sizes) + 4
                    box_height = line_height * len(extra_text_lines) + 6
                    cv2.rectangle(
                        output_frame,
                        (x, text_y - box_height),
                        (x + max_width + 10, text_y),
                        color,
                        -1
                    )
                    current_y = text_y - box_height + line_height
                    for line in extra_text_lines:
                        cv2.putText(
                            output_frame,
                            line,
                            (x + 5, current_y - 2),
                            font,
                            font_scale,
                            (0, 0, 0),
                            thickness
                        )
                        current_y += line_height
            self.last_output_frame = output_frame
        finally:
            self.processing_frame = False

    def _apply_recognition_updates(self, infos, log_messages, unauthorized=False):
        for msg in log_messages:
            self.log(msg)
        if unauthorized:
            self.log("ALERT: Person detected without permission")
        self.update_live_person_info(infos)
    
    def add_person(self):
        """Add new person to dataset"""
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showwarning("Warning", "Please enter a name")
            return
        
        rank = self.rank_entry.get().strip() if self.rank_entry else ""
        if not rank:
            messagebox.showwarning("Warning", "Please enter a rank")
            return
        
        position = self.position_entry.get().strip() if self.position_entry else ""
        if not position:
            messagebox.showwarning("Warning", "Please enter a position/role")
            return
        
        perm_value = self.permission_var.get().strip().lower()
        if perm_value not in ("yes", "no"):
            messagebox.showwarning("Warning", "Select permission (Yes/No)")
            return
        has_permission = perm_value == "yes"
        
        # Check if already exists
        if name in self.dataset_manager.list_people():
            messagebox.showwarning("Warning", f"Person '{name}' already exists")
            return
        
        self.log(f"Adding person: {name}")
        
        # Stop camera if running
        was_running = self.camera_running
        if was_running:
            self.stop_camera()
        
        self.root.update_idletasks()
        success = self.dataset_manager.add_person(
            name, self.face_detector, self.face_embedder,
            rank=rank, position=position, has_permission=has_permission,
            camera_id=Config.CAMERA_ID,
            num_images=Config.IMAGES_PER_PERSON,
            delay=Config.CAPTURE_DELAY
        )
        
        if success:
            # Update recognizer
            self.face_recognizer.update_embeddings()
            self.refresh_person_list()
            self.log(f"✓ Successfully added {name}")
            messagebox.showinfo("Success", f"Successfully added {name}")
            self.name_entry.delete(0, tk.END)
            self.rank_entry.delete(0, tk.END)
            self.position_entry.delete(0, tk.END)
            self.permission_var.set("Yes")
            if self.permission_combo:
                self.permission_combo.set("Yes")
        else:
            self.log(f"✗ Failed to add {name}")
            messagebox.showerror("Error", f"Failed to add {name}")
        
        # Restart camera if it was running
        if was_running or not self.camera_running:
            self.root.after(100, self.start_camera)

    def _selected_person_name(self) -> Optional[str]:
        name = self.person_select_var.get().strip()
        if not name:
            messagebox.showwarning("Select Person", "Please select a person first.")
            return None
        return name

    def log_selected_attendance(self):
        name = self._selected_person_name()
        if not name:
            return
        success = self.attendance_logger.log_attendance(name, similarity=1.0, face_image=None)
        if success:
            self.log(f"Manual attendance logged for {name}")
            messagebox.showinfo("Success", f"Attendance logged for {name}")
        else:
            messagebox.showinfo("Info", f"{name} already logged today.")

    def remove_selected_attendance(self):
        name = self._selected_person_name()
        if not name:
            return
        if not messagebox.askyesno("Confirm", f"Remove attendance records for {name}?"):
            return
        removed = self.attendance_logger.remove_person_records(name)
        if removed:
            self.log(f"Removed attendance records for {name}")
            messagebox.showinfo("Removed", f"Attendance records removed for {name}")
        else:
            messagebox.showinfo("Info", f"No attendance records found for {name}")

    def clear_attendance_log(self):
        if not messagebox.askyesno("Confirm", "Clear all attendance records?"):
            return
        self.attendance_logger.clear_attendance()
        self.log("Attendance log cleared")
        messagebox.showinfo("Cleared", "Attendance log cleared.")
    
    def refresh_person_list(self):
        """Refresh dataset summary and list window"""
        people = self.dataset_manager.list_people()
        self.dataset_status_var.set(f"Dataset: {len(people)} people")
        
        prev_selected = None
        if self.person_listbox:
            selection = self.person_listbox.curselection()
            if selection:
                prev_selected = self.person_listbox.get(selection[0])
            self.person_listbox.delete(0, tk.END)
            selected_index = None
            for idx, person in enumerate(people):
                self.person_listbox.insert(tk.END, person)
                if person == prev_selected:
                    selected_index = idx
            if selected_index is not None:
                self.person_listbox.selection_set(selected_index)
                self.person_listbox.activate(selected_index)
            self.show_selected_person_details()
        else:
            self.update_person_details(None)
        
        if hasattr(self, 'person_combo'):
            current = self.person_select_var.get()
            self.person_combo['values'] = people
            if current not in people:
                self.person_select_var.set('')
        return people
    
    def open_dataset_manager(self):
        """Open dataset management window"""
        if self.dataset_window and self.dataset_window.winfo_exists():
            self.dataset_window.lift()
            self.dataset_window.focus_force()
            return
        
        self.dataset_window = tk.Toplevel(self.root)
        self.dataset_window.title("Dataset Manager")
        self.dataset_window.geometry("450x520")
        self.dataset_window.protocol("WM_DELETE_WINDOW", self.close_dataset_manager)
        self.dataset_window.transient(self.root)
        
        container = ttk.Frame(self.dataset_window, padding="10")
        container.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(container, text="Manage People", font=('Arial', 12, 'bold')).pack(anchor=tk.W)
        ttk.Label(container, textvariable=self.dataset_status_var).pack(anchor=tk.W, pady=(0, 10))
        
        list_frame = ttk.Frame(container)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        self.person_listbox = tk.Listbox(
            list_frame, height=12, exportselection=False
        )
        self.person_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.person_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.person_listbox.config(yscrollcommand=scrollbar.set)
        self.person_listbox.bind("<Double-Button-1>", lambda _: self.view_person_info())
        self.person_listbox.bind("<<ListboxSelect>>", lambda _: self.show_selected_person_details())
        
        button_frame = ttk.Frame(container, padding=(0, 10, 0, 0))
        button_frame.pack(fill=tk.X)
        
        ttk.Button(button_frame, text="Refresh List", command=self.refresh_person_list).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="View Person Info", command=self.view_person_info).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="Delete Person", command=self.delete_person).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="Show Dataset Stats", command=self.show_dataset_stats).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="Rebuild Embeddings", command=self.rebuild_embeddings).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="Close", command=self.close_dataset_manager).pack(fill=tk.X, pady=10)
        
        details_frame = ttk.LabelFrame(container, text="Person Details", padding="5")
        details_frame.pack(fill=tk.BOTH, expand=True)
        self.person_details_text = tk.Text(details_frame, height=6, wrap=tk.WORD)
        self.person_details_text.pack(fill=tk.BOTH, expand=True)
        self.person_details_text.configure(state='disabled')
        
        self.refresh_person_list()
        self.update_person_details(None)
    
    def close_dataset_manager(self):
        """Close dataset manager window"""
        if self.dataset_window:
            self.dataset_window.destroy()
        self.dataset_window = None
        self.person_listbox = None
        self.person_details_text = None
    
    def _current_selection_name(self) -> Optional[str]:
        if not self.person_listbox:
            return None
        selection = self.person_listbox.curselection()
        if not selection:
            return None
        return self.person_listbox.get(selection[0])
    
    def get_selected_person(self) -> Optional[str]:
        """Get selected person from list"""
        if not self.person_listbox:
            messagebox.showwarning("Dataset Manager", "Open the dataset manager to select a person.")
            return None
        name = self._current_selection_name()
        if not name:
            messagebox.showwarning("Select Person", "Please select a person first.")
            return None
        return name
    
    def show_selected_person_details(self):
        """Update detail panel based on list selection"""
        name = self._current_selection_name()
        if not name:
            self.update_person_details(None)
            return
        info = self.dataset_manager.get_person_info(name)
        self.update_person_details(info)
    
    def update_person_details(self, info: Optional[Dict]):
        """Render details in the dataset manager window"""
        if not self.person_details_text:
            return
        self.person_details_text.configure(state='normal')
        self.person_details_text.delete("1.0", tk.END)
        if not info:
            self.person_details_text.insert(tk.END, "Select a person to view details.")
        else:
            permission_text = "Yes" if info.get('has_permission') else "No"
            metadata = info.get('metadata', {})
            added_ts = metadata.get('added_timestamp')
            rebuilt_ts = metadata.get('rebuilt_timestamp')
            if added_ts:
                added_str = datetime.fromtimestamp(added_ts).strftime("%Y-%m-%d %H:%M")
            else:
                added_str = "N/A"
            if rebuilt_ts:
                rebuilt_str = datetime.fromtimestamp(rebuilt_ts).strftime("%Y-%m-%d %H:%M")
            else:
                rebuilt_str = "N/A"
            self.person_details_text.insert(
                tk.END,
                (
                    f"Name: {info.get('name','')}\n"
                    f"Rank: {info.get('rank','')}\n"
                    f"Position: {info.get('position','N/A')}\n"
                    f"Has Permission: {permission_text}\n"
                    f"Images: {info.get('num_images','N/A')}\n"
                    f"Embeddings: {info.get('num_embeddings','N/A')}\n"
                    f"Directory: {info.get('directory','')}\n"
                    f"Added: {added_str}\n"
                    f"Last Rebuilt: {rebuilt_str}\n"
                )
            )
        self.person_details_text.configure(state='disabled')
    
    def _format_person_message(self, info: Dict) -> str:
        permission_text = "Yes" if info.get('has_permission') else "No"
        message = (
            f"Name: {info.get('name','')}\n"
            f"Rank: {info.get('rank','')}\n"
            f"Position: {info.get('position','N/A')}\n"
            f"Has Permission: {permission_text}\n"
        )
        return message
    
    def view_person_info(self):
        """Show info about selected person"""
        name = self.get_selected_person()
        if not name:
            return
        
        info = self.dataset_manager.get_person_info(name)
        if not info:
            messagebox.showerror("Error", f"No data found for {name}")
            return
        
        message = self._format_person_message(info)
        self.update_person_details(info)
        self.log(f"Viewing info for {name}")
        messagebox.showinfo("Person Info", message)
    
    def delete_person(self):
        """Delete selected person from dataset"""
        name = self.get_selected_person()
        if not name:
            return
        
        if not messagebox.askyesno("Confirm Delete", f"Delete all data for '{name}'?"):
            return
        
        was_running = self.camera_running
        if was_running:
            self.stop_camera()
        
        removed = self.dataset_manager.remove_person(name)
        if removed:
            self.face_recognizer.update_embeddings()
            self.refresh_person_list()
            self.update_person_details(None)
            self.log(f"Deleted {name} from dataset")
            messagebox.showinfo("Deleted", f"Removed {name} from dataset")
        else:
            messagebox.showwarning("Not Found", f"No data found for {name}")
        
        if was_running:
            self.root.after(100, self.start_camera)
    
    def update_threshold(self, value):
        """Update similarity threshold"""
        threshold = float(value)
        self.threshold_label.config(text=f"{threshold:.2f}")
        self.face_recognizer.set_threshold(threshold)
    
    def show_dataset_stats(self):
        """Show dataset statistics"""
        stats = self.dataset_manager.get_stats()
        
        message = f"Dataset Statistics\n\n"
        message += f"Total People: {stats['num_people']}\n\n"
        
        for person in stats['people']:
            message += f"• {person['name']}: {person['num_embeddings']} embeddings\n"
        
        self.log("Showing dataset stats")
        messagebox.showinfo("Dataset Statistics", message)
    
    def rebuild_embeddings(self):
        """Rebuild all embeddings"""
        if not messagebox.askyesno("Confirm", "Rebuild all embeddings? This may take a while."):
            return
        
        self.log("Rebuilding embeddings...")
        
        def rebuild_thread():
            self.dataset_manager.rebuild_embeddings(
                self.face_detector, self.face_embedder
            )
            self.face_recognizer.update_embeddings()
            self.root.after(0, self._on_rebuild_complete)
        
        threading.Thread(target=rebuild_thread, daemon=True).start()
    
    def _on_rebuild_complete(self):
        """Handle UI updates after rebuild completes"""
        self.refresh_person_list()
        self.log("✓ Embeddings rebuilt successfully")
        messagebox.showinfo("Success", "Embeddings rebuilt successfully")
    
    def show_attendance(self):
        """Show today's attendance"""
        records = self.attendance_logger.get_today_attendance()
        
        if not records:
            messagebox.showinfo("Attendance", "No attendance records for today")
            return
        records_by_name = {}
        for row in records:
            records_by_name.setdefault(row['Name'], []).append(row)
        
        top = tk.Toplevel(self.root)
        top.title("Today's Attendance")
        top.geometry("520x360")
        top.columnconfigure(0, weight=1)
        top.columnconfigure(1, weight=2)
        top.rowconfigure(0, weight=1)

        list_frame = ttk.Frame(top)
        list_frame.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))
        list_frame.rowconfigure(0, weight=1)
        list_frame.columnconfigure(0, weight=1)
        listbox = tk.Listbox(list_frame, exportselection=False)
        listbox.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))
        scroll = ttk.Scrollbar(list_frame, orient='vertical', command=listbox.yview)
        scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))
        listbox.config(yscrollcommand=scroll.set)

        detail = scrolledtext.ScrolledText(top, wrap=tk.WORD)
        detail.grid(row=0, column=1, sticky=(tk.N, tk.S, tk.E, tk.W), padx=(5, 0))
        detail.configure(state='disabled')
        
        names = sorted(records_by_name.keys())
        for name in names:
            listbox.insert(tk.END, name)

        def show_details(event=None):
            selection = listbox.curselection()
            if not selection:
                return
            name = listbox.get(selection[0])
            person_info = self.dataset_manager.get_person_info(name)
            entries = records_by_name.get(name, [])
            detail.configure(state='normal')
            detail.delete("1.0", tk.END)
            if person_info:
                detail.insert(tk.END, self._format_person_message(person_info))
                detail.insert(tk.END, "\n")
            detail.insert(tk.END, f"Today's records for {name}:\n")
            detail.insert(tk.END, "-"*60 + "\n")
            for entry in entries:
                detail.insert(
                    tk.END,
                    f"Time: {entry['Time']}  Similarity: {entry['Similarity']}\n"
                )
            detail.configure(state='disabled')

        listbox.bind("<<ListboxSelect>>", show_details)
        if names:
            listbox.selection_set(0)
            show_details()
        
        self.log("Showing today's attendance in detail window")
    
    def export_attendance(self):
        """Export attendance to file"""
        stats = self.attendance_logger.get_stats()
        message = f"Attendance file location:\n{self.attendance_logger.attendance_file}\n\n"
        message += f"Total records: {stats['total_records']}\n"
        message += f"Total people: {stats['total_people']}\n"
        message += f"Today's records: {stats['today_records']}"
        
        self.log("Attendance file info displayed")
        messagebox.showinfo("Export Attendance", message)
    
    def show_all_attendance(self):
        """Show all attendance records"""
        records = self.attendance_logger.get_all_attendance()
        if not records:
            messagebox.showinfo("Attendance", "No attendance records available")
            return

        by_name = {}
        for row in records:
            by_name.setdefault(row['Name'], []).append(row)
        
        top = tk.Toplevel(self.root)
        top.title("Full Attendance Log")
        top.geometry("620x430")
        
        top.columnconfigure(0, weight=1)
        top.columnconfigure(1, weight=2)
        top.rowconfigure(0, weight=1)
        
        list_frame = ttk.Frame(top)
        list_frame.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))
        list_frame.rowconfigure(0, weight=1)
        list_frame.columnconfigure(0, weight=1)
        name_list = tk.Listbox(list_frame, exportselection=False)
        name_list.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))
        scroll = ttk.Scrollbar(list_frame, orient='vertical', command=name_list.yview)
        scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))
        name_list.config(yscrollcommand=scroll.set)
        
        detail_text = scrolledtext.ScrolledText(top, wrap=tk.WORD)
        detail_text.grid(row=0, column=1, sticky=(tk.N, tk.S, tk.E, tk.W), padx=(5, 0))
        detail_text.configure(state='disabled')
        
        names = sorted(by_name.keys())
        for name in names:
            name_list.insert(tk.END, name)

        def show_details(event=None):
            selection = name_list.curselection()
            if not selection:
                return
            name = name_list.get(selection[0])
            records_list = by_name.get(name, [])
            info = self.dataset_manager.get_person_info(name)
            
            detail_text.configure(state='normal')
            detail_text.delete("1.0", tk.END)
            
            if info:
                detail_text.insert(tk.END, self._format_person_message(info))
                detail_text.insert(tk.END, "\n")
            detail_text.insert(tk.END, f"Attendance records for {name}:\n")
            detail_text.insert(tk.END, "-" * 60 + "\n")
            for row in records_list:
                detail_text.insert(
                    tk.END,
                    f"Date: {row['Date']}  Time: {row['Time']}  Similarity: {row['Similarity']}\n"
                )
            detail_text.configure(state='disabled')

        name_list.bind("<<ListboxSelect>>", show_details)
        if names:
            name_list.selection_set(0)
            show_details()
    
    def run(self):
        """Run the GUI"""
        self.log("System initialized")
        self.log(f"Known people: {len(self.dataset_manager.list_people())}")
        self.root.mainloop()
    
    def __del__(self):
        """Cleanup"""
        if self.cap:
            self.cap.release()

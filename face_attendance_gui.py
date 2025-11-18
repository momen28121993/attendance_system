"""
GUI Application using Tkinter
Simple interface for face recognition attendance system
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
from datetime import datetime
from typing import Optional, Dict
import cv2
from PIL import Image, ImageTk

class AttendanceGUI:
    """GUI for Face Recognition Attendance System"""
    
    def __init__(self, face_recognizer, dataset_manager, attendance_logger,
                 face_detector, face_embedder, siamese_model):
        """
        Args:
            face_recognizer: FaceRecognizer instance
            dataset_manager: DatasetManager instance
            attendance_logger: AttendanceLogger instance
            face_detector: FaceDetector instance
            face_embedder: FaceEmbedder instance
            siamese_model: SiameseNetwork instance
        """
        self.face_recognizer = face_recognizer
        self.dataset_manager = dataset_manager
        self.attendance_logger = attendance_logger
        self.face_detector = face_detector
        self.face_embedder = face_embedder
        self.siamese_model = siamese_model
        
        # Camera state
        self.camera_running = False
        self.cap = None
        self.frame_interval_ms = 125  # ~8 frames per second
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("Face Recognition Attendance System")
        self.root.geometry("1000x700")
        
        # Add person fields
        self.rank_entry = None
        self.age_entry = None
        self.permission_var = tk.StringVar(value="Yes")
        self.permission_combo = None
        
        # Dataset management UI state
        self.dataset_window = None
        self.person_listbox = None
        self.person_details_text = None
        self.dataset_status_var = tk.StringVar(value="Dataset: 0 people")
        self.person_info_text = None
        
        self.setup_ui()
        self.refresh_person_list()
    
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
        control_frame = ttk.LabelFrame(parent, text="Controls", padding="10")
        control_frame.grid(row=0, column=0, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
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
        
        ttk.Label(control_frame, text="Age:").pack()
        self.age_entry = ttk.Entry(control_frame)
        self.age_entry.pack(fill=tk.X, pady=5)
        
        ttk.Label(control_frame, text="Has Permission:").pack()
        self.permission_combo = ttk.Combobox(
            control_frame,
            textvariable=self.permission_var,
            values=["Yes", "No"],
            state="readonly"
        )
        self.permission_combo.pack(fill=tk.X, pady=5)
        self.permission_combo.current(0)
        
        ttk.Button(control_frame, text="Add Person", 
                  command=self.add_person).pack(fill=tk.X, pady=5)
        
        # Settings
        ttk.Separator(control_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        ttk.Label(control_frame, text="Settings", font=('Arial', 10, 'bold')).pack(pady=5)
        
        ttk.Label(control_frame, text="Threshold:").pack()
        self.threshold_var = tk.DoubleVar(value=0.6)
        threshold_scale = ttk.Scale(control_frame, from_=0.3, to=0.9, 
                                   variable=self.threshold_var,
                                   command=self.update_threshold)
        threshold_scale.pack(fill=tk.X, pady=5)
        self.threshold_label = ttk.Label(control_frame, text="0.60")
        self.threshold_label.pack()
        
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
        
        ttk.Button(control_frame, text="Export Attendance", 
                  command=self.export_attendance).pack(fill=tk.X, pady=5)
    
    def setup_display_panel(self, parent):
        """Setup display panel"""
        display_frame = ttk.Frame(parent)
        display_frame.grid(row=0, column=1, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        display_frame.rowconfigure(0, weight=3)
        display_frame.rowconfigure(1, weight=1)
        display_frame.columnconfigure(0, weight=2)
        display_frame.columnconfigure(1, weight=1)
        
        # Video display
        video_frame = ttk.LabelFrame(display_frame, text="Camera Feed", padding="5")
        video_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        self.video_label = ttk.Label(video_frame)
        self.video_label.pack(expand=True, fill=tk.BOTH)
        
        # Log display
        log_frame = ttk.LabelFrame(display_frame, text="System Log", padding="5")
        log_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 8))
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=10, wrap=tk.WORD)
        self.log_text.pack(expand=True, fill=tk.BOTH)
        
        # Live person info display
        info_frame = ttk.LabelFrame(display_frame, text="Detected Person Details", padding="5")
        info_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.person_info_text = tk.Text(info_frame, height=10, wrap=tk.WORD)
        self.person_info_text.pack(expand=True, fill=tk.BOTH)
        self.person_info_text.configure(state='disabled')
        self.update_live_person_info(None)
    
    def log(self, message):
        """Add message to log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
    
    def update_live_person_info(self, info: Optional[Dict]):
        """Show the latest recognized person's info"""
        if not self.person_info_text:
            return
        self.person_info_text.configure(state='normal')
        self.person_info_text.delete("1.0", tk.END)
        if info:
            text = self._format_person_message(info)
        else:
            text = "No person detected.\nRecognized person details will appear here."
        self.person_info_text.insert(tk.END, text)
        self.person_info_text.configure(state='disabled')
    
    def toggle_camera(self):
        """Start/stop camera"""
        if not self.camera_running:
            self.start_camera()
        else:
            self.stop_camera()
    
    def start_camera(self):
        """Start camera feed"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Failed to open camera")
            return
        
        self.camera_running = True
        self.camera_btn.config(text="Stop Camera")
        self.log("Camera started")
        
        # Start update loop
        self.update_camera()
    
    def stop_camera(self):
        """Stop camera feed"""
        self.camera_running = False
        if self.cap:
            self.cap.release()
        self.camera_btn.config(text="Start Camera")
        self.video_label.config(image='')
        self.log("Camera stopped")
    
    def update_camera(self):
        """Update camera frame"""
        if not self.camera_running:
            return
        
        ret, frame = self.cap.read()
        if ret:
            # Recognize faces
            results = self.face_recognizer.recognize_from_frame(frame)
            
            # Log attendance
            for result in results:
                if result['verified']:
                    success = self.attendance_logger.log_attendance(
                        result['name'],
                        result['similarity'],
                        result['face']
                    )
                    if success:
                        self.log(f"✓ {result['name']} - Attendance logged")
            
            # Determine first recognized person for details panel
            detected_info = None
            for result in results:
                if result['verified'] and result.get('info'):
                    detected_info = result['info']
                    break
            
            self.update_live_person_info(detected_info)
            
            # Draw results
            frame = self.face_recognizer.draw_results(frame, results)
            
            # Convert to PhotoImage
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = img.resize((640, 480), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image=img)
            
            self.video_label.config(image=photo)
            self.video_label.image = photo
        
        # Schedule next update
        self.root.after(self.frame_interval_ms, self.update_camera)
    
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
        
        age_text = self.age_entry.get().strip() if self.age_entry else ""
        try:
            age = int(age_text)
            if age <= 0:
                raise ValueError
        except (ValueError, TypeError):
            messagebox.showwarning("Warning", "Please enter a valid age")
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
            rank=rank, age=age, has_permission=has_permission
        )
        
        if success:
            # Update recognizer
            self.face_recognizer.update_embeddings()
            self.refresh_person_list()
            self.log(f"✓ Successfully added {name}")
            messagebox.showinfo("Success", f"Successfully added {name}")
            self.name_entry.delete(0, tk.END)
            self.rank_entry.delete(0, tk.END)
            self.age_entry.delete(0, tk.END)
            self.permission_var.set("Yes")
            if self.permission_combo:
                self.permission_combo.set("Yes")
        else:
            self.log(f"✗ Failed to add {name}")
            messagebox.showerror("Error", f"Failed to add {name}")
        
        # Restart camera if it was running
        if was_running:
            self.root.after(100, self.start_camera)
    
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
                    f"Age: {info.get('age','N/A')}\n"
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
            f"Age: {info.get('age','N/A')}\n"
            f"Has Permission: {permission_text}\n"
            f"Images: {info.get('num_images','N/A')}\n"
            f"Embeddings: {info.get('num_embeddings','N/A')}\n"
            f"Directory: {info.get('directory','')}\n"
        )
        metadata = info.get('metadata', {})
        if metadata:
            message += "\nMetadata:\n"
            for key, value in metadata.items():
                message += f"  • {key}: {value}\n"
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
        
        message = f"Attendance - {datetime.now().strftime('%Y-%m-%d')}\n\n"
        
        for record in records:
            message += f"{record['Name']:<15} {record['Time']:<10} ({record['Similarity']})\n"
        
        message += f"\nTotal: {len(records)} records"
        
        self.log("Showing today's attendance")
        messagebox.showinfo("Today's Attendance", message)
    
    def export_attendance(self):
        """Export attendance to file"""
        stats = self.attendance_logger.get_stats()
        message = f"Attendance file location:\n{self.attendance_logger.attendance_file}\n\n"
        message += f"Total records: {stats['total_records']}\n"
        message += f"Total people: {stats['total_people']}\n"
        message += f"Today's records: {stats['today_records']}"
        
        self.log("Attendance file info displayed")
        messagebox.showinfo("Export Attendance", message)
    
    def run(self):
        """Run the GUI"""
        self.log("System initialized")
        self.log(f"Known people: {len(self.dataset_manager.list_people())}")
        self.root.mainloop()
    
    def __del__(self):
        """Cleanup"""
        if self.cap:
            self.cap.release()

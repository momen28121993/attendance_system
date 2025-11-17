"""
Attendance Logger Module
Handles attendance tracking and logging
"""

import csv
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict
import cv2

class AttendanceLogger:
    """Log attendance with timestamps and optional photos"""
    
    def __init__(self, attendance_file: Path, log_interval=300, save_photos=True):
        """
        Args:
            attendance_file: Path to CSV attendance log
            log_interval: Minimum seconds between logs for same person
            save_photos: Whether to save face photos with attendance
        """
        self.attendance_file = Path(attendance_file)
        self.log_interval = log_interval
        self.save_photos = save_photos
        
        # Create photos directory
        self.photos_dir = self.attendance_file.parent / "photos"
        if self.save_photos:
            self.photos_dir.mkdir(exist_ok=True)
        
        # Track recent logs to prevent duplicates
        self.recent_logs: Dict[str, float] = {}
        
        # Initialize CSV file
        self._initialize_csv()
    
    def _initialize_csv(self):
        """Initialize CSV file with headers if it doesn't exist"""
        if not self.attendance_file.exists():
            with open(self.attendance_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'Name', 
                    'Date', 
                    'Time', 
                    'Timestamp', 
                    'Similarity',
                    'Photo'
                ])
            print(f"✓ Created attendance log: {self.attendance_file}")
    
    def can_log(self, name: str) -> bool:
        """
        Check if enough time has passed to log this person again
        Args:
            name: Person's name
        Returns:
            True if logging is allowed
        """
        current_time = time.time()
        last_log_time = self.recent_logs.get(name, 0)
        
        if current_time - last_log_time >= self.log_interval:
            return True
        return False
    
    def log_attendance(self, name: str, similarity: float, 
                      face_image: Optional[object] = None) -> bool:
        """
        Log attendance for a person
        Args:
            name: Person's name
            similarity: Similarity score from verification
            face_image: Face image (numpy array) to save
        Returns:
            True if logged successfully
        """
        # Check if can log
        if not self.can_log(name):
            return False
        
        # Get timestamp
        current_time = time.time()
        dt = datetime.fromtimestamp(current_time)
        date_str = dt.strftime('%Y-%m-%d')
        time_str = dt.strftime('%H:%M:%S')
        
        # Save photo if provided
        photo_filename = None
        if self.save_photos and face_image is not None:
            photo_filename = f"{name}_{dt.strftime('%Y%m%d_%H%M%S')}.jpg"
            photo_path = self.photos_dir / photo_filename
            cv2.imwrite(str(photo_path), face_image)
        
        # Write to CSV
        try:
            with open(self.attendance_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    name,
                    date_str,
                    time_str,
                    current_time,
                    f"{similarity:.4f}",
                    photo_filename or ''
                ])
            
            # Update recent logs
            self.recent_logs[name] = current_time
            
            print(f"✓ Logged attendance: {name} at {time_str} (similarity: {similarity:.4f})")
            return True
        
        except Exception as e:
            print(f"❌ Error logging attendance: {e}")
            return False
    
    def get_today_attendance(self) -> list:
        """Get today's attendance records"""
        today = datetime.now().strftime('%Y-%m-%d')
        records = []
        
        try:
            with open(self.attendance_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['Date'] == today:
                        records.append(row)
        except Exception as e:
            print(f"⚠ Error reading attendance: {e}")
        
        return records
    
    def get_attendance_by_date(self, date: str) -> list:
        """
        Get attendance records for a specific date
        Args:
            date: Date string in format 'YYYY-MM-DD'
        """
        records = []
        
        try:
            with open(self.attendance_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['Date'] == date:
                        records.append(row)
        except Exception as e:
            print(f"⚠ Error reading attendance: {e}")
        
        return records
    
    def get_person_attendance(self, name: str) -> list:
        """Get all attendance records for a person"""
        records = []
        
        try:
            with open(self.attendance_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['Name'] == name:
                        records.append(row)
        except Exception as e:
            print(f"⚠ Error reading attendance: {e}")
        
        return records
    
    def get_all_attendance(self) -> list:
        """Get all attendance records"""
        records = []
        
        try:
            with open(self.attendance_file, 'r') as f:
                reader = csv.DictReader(f)
                records = list(reader)
        except Exception as e:
            print(f"⚠ Error reading attendance: {e}")
        
        return records
    
    def get_stats(self) -> Dict:
        """Get attendance statistics"""
        all_records = self.get_all_attendance()
        today_records = self.get_today_attendance()
        
        # Count unique people
        all_people = set(record['Name'] for record in all_records)
        today_people = set(record['Name'] for record in today_records)
        
        stats = {
            'total_records': len(all_records),
            'total_people': len(all_people),
            'today_records': len(today_records),
            'today_people': len(today_people),
            'today_date': datetime.now().strftime('%Y-%m-%d')
        }
        
        return stats
    
    def print_today_attendance(self):
        """Print today's attendance in formatted table"""
        records = self.get_today_attendance()
        
        if not records:
            print("\nNo attendance records for today")
            return
        
        print(f"\n{'='*70}")
        print(f"ATTENDANCE - {datetime.now().strftime('%Y-%m-%d')}")
        print(f"{'='*70}")
        print(f"{'Name':<20} {'Time':<15} {'Similarity':<12}")
        print(f"{'-'*70}")
        
        for record in records:
            print(f"{record['Name']:<20} {record['Time']:<15} {record['Similarity']:<12}")
        
        print(f"{'='*70}")
        print(f"Total: {len(records)} records, {len(set(r['Name'] for r in records))} people")
    
    def clear_recent_logs(self):
        """Clear recent logs cache (useful for testing)"""
        self.recent_logs.clear()
        print("✓ Cleared recent logs cache")

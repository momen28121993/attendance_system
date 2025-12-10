# Build and Run (macOS and Windows)

The app is plain Python and runs on either macOS or Windows. Build binaries with PyInstaller if you want a double‑clickable app; otherwise run the scripts directly.

## 1) Prerequisites
- Python 3.10 or 3.11 (64-bit) with `pip`
- A webcam and permission to access it
- Optional: Git for cloning, `ffmpeg` for video debugging

### macOS specifics
- Install Homebrew if missing: `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`
- `brew install python@3.11 libomp cmake` (libomp helps `faiss-cpu`; cmake is sometimes required by OpenCV/mediapipe wheels)
- Allow Terminal to use the camera: System Settings → Privacy & Security → Camera

### Windows specifics
- Install Python 3.10/3.11 (64-bit) from python.org and check “Add Python to PATH”
- If you’re offline, the repo includes `insightface-0.7.3-cp310-cp310-win_amd64.whl`; install it with `pip install insightface-0.7.3-cp310-cp310-win_amd64.whl`
- Ensure you have the Microsoft Visual C++ runtime (most modern Windows installs already do)

## 2) Create a virtual environment
From the project root:
```bash
python -m venv .venv
```
- macOS: `source .venv/bin/activate`
- Windows (PowerShell): `.venv\\Scripts\\Activate.ps1`

## 3) Install dependencies
```bash
pip install --upgrade pip
pip install -r face_attendance_requirements.txt
```
- If `faiss-cpu` fails on macOS, run `brew install libomp` then retry.
- The first `insightface` run will download the `buffalo_l` model to `~/.insightface/models/`; you can point `Config.EMBEDDING_MODEL` to a local ONNX if you prefer.

## 4) Run the app
```bash
python face_attendance_run_gui.py
```
Data folders are created under `data/` on first launch. Use the GUI to add people, capture samples, and log attendance.

## 5) Build standalone binaries with PyInstaller
Install the builder:
```bash
pip install pyinstaller
```

Run the build (per OS; build on the OS you target):
- macOS:
```bash
pyinstaller --onefile --windowed --name face-attendance \
  --add-data "model:model" \
  face_attendance_run_gui.py
```
- Windows (note the `;` in `--add-data`):
```bash
pyinstaller --onefile --windowed --name face-attendance ^
  --add-data "model;model" ^
  face_attendance_run_gui.py
```

Artifacts live in `dist/`. On macOS you may need to right-click → Open the first time because the app is unsigned. On Windows, double-click `dist\\face-attendance.exe` to start the GUI.

## Anti-spoofing (SilentFace)
- Torch is listed in `face_attendance_requirements.txt`; ensure it is installed for liveness checks.
- The first run will download the ~2MB SilentFace weight to `model/antispoof/`. The file is ignored by Git.

# 🎯 FaceAttend — Smart Face Attendance System

A real-time face recognition attendance system built with **OpenCV**, **Python**, and **Streamlit**. Register people using your webcam, auto-train the model, and mark attendance automatically — all through a clean web UI.

---

## 📸 Features

- **Register People** — collect face images live from webcam
- **Auto-Train** — model trains automatically right after registration, no manual step
- **Live Attendance** — real-time face recognition marks attendance with date & time
- **Attendance Log** — filter records by date or person, export as CSV
- **Top Navbar UI** — clean dark interface, no sidebar issues

---

## 🗂️ Project Structure

```
face_attendance_system/
│
├── app.py              # Streamlit UI — main entry point
├── collect_faces.py    # Collector class — captures face images from webcam
├── train_model.py      # TrainModel class — trains LBPH recognizer
├── recognize.py        # Recognizer class — real-time face recognition
│
├── haar_face.xml       # Haar cascade for face detection (download separately)
├── trainer.yml         # Auto-generated after first registration
├── labels.pkl          # Auto-generated after first registration
├── attendance.csv      # Auto-created on first attendance mark
│
└── dataset/
    └── {name}/         # Auto-created per registered person
        ├── 0.jpg
        ├── 1.jpg
        └── ...
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/face-attendance-system.git
cd face-attendance-system
```

### 2. Install dependencies

```bash
pip install streamlit opencv-contrib-python pandas numpy
```

> ⚠️ Make sure you install `opencv-contrib-python` and **not** plain `opencv-python` — the LBPH face recognizer is only in the contrib package.

### 3. Download the Haar Cascade file

Download `haarcascade_frontalface_default.xml` from the OpenCV GitHub repo and rename it to `haar_face.xml` in the project folder:

```bash
wget https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml -O haar_face.xml
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

Then open your browser at `http://localhost:8501`

---

## 🧭 How to Use

### Step 1 — Register a Person
1. Click **📸 Register** in the top navbar
2. Enter the person's name
3. Set the number of images to collect (default: 100)
4. Click **▶ Start Collection & Auto-Train**
5. Look at the webcam — face images will be collected automatically
6. Once collection finishes, the model trains automatically
7. Done! Click **Go to Live Attendance**

### Step 2 — Mark Attendance
1. Click **🎯 Live Attendance** in the navbar
2. Click **▶ Start Camera**
3. The system recognizes registered faces and marks attendance with date & time
4. Camera runs for ~10 seconds per click — click Start again to continue
5. Click **⏹ Stop Camera** when done

### Step 3 — View Records
1. Click **📋 Attendance Log**
2. Filter by date or person
3. Click **⬇ Export CSV** to download records

---

## 🏗️ Architecture

```
collect_faces.py          train_model.py
  Collector class    →      TrainModel class
  - Opens webcam            - Reads dataset/
  - Detects faces           - Trains LBPH recognizer
  - Saves grayscale         - Saves trainer.yml
    100x100 crops             and labels.pkl

recognize.py
  Recognizer class
  - Loads trainer.yml + labels.pkl
  - get_frame() → returns annotated frame + newly marked names
  - check()     → prevents duplicate marking same day
  - release()   → closes webcam cleanly
```

**`app.py` coordinates all three** — it calls `Collector` for registration, `TrainModel` for auto-training, and `Recognizer.get_frame()` frame-by-frame for live attendance.

---

## 📦 Dependencies

| Package | Version | Purpose |
|---|---|---|
| `streamlit` | ≥ 1.28 | Web UI |
| `opencv-contrib-python` | ≥ 4.8 | Face detection & LBPH recognition |
| `pandas` | ≥ 2.0 | Attendance CSV handling |
| `numpy` | ≥ 1.24 | Image array operations |

---

## 🔧 Configuration

| Variable | File | Default | Description |
|---|---|---|---|
| `CASCADE_FILE` | `app.py` | `haar_face.xml` | Path to Haar cascade |
| `DATASET_PATH` | `app.py` | `dataset/` | Folder for face images |
| `TRAINER_FILE` | `app.py` | `trainer.yml` | Saved LBPH model |
| `LABELS_FILE` | `app.py` | `labels.pkl` | Label → name mapping |
| `ATTENDANCE_FILE` | `app.py` | `attendance.csv` | Attendance records |
| confidence threshold | `recognize.py` | `< 70` | Recognition strictness (lower = stricter) |

---

## ⚠️ Known Limitations

- Works best with **good lighting** and a **forward-facing camera**
- One person per frame gives the most accurate results
- Camera runs in **300-frame batches** (~10 seconds) in Streamlit — click Start again to continue
- Attendance is marked **once per person per day** — re-running the same day won't duplicate

---

## 🚀 Future Improvements

- [ ] Multi-camera support
- [ ] Export attendance as Excel (`.xlsx`)
- [ ] Email report at end of day
- [ ] Deep learning model (FaceNet / dlib) for higher accuracy
- [ ] Admin login and access control


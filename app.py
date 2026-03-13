import streamlit as st
import cv2 as cv
import pandas as pd
import os
import csv
import datetime
from train_model import TrainModel
from recognize import Recognizer

# ─────────────────────────────────────────────
#  PAGE CONFIG  — sidebar completely collapsed
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="FaceAttend",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
for key, val in {
    "page": "dashboard",
    "marked_today": set(),
    "camera_on": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
ATTENDANCE_FILE = "attendance.csv"
DATASET_PATH    = "dataset"
TRAINER_FILE    = "trainer.yml"
LABELS_FILE     = "labels.pkl"
CASCADE_FILE    = "haar_face.xml"

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def get_stats():
    people = 0
    if os.path.exists(DATASET_PATH):
        for f in os.listdir(DATASET_PATH):
            if os.path.isdir(os.path.join(DATASET_PATH, f)):
                people += 1
    today_count = 0
    if os.path.exists(ATTENDANCE_FILE):
        try:
            df = pd.read_csv(ATTENDANCE_FILE)
            today = datetime.datetime.now().strftime("%Y-%m-%d")
            today_count = len(df[df["Date"] == today])
        except Exception:
            pass
    return people, today_count

def model_exists():
    return os.path.exists(TRAINER_FILE) and os.path.exists(LABELS_FILE)

def ensure_csv():
    if not os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "w", newline="") as f:
            csv.writer(f).writerow(["Name", "Date", "Time"])

def nav_to(page):
    st.session_state.page = page
    st.rerun()

# ─────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Global ── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"] {
    background-color: #07070f !important;
    font-family: 'Space Grotesk', sans-serif;
    color: #cccce0;
}

/* ── Hide ALL sidebar chrome for good ── */
[data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"],
button[kind="header"] { display: none !important; }

#MainMenu, footer, header { visibility: hidden; }

/* ── Remove default top padding ── */
[data-testid="stAppViewBlockContainer"] {
    padding-top: 1rem !important;
}
[data-testid="stMainBlockContainer"] {
    padding-top: 0 !important;
    max-width: 1200px;
}

/* ── Top navbar wrapper ── */
.navbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #0d0d1f;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 0.7rem 1.4rem;
    margin-bottom: 1.4rem;
    position: relative;
    overflow: hidden;
}
.navbar::before {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(108,93,211,0.4), transparent);
}
.navbar-logo {
    font-size: 1.2rem;
    font-weight: 700;
    color: #f0f0ff;
    letter-spacing: -0.03em;
    white-space: nowrap;
}
.navbar-logo em { font-style: normal; color: #9c8ef0; }

/* ── Nav buttons — active vs inactive ── */
.stButton > button {
    background: transparent !important;
    color: #55556a !important;
    border: 1px solid transparent !important;
    border-radius: 9px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.84rem !important;
    font-weight: 500 !important;
    padding: 0.42rem 1rem !important;
    width: 100% !important;
    transition: all 0.15s !important;
    box-shadow: none !important;
    letter-spacing: 0.01em !important;
}
.stButton > button:hover {
    background: rgba(108,93,211,0.1) !important;
    color: #c4beff !important;
    border-color: rgba(108,93,211,0.2) !important;
    transform: none !important;
    box-shadow: none !important;
}

/* Active nav button — wrap in .nav-active div */
.nav-active .stButton > button {
    background: rgba(108,93,211,0.16) !important;
    color: #c4beff !important;
    border-color: rgba(108,93,211,0.35) !important;
    font-weight: 600 !important;
}

/* Action buttons inside pages (not nav) */
.action-btn .stButton > button {
    background: linear-gradient(135deg, #6c5dd3, #4a3db0) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 0.6rem 1.2rem !important;
    box-shadow: 0 4px 14px rgba(108,93,211,0.28) !important;
}
.action-btn .stButton > button:hover {
    box-shadow: 0 6px 20px rgba(108,93,211,0.48) !important;
    transform: translateY(-1px) !important;
}

/* Download button */
.stDownloadButton > button {
    background: rgba(108,93,211,0.1) !important;
    color: #c4beff !important;
    border: 1px solid rgba(108,93,211,0.3) !important;
    border-radius: 10px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 500 !important;
    box-shadow: none !important;
}

/* ── Cards ── */
.card {
    background: #0d0d1f;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 1.5rem 1.7rem;
    margin-bottom: 1.1rem;
    position: relative;
    overflow: hidden;
}
.card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(108,93,211,0.5), transparent);
}
.card-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    font-weight: 500;
    color: #6c5dd3;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-bottom: 1.1rem;
    padding-bottom: 0.65rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}

/* ── Stat cards ── */
.stats-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.9rem;
    margin-bottom: 1.4rem;
}
.stat-card {
    background: #0d0d1f;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 1.3rem 1rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.stat-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 20%; right: 20%; height: 1px;
    background: linear-gradient(90deg, transparent, #6c5dd3, transparent);
}
.stat-n {
    font-size: 2.3rem; font-weight: 700;
    background: linear-gradient(135deg, #fff 20%, #c4beff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1;
}
.stat-l {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.58rem; color: #44445a;
    text-transform: uppercase; letter-spacing: 0.12em; margin-top: 0.35rem;
}

/* ── Person cards ── */
.people-grid { display: flex; flex-wrap: wrap; gap: 0.7rem; }
.p-card {
    background: #111128; border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px; padding: 0.9rem 0.8rem;
    text-align: center; min-width: 88px;
    transition: border-color 0.18s, transform 0.18s;
}
.p-card:hover { border-color: rgba(108,93,211,0.4); transform: translateY(-2px); }
.p-av {
    width: 36px; height: 36px; border-radius: 50%;
    background: linear-gradient(135deg, #6c5dd3, #4a3db0);
    margin: 0 auto 0.45rem;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.95rem; font-weight: 700; color: #fff;
}
.p-name { font-size: 0.78rem; font-weight: 600; color: #f0f0ff; }
.p-sub  { font-family: 'JetBrains Mono', monospace; font-size: 0.6rem;
          color: #44445a; margin-top: 2px; }

/* ── Marked list ── */
.marked-item {
    display: flex; align-items: center; gap: 9px;
    padding: 0.5rem 0; border-bottom: 1px solid rgba(255,255,255,0.05);
    font-size: 0.84rem; color: #f0f0ff; font-weight: 500;
}
.marked-av {
    width: 27px; height: 27px; border-radius: 50%;
    background: linear-gradient(135deg, #23d18b, #18a070);
    display: flex; align-items: center; justify-content: center;
    font-size: 0.7rem; font-weight: 700; color: #fff; flex-shrink: 0;
}

/* ── Alerts ── */
.a-ok  { background:rgba(35,209,139,0.07); border:1px solid rgba(35,209,139,0.2);
          border-left:3px solid #23d18b; border-radius:10px;
          padding:0.85rem 1rem; color:#7fe8c0; font-size:0.8rem; margin:0.5rem 0; }
.a-err { background:rgba(240,79,106,0.07); border:1px solid rgba(240,79,106,0.2);
          border-left:3px solid #f04f6a; border-radius:10px;
          padding:0.85rem 1rem; color:#f9a0b0; font-size:0.8rem; margin:0.5rem 0; }
.a-inf { background:rgba(108,93,211,0.07); border:1px solid rgba(108,93,211,0.2);
          border-left:3px solid #6c5dd3; border-radius:10px;
          padding:0.85rem 1rem; color:#c4beff; font-size:0.8rem; margin:0.5rem 0; }

/* ── Form widgets ── */
.stTextInput input {
    background: #111128 !important; border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important; color: #f0f0ff !important;
    font-family: 'Space Grotesk', sans-serif !important; font-size: 0.88rem !important;
}
.stTextInput input:focus {
    border-color: #6c5dd3 !important;
    box-shadow: 0 0 0 3px rgba(108,93,211,0.15) !important;
}
.stTextInput label { color: #55556a !important; font-size: 0.78rem !important; }
.stSlider > div > div > div { background: #6c5dd3 !important; }
.stSlider label { color: #55556a !important; font-size: 0.78rem !important; }
.stProgress > div > div {
    background: linear-gradient(90deg, #6c5dd3, #9c8ef0) !important;
    border-radius: 4px !important;
}
.stSelectbox > div > div {
    background: #111128 !important; border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important; color: #f0f0ff !important;
}
.stSelectbox label { color: #55556a !important; font-size: 0.78rem !important; }
[data-testid="stDataFrame"] {
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 12px !important; overflow: hidden !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  TOP NAVBAR
# ─────────────────────────────────────────────
NAV_ITEMS = [
    ("dashboard",       "📊  Dashboard"),
    ("register",        "📸  Register"),
    ("live_attendance", "🎯  Live Attendance"),
    ("attendance_log",  "📋  Attendance Log"),
]

# Logo + nav in one row
logo_col, *nav_cols, spacer = st.columns([2, 1, 1, 1, 1, 1])

with logo_col:
    st.markdown(
        "<div class='navbar-logo' style='padding:0.42rem 0;'>"
        "Face<em>Attend</em></div>",
        unsafe_allow_html=True
    )

for col, (page_key, label) in zip(nav_cols, NAV_ITEMS):
    with col:
        is_active = st.session_state.page == page_key
        if is_active:
            st.markdown("<div class='nav-active'>", unsafe_allow_html=True)
        if st.button(label, key=f"nav_{page_key}"):
            nav_to(page_key)
        if is_active:
            st.markdown("</div>", unsafe_allow_html=True)

# Thin separator line
st.markdown(
    "<hr style='border:none;border-top:1px solid rgba(255,255,255,0.06);margin:0 0 1.4rem 0;'>",
    unsafe_allow_html=True
)


# ═══════════════════════════════════════════════════════
#  DASHBOARD
# ═══════════════════════════════════════════════════════
if st.session_state.page == "dashboard":
    people, today_count = get_stats()
    model_sym   = "✓" if model_exists() else "✗"
    model_color = "color:#23d18b;-webkit-text-fill-color:#23d18b" \
                  if model_exists() else "color:#f04f6a;-webkit-text-fill-color:#f04f6a"

    st.markdown(f"""
    <div class="stats-row">
        <div class="stat-card">
            <div class="stat-n">{people}</div>
            <div class="stat-l">Registered People</div>
        </div>
        <div class="stat-card">
            <div class="stat-n">{today_count}</div>
            <div class="stat-l">Marked Today</div>
        </div>
        <div class="stat-card">
            <div class="stat-n" style="font-size:1.8rem;{model_color}">{model_sym}</div>
            <div class="stat-l">Model Status</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_r = st.columns([1, 1.6])

    with col_l:
        st.markdown('<div class="card"><div class="card-label">⚡ Quick Actions</div>', unsafe_allow_html=True)
        st.markdown("<div class='action-btn'>", unsafe_allow_html=True)
        if st.button("📸   Register New Person", key="dash_reg"):
            nav_to("register")
        st.markdown("</div><div style='height:0.4rem'></div><div class='action-btn'>", unsafe_allow_html=True)
        if st.button("🎯   Start Live Attendance", key="dash_live"):
            nav_to("live_attendance")
        st.markdown("</div><div style='height:0.4rem'></div><div class='action-btn'>", unsafe_allow_html=True)
        if st.button("📋   View Attendance Log", key="dash_log"):
            nav_to("attendance_log")
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="card"><div class="card-label">🕐 Recent Attendance</div>', unsafe_allow_html=True)
        if os.path.exists(ATTENDANCE_FILE):
            try:
                df = pd.read_csv(ATTENDANCE_FILE)
                if len(df) > 0:
                    st.dataframe(df.tail(8)[["Name","Date","Time"]],
                                 use_container_width=True, hide_index=True)
                else:
                    st.markdown('<div class="a-inf">No records yet.</div>', unsafe_allow_html=True)
            except Exception:
                st.markdown('<div class="a-err">Could not load records.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="a-inf">No attendance file yet.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Registered people
    st.markdown('<div class="card"><div class="card-label">👥 Registered People</div>', unsafe_allow_html=True)
    if os.path.exists(DATASET_PATH):
        persons = [p for p in os.listdir(DATASET_PATH)
                   if os.path.isdir(os.path.join(DATASET_PATH, p))]
        if persons:
            cards_html = "".join([
                f'<div class="p-card">'
                f'<div class="p-av">{p[0].upper()}</div>'
                f'<div class="p-name">{p}</div>'
                f'<div class="p-sub">{len(os.listdir(os.path.join(DATASET_PATH, p)))} imgs</div>'
                f'</div>'
                for p in persons
            ])
            st.markdown(f'<div class="people-grid">{cards_html}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="a-inf">No people registered yet.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
#  REGISTER  — collect faces + AUTO-TRAIN
# ═══════════════════════════════════════════════════════
elif st.session_state.page == "register":

    st.markdown('<div class="card"><div class="card-label">📸 New Registration</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns([1, 1])
    with col_a:
        name = st.text_input("Full Name", placeholder="e.g. Upendra")
    with col_b:
        num_images = st.slider("Images to collect", 30, 150, 100, 10)
    st.markdown("<div class='action-btn'>", unsafe_allow_html=True)
    start = st.button("▶  Start Collection & Auto-Train")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if start:
        if not name.strip():
            st.markdown('<div class="a-err">⚠ Please enter a name.</div>', unsafe_allow_html=True)
        elif not os.path.exists(CASCADE_FILE):
            st.markdown('<div class="a-err">⚠ haar_face.xml not found in project folder.</div>',
                        unsafe_allow_html=True)
        else:
            folder = os.path.join(DATASET_PATH, name.strip())
            os.makedirs(folder, exist_ok=True)
            haar = cv.CascadeClassifier(CASCADE_FILE)
            cap  = cv.VideoCapture(0)

            if not cap.isOpened():
                st.markdown('<div class="a-err">⚠ Cannot open camera.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="card"><div class="card-label">📷 Step 1 — Collecting</div>',
                            unsafe_allow_html=True)
                frame_ph  = st.image([])
                prog      = st.progress(0)
                status_ph = st.empty()
                st.markdown('</div>', unsafe_allow_html=True)

                count = 0
                while count < num_images:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    gray  = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                    faces = haar.detectMultiScale(gray, 1.15, 3)
                    for (x, y, w, h) in faces:
                        crop = cv.resize(cv.equalizeHist(gray[y:y+h, x:x+w]), (100, 100))
                        cv.imwrite(f"{folder}/{len(os.listdir(folder))}.jpg", crop)
                        cv.rectangle(frame, (x, y), (x+w, y+h), (108, 93, 211), 2)
                        count += 1
                    cv.putText(frame, f"{name.strip()}  [{count}/{num_images}]",
                               (12, 34), cv.FONT_HERSHEY_SIMPLEX, 0.8, (108, 93, 211), 2)
                    frame_ph.image(cv.cvtColor(frame, cv.COLOR_BGR2RGB),
                                   channels="RGB", use_container_width=True)
                    prog.progress(min(count / num_images, 1.0))
                    status_ph.markdown(
                        f'<div class="a-inf">📷 Collecting...  {count} / {num_images}</div>',
                        unsafe_allow_html=True)

                cap.release()
                frame_ph.empty()
                status_ph.markdown(
                    f'<div class="a-ok">✅ Collected {count} images for <b>{name.strip()}</b></div>',
                    unsafe_allow_html=True)

                # AUTO-TRAIN
                st.markdown('<div class="card"><div class="card-label">🧠 Step 2 — Training Model</div>',
                            unsafe_allow_html=True)
                train_ph = st.empty()
                train_ph.markdown('<div class="a-inf">⏳ Training model, please wait...</div>',
                                  unsafe_allow_html=True)
                try:
                    TrainModel().train()
                    train_ph.markdown(
                        '<div class="a-ok">✅ Model trained! Ready for Live Attendance.</div>',
                        unsafe_allow_html=True)
                    st.balloons()
                except Exception as e:
                    train_ph.markdown(f'<div class="a-err">⚠ Training failed: {e}</div>',
                                      unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

                st.markdown("<div class='action-btn'>", unsafe_allow_html=True)
                if st.button("🎯  Go to Live Attendance"):
                    nav_to("live_attendance")
                st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
#  LIVE ATTENDANCE
# ═══════════════════════════════════════════════════════
elif st.session_state.page == "live_attendance":

    st.markdown('<div class="card"><div class="card-label">🎯 Live Recognition</div>',
                unsafe_allow_html=True)

    if not model_exists():
        st.markdown(
            '<div class="a-err">⚠ No model found. Register someone first — model trains automatically.</div>',
            unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("<div class='action-btn'>", unsafe_allow_html=True)
        if st.button("📸  Go to Register"):
            nav_to("register")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        col1, col2, _ = st.columns([1, 1, 3])
        with col1:
            st.markdown("<div class='action-btn'>", unsafe_allow_html=True)
            start_btn = st.button("▶  Start Camera")
            st.markdown("</div>", unsafe_allow_html=True)
        with col2:
            stop_btn = st.button("⏹  Stop Camera")
        st.markdown('</div>', unsafe_allow_html=True)

        if start_btn:
            st.session_state.camera_on    = True
            st.session_state.marked_today = set()
        if stop_btn:
            st.session_state.camera_on    = False
            st.session_state.marked_today = set()

        if st.session_state.camera_on:
            try:
                recognizer = Recognizer()
            except Exception as e:
                st.markdown(f'<div class="a-err">⚠ Could not load model: {e}</div>',
                            unsafe_allow_html=True)
                st.session_state.camera_on = False
                st.stop()

            ensure_csv()

            col_feed, col_panel = st.columns([1.8, 1])
            with col_feed:
                st.markdown('<div class="card"><div class="card-label">📷 Camera Feed</div>',
                            unsafe_allow_html=True)
                frame_ph = st.image([])
                st.markdown('</div>', unsafe_allow_html=True)
            with col_panel:
                st.markdown('<div class="card"><div class="card-label">✅ Marked Today</div>',
                            unsafe_allow_html=True)
                names_ph = st.empty()
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('<div class="card"><div class="card-label">◉ Status</div>',
                            unsafe_allow_html=True)
                status_ph = st.empty()
                st.markdown('</div>', unsafe_allow_html=True)

            for _ in range(300):
                if not st.session_state.camera_on:
                    break
                frame, newly_marked = recognizer.get_frame()
                if frame is None:
                    st.markdown('<div class="a-err">⚠ Camera read failed.</div>', unsafe_allow_html=True)
                    break
                for nm in newly_marked:
                    st.session_state.marked_today.add(nm)

                frame_ph.image(cv.cvtColor(frame, cv.COLOR_BGR2RGB),
                               channels="RGB", use_container_width=True)

                if st.session_state.marked_today:
                    items = "".join([
                        f'<div class="marked-item">'
                        f'<div class="marked-av">{n[0].upper()}</div>{n}'
                        f'</div>'
                        for n in sorted(st.session_state.marked_today)
                    ])
                    names_ph.markdown(items, unsafe_allow_html=True)
                    status_ph.markdown('<div class="a-ok">✅ Recognition active</div>',
                                       unsafe_allow_html=True)
                else:
                    names_ph.markdown(
                        '<div style="color:#44445a;font-size:0.8rem;padding:0.3rem 0;">'
                        'No one marked yet</div>',
                        unsafe_allow_html=True)
                    status_ph.markdown('<div class="a-inf">👀 Scanning for faces...</div>',
                                       unsafe_allow_html=True)

            recognizer.release()
            if st.session_state.camera_on:
                st.markdown('<div class="a-inf">Click ▶ Start Camera to continue.</div>',
                            unsafe_allow_html=True)
        else:
            st.markdown('<div class="a-inf">Click ▶ Start Camera to begin.</div>',
                        unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
#  ATTENDANCE LOG
# ═══════════════════════════════════════════════════════
elif st.session_state.page == "attendance_log":

    st.markdown('<div class="card"><div class="card-label">📋 Attendance Records</div>',
                unsafe_allow_html=True)

    if not os.path.exists(ATTENDANCE_FILE):
        st.markdown('<div class="a-inf">No records found yet.</div>', unsafe_allow_html=True)
    else:
        try:
            df = pd.read_csv(ATTENDANCE_FILE)
            if len(df) == 0:
                st.markdown('<div class="a-inf">File is empty.</div>', unsafe_allow_html=True)
            else:
                col1, col2 = st.columns(2)
                with col1:
                    dates    = ["All"] + sorted(df["Date"].unique().tolist(), reverse=True)
                    sel_date = st.selectbox("Filter by Date", dates)
                with col2:
                    names    = ["All"] + sorted(df["Name"].unique().tolist())
                    sel_name = st.selectbox("Filter by Person", names)

                filtered = df.copy()
                if sel_date != "All":
                    filtered = filtered[filtered["Date"] == sel_date]
                if sel_name != "All":
                    filtered = filtered[filtered["Name"] == sel_name]

                st.markdown(f"""
                <div style="display:flex;gap:0.7rem;margin:0.8rem 0 1rem;">
                    <span style="background:rgba(108,93,211,0.12);color:#c4beff;
                                 border:1px solid rgba(108,93,211,0.25);border-radius:6px;
                                 padding:0.18rem 0.65rem;font-size:0.68rem;font-weight:600;
                                 font-family:'JetBrains Mono',monospace;">TOTAL {len(df)}</span>
                    <span style="background:rgba(35,209,139,0.08);color:#7fe8c0;
                                 border:1px solid rgba(35,209,139,0.2);border-radius:6px;
                                 padding:0.18rem 0.65rem;font-size:0.68rem;font-weight:600;
                                 font-family:'JetBrains Mono',monospace;">SHOWING {len(filtered)}</span>
                </div>""", unsafe_allow_html=True)

                st.dataframe(filtered.sort_values("Date", ascending=False),
                             use_container_width=True, hide_index=True)

                st.download_button(
                    label="⬇  Export CSV",
                    data=filtered.to_csv(index=False).encode("utf-8"),
                    file_name=f"attendance_{sel_date}.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.markdown(f'<div class="a-err">⚠ Error: {e}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
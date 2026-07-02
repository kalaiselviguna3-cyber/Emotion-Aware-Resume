import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

import streamlit as st
import torch
import cv2
import numpy as np
import tempfile
import librosa
import PyPDF2
import sqlite3
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------------- DATABASE ----------------
conn = sqlite3.connect("candidates.db", check_same_thread=False)
cursor = conn.cursor()

# HR users table
cursor.execute("""
CREATE TABLE IF NOT EXISTS hr_users (
    username TEXT PRIMARY KEY,
    password TEXT
)
""")

# REMOVE OLD HR USERS & INSERT NEW CREDENTIALS
cursor.execute("""
DELETE FROM hr_users
""")

cursor.execute("""
INSERT INTO hr_users VALUES ('dhanusri','dhanu@12')
""")

# Candidate table
cursor.execute("""
CREATE TABLE IF NOT EXISTS candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    resume_confidence TEXT,
    face_emotion TEXT,
    voice_stress TEXT,
    final_score INTEGER
)
""")
conn.commit()

# ---------------- LOAD TRANSFORMER (PYTORCH ONLY) ----------------
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

# ---------------- RESUME TEXT ANALYSIS ----------------
def analyze_resume(pdf):
    reader = PyPDF2.PdfReader(pdf)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()

    inputs = tokenizer(text[:512], return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)

    prob = torch.softmax(outputs.logits, dim=1)[0][1].item()
    return ("Confident", 80) if prob > 0.6 else ("Low Confidence", 50)

# ---------------- FACE ANALYSIS ----------------
face_net = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def analyze_video(video):
    temp = tempfile.NamedTemporaryFile(delete=False)
    temp.write(video.read())

    cap = cv2.VideoCapture(temp.name)
    faces = frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frames > 100:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces += len(face_net.detectMultiScale(gray, 1.3, 5))
        frames += 1

    cap.release()

    if faces > 35:
        return "Calm & Confident", 80
    elif faces > 15:
        return "Normal", 60
    else:
        return "Nervous", 40

# ---------------- VOICE STRESS (SAFE VERSION) ----------------
def analyze_voice(video):
    try:
        temp = tempfile.NamedTemporaryFile(delete=False)
        temp.write(video.read())

        y, sr = librosa.load(temp.name, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

        if np.mean(mfcc) > -200:
            return "Low Stress", 80
        else:
            return "High Stress", 50
    except Exception:
        return "Audio Not Detected", 60

# ---------------- LOGIN ----------------
def login():
    st.title("🔐 HR Login")

    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")

    if st.button("Login"):
        cursor.execute(
            "SELECT * FROM hr_users WHERE username=? AND password=?",
            (user, pwd)
        )
        if cursor.fetchone():
            st.session_state.logged_in = True
            st.success("Login Successful")
        else:
            st.error("Invalid Username or Password")

# ---------------- DASHBOARD ----------------
def dashboard():
    st.title("Emotion-Aware Resume Screening System")

    name = st.text_input("Candidate Name")
    resume = st.file_uploader("Upload Resume (PDF)", type="pdf")
    video = st.file_uploader("Upload Video Resume (MP4)", type="mp4")

    if st.button("Analyze Candidate"):
        if name and resume and video:
            r_label, r_score = analyze_resume(resume)
            f_label, f_score = analyze_video(video)
            v_label, v_score = analyze_voice(video)

            final_score = int((r_score + f_score + v_score) / 3)

            cursor.execute(
                "INSERT INTO candidates VALUES (NULL,?,?,?,?,?)",
                (name, r_label, f_label, v_label, final_score)
            )
            conn.commit()

            st.metric("Final Candidate Score", final_score)
            st.write("Resume Confidence:", r_label)
            st.write("Facial Emotion:", f_label)
            st.write("Voice Stress:", v_label)
        else:
            st.error("Please upload all inputs")

    st.divider()
    st.subheader("📋 Stored Candidate Records")

    cursor.execute("SELECT * FROM candidates")
    for row in cursor.fetchall():
        st.write(
            f"ID:{row[0]} | Name:{row[1]} | Resume:{row[2]} | "
            f"Face:{row[3]} | Voice:{row[4]} | Score:{row[5]}"
        )

    if st.button("Logout"):
        st.session_state.logged_in = False

# ---------------- MAIN ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if st.session_state.logged_in:
    dashboard()
else:
    login()
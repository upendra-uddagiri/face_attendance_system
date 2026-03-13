import cv2 as cv
import pickle
import datetime
import csv
import os
import pandas
 
 
class Recognizer:
    def __init__(self):
        self.haar_cascade = cv.CascadeClassifier("haar_face.xml")
        self.recognizer = cv.face.LBPHFaceRecognizer_create()
        self.recognizer.read("trainer.yml")
        self.color = ()
        with open("labels.pkl", "rb") as f:
            self.label_map = pickle.load(f)
        self.capture = cv.VideoCapture(0)
        self.attendance_file = "attendance.csv"
        if not os.path.exists(self.attendance_file):
            with open(self.attendance_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Name", "Date", "Time"])
 
    def check(self, name):
        with open(self.attendance_file, "r") as f:
            data = pandas.read_csv(f)
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        # Fixed bug: now checks BOTH name AND today's date
        already = ((data["Name"] == name) & (data["Date"] == today)).any()
        return not already  # True = not marked yet
 
    def get_frame(self):
        """
        Called by Streamlit frame by frame.
        Returns:
            frame      : annotated BGR frame (to display)
            names_marked: list of names marked in this frame
        """
        names_marked = []
        ret, frame = self.capture.read()
        if not ret:
            return None, names_marked
 
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces_rect = self.haar_cascade.detectMultiScale(gray, 1.5, 3)
 
        for (x, y, w, h) in faces_rect:
            face = gray[y:y + h, x:x + w]
            face = cv.resize(face, (100, 100))
            face = cv.equalizeHist(face)
 
            label, confidence = self.recognizer.predict(face)
            accuracy = 100 - confidence
            name = self.label_map[label]
 
            now  = datetime.datetime.now()
            date = now.strftime("%Y-%m-%d")
            time = now.strftime("%H:%M:%S")
 
            if confidence < 70:
                self.color = (99, 241, 130)   # green
                if self.check(name):
                    with open(self.attendance_file, "a", newline="") as f:
                        csv.writer(f).writerow([name, date, time])
                    names_marked.append(name)
                display_name = f"{name} ✓  {round(accuracy, 1)}%"
            else:
                self.color = (239, 68, 68)    # red
                display_name = f"Unknown  {round(accuracy, 1)}%"
 
            cv.rectangle(frame, (x, y), (x + w, y + h), self.color, 2)
            cv.putText(frame, display_name,
                       (x, y - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.75, self.color, 2)
            cv.putText(frame, f"{date}  {time}",
                       (x, y + h + 22),
                       cv.FONT_HERSHEY_SIMPLEX, 0.55, self.color, 2)
 
        return frame, names_marked
 
    def release(self):
        """Release webcam when done."""
        self.capture.release()
 
    def recognize(self):
        """
        Original standalone mode (works without Streamlit).
        Press 'd' to quit.
        """
        while True:
            frame, _ = self.get_frame()
            if frame is None:
                break
            cv.imshow("Face Attendance System", frame)
            if cv.waitKey(1) & 0xFF == ord('d'):
                break
        self.release()
        cv.destroyAllWindows()
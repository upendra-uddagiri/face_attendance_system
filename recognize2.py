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
        self.color=()
        with open("labels.pkl", "rb") as f:
            self.label_map = pickle.load(f)
        self.capture = cv.VideoCapture(0)
        self.attendance_file = "attendance.csv"
        if not os.path.exists(self.attendance_file):
            with open(self.attendance_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Name", "Date", "Time"])
                f.close()
    def check(self,name):
        with open(self.attendance_file,"r") as f:
            data=pandas.read_csv(f)
            f.close()
        if name in data["Name"].values:
            return False
        else:
            return True
    def recognize(self):
        while True:
            ret, frame = self.capture.read()
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            faces_rect = self.haar_cascade.detectMultiScale(gray, 1.5, 3)
            for (x, y, w, h) in faces_rect:
                face = gray[y:y + h, x:x + w]
                face = cv.resize(face, (100, 100))
                face = cv.equalizeHist(face)
                label, confidence = self.recognizer.predict(face)
                accuracy = 100 - confidence
                name = self.label_map[label]
                now = datetime.datetime.now()
                date = now.strftime("%Y-%m-%d")
                time = now.strftime("%H:%M:%S")
                if confidence < 70:
                    self.color=(0,255,0)
                    if self.check(name):
                        with open(self.attendance_file, "a", newline="") as f:
                                writer = csv.writer(f)
                                writer.writerow([name, date, time])
                    else:
                        print('already plotted the attendance')
                    name+=":Attendance Marked"
                else:
                    name = "Unknown"
                    self.color=(0,0,255)
                cv.rectangle(frame, (x, y), (x + w, y + h),self.color, 2)
                cv.putText(frame,
                           f"{name}  {round(accuracy, 2)}",
                           (x, y - 10),
                           cv.FONT_HERSHEY_SIMPLEX,
                           0.9,
                                   self.color,
                           2)
                cv.putText(frame,
                           f"{date} {time}",
                           (x, y + h + 20),
                           cv.FONT_HERSHEY_SIMPLEX,
                           0.6,
                           self.color,
                           2)
            cv.imshow("Face Attendance System", frame)
            if cv.waitKey(1) & 0xFF == ord('d'):
                break
        self.capture.release()
        cv.destroyAllWindows()

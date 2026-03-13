import cv2 as cv
import os
class Collector:
    def __init__(self):
        self.haar_cascade = cv.CascadeClassifier('haar_face.xml')
        self.capture = cv.VideoCapture(0)
        self.count=0
    def collect(self):
        name = input("Enter your name: ")
        folder_name = f"dataset/{name}"
        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)
            print(f"Folder '{folder_name}' created.")
            while self.count < 100:
                istrue, frame = self.capture.read()
                img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

                faces_rect = self.haar_cascade.detectMultiScale(
                    img,
                    scaleFactor=1.15,
                    minNeighbors=3
                )

                for (x, y, w, h) in faces_rect:
                    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cropped_img = img[y:y + h, x:x + w]
                    cropped_img = cv.resize(cropped_img, (100, 100), interpolation=cv.INTER_AREA)
                    cropped_img = cv.equalizeHist(cropped_img)
                    cv.imwrite(f'{folder_name}/{self.count}.jpg', cropped_img)
                    self.count += 1
                    print(f"Images: {self.count}/50")

                cv.imshow("Collecting Faces", img)
                cv.waitKey(1)
            self.capture.release()
            cv.destroyAllWindows()
        else:
            print(f"Folder '{folder_name}' already exists.")
        print("Dataset collection complete ✅")
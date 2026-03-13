import cv2 as cv
import os
import numpy as np
import pickle
class TrainModel:
    def __init__(self):
        self.dataset_path = "dataset"

        self.faces = []
        self.labels = []
        self.label_map = {}
        self.label_id = 0
    def train(self):
        for person_name in os.listdir(self.dataset_path):
            person_folder = os.path.join(self.dataset_path, person_name)

            if not os.path.isdir(person_folder):
                continue

            self.label_map[self.label_id] = person_name

            for image_name in os.listdir(person_folder):
                image_path = os.path.join(person_folder, image_name)

                img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

                if img is None:
                    continue

                self.faces.append(img)
                self.labels.append(self.label_id)

            self.label_id += 1

        faces = np.array(self.faces)
        labels = np.array(self.labels)

        recognizer = cv.face.LBPHFaceRecognizer_create()
        recognizer.train(faces, labels)

        recognizer.save("trainer.yml")

        # Save label map
        with open("labels.pkl", "wb") as f:
            pickle.dump(self.label_map, f)

        print("Training completed!")
from recognize2 import *
from collect_faces import *
from train_model import *

collect=Collector()
traim=TrainModel()
recognizer=Recognizer()
choice=input("enter yes to collect data and train model:").lower()
if choice=="yes":
  collect.collect()
  train.traimmodel()
recognize.recognize()

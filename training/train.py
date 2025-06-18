from ultralytics import YOLO
import torch


device = 0 if torch.cuda.is_available() else "cpu"
model = YOLO("weights/best.pt")

results= model.train(
                     data= "training/data.yaml", 
                     epochs= 100, 
                     imgsz= 640,
                     optimizer= "Adam",
                     device= device
                     )
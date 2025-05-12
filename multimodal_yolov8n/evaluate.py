from ultralytics import YOLO
from fusion_model import load_fusion_model
import torch

def evaluate():
    # Załaduj wytrenowany model z fuzją
    model = load_fusion_model()
    model.load_state_dict(torch.load('path/to/trained_model.pt'))
    
    # Ewaluacja na danych RGB lub IR
    results = model.val(data='configs/data.yaml')  # Lub 'configs/data_ir.yaml'
    print(results)

if __name__ == "__main__":
    evaluate()
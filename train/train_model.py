import os
import torch
from ultralytics import YOLO

# This is the "Master Key" fix for PyTorch 2.6+
# It forces the loader to allow all necessary YOLO components
import torch.serialization
torch.serialization.safe_globals = lambda *args, **kwargs: None 

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
DATA_YAML_PATH = os.path.join(BASE_DIR, "..", "model", "data.yaml")

def start_training():
    # We use weights_only=False to bypass the strict pickling checks
    model = YOLO("yolov8n.pt") 

    print(f"Starting training using: {DATA_YAML_PATH}")
    model.train(
        data=DATA_YAML_PATH,
        epochs=10, 
        imgsz=640,
        batch=16,
        name="my_custom_model"
    )

if __name__ == "__main__":
    start_training()
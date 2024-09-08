from dataclasses import dataclass
from functools import lru_cache
import os

@dataclass
class Config:
    face_yolo_path = "./yolo-models/yolov8n-face.pt"
    general_yolo_path = "./yolo-models/yolov8n.pt"
    crop_save_path = "./crops"
    
    def __post_init__(self):
        if not os.path.exists(self.crop_save_path):
            os.makedirs(self.crop_save_path)
    
    
@lru_cache(maxsize = None)   
def get_config():
    return Config()
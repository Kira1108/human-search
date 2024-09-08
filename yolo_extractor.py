import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image
from ultralytics import YOLO
from functools import lru_cache
from config import get_config

def to_base_name(fp:Union[str, Path]):
    return os.path.basename(str(fp)).split(".")[0]


@dataclass
class YoloExtractor:
    """Face Extractor extracts faces from images using yolo model desgined specifically for faces."""
    yolo: YOLO
    save_folder: Union[str, Path] = "./labelled"
    keep_entities: list = None
    
    def __post_init__(self):
        self.save_folder = Path(self.save_folder)
        
    def extract_one(self, 
                    im: Union[str, np.array,Image.Image], 
                    extract_path:str = None):
        
        results = self.yolo(im)
        
        has_crops = []
        
        for i, result in enumerate(results):
                
            # extract bounding box metadata
            boxes = result.boxes 
            masks = result.masks  
            keypoints = result.keypoints 
            probs = result.probs 
            
            if extract_path:
                result.save_crop(extract_path)
                has_crop = self.clean_directory(extract_path)
                has_crops.append(has_crop)
                
        return any(has_crops)
                        
    def clean_directory(self, directory:str) -> bool:
        directory = str(directory)
        
        # if found the directory
        if os.path.exists(directory):
            
            # loop over sub directories
            for folder in Path(directory).iterdir():
                
                # remove unwanted entities
                if folder.name not in self.keep_entities:
                    shutil.rmtree(folder)
                    
            # if save_path is empty, remove the folder
            if len(list(Path(directory).iterdir())) == 0:
                shutil.rmtree(directory)
                # nothing is left
                return False
            
            return True
        
        # folder contains entities   
        return False   
        
    
    def extract_many(self, image_fps:list) -> dict:
        
        os.makedirs(self.save_folder, exist_ok=True)
        
        metas = {}
        for im_path in image_fps:
            has_crop = self.extract_one(im_path, self.save_folder / to_base_name(im_path))
            metas[str(im_path.absolute())] = has_crop
        return metas
    
@lru_cache(maxsize = None)   
def create_face_extractor(save_folder:str = "faces"):
    config = get_config()
    model_path = config.face_yolo_path    
    model = YOLO(model_path)
    save_folder = Path(config.crop_save_path) / save_folder
    extractor = YoloExtractor(model, keep_entities=["face"], save_folder=save_folder)
    return extractor

@lru_cache(maxsize = None)   
def create_person_extractor(save_folder:str = "persons"):
    config = get_config()
    model_path = config.general_yolo_path    
    model = YOLO(model_path)
    save_folder = Path(config.crop_save_path) / save_folder
    extractor = YoloExtractor(model, keep_entities=["person"], save_folder=save_folder)
    return extractor
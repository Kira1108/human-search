from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Union

import numpy as np
import timm
import torch
from PIL import Image
from sklearn.preprocessing import normalize
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from transformers import AutoImageProcessor, AutoModel

@dataclass
class VitEncoder:
    model_name: str = 'google/vit-base-patch16-224-in21k'
    
    def __post_init__(self):
        if torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'
        
        self.processor = AutoImageProcessor.from_pretrained(self.model_name, device=self.device)
        self.model = AutoModel.from_pretrained(self.model_name)
        
    def __call__(self, img: Union[str, np.array, Image.Image]) -> np.array:
        if isinstance(img, str):
            img = Image.open(img)     
        inputs = self.processor(images=[np.array(img)], return_tensors='pt')
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        outputs = outputs.last_hidden_state[:, 0].squeeze(0).cpu().numpy()
        return normalize(outputs.reshape(1, -1), norm="l2").flatten()
        # return self.model(**inputs).last_hidden_state[:, 0].detach().cpu().numpy().tolist()[0]
    
    @classmethod
    def create(cls):
        return create_vit_encoder()

@lru_cache(maxsize = None)
def create_vit_encoder():
    return VitEncoder()

class TimmEncoder:
    def __init__(self, modelname):
        # Load the pre-trained model
        self.model = timm.create_model(
            modelname, pretrained=True, num_classes=0, global_pool="avg"
        )
        self.model.eval()

        # Get the input size required by the model
        self.input_size = self.model.default_cfg["input_size"]

        config = resolve_data_config({}, model=modelname)
        # Get the preprocessing function provided by TIMM for the model
        self.preprocess = create_transform(**config)

    def __call__(self, imagepath):
        # Preprocess the input image
        input_image = Image.open(imagepath).convert("RGB")  # Convert to RGB if needed
        input_image = self.preprocess(input_image)

        # Convert the image to a PyTorch tensor and add a batch dimension
        input_tensor = input_image.unsqueeze(0)

        # Perform inference
        with torch.no_grad():
            output = self.model(input_tensor)

        # Extract the feature vector
        feature_vector = output.squeeze().numpy()

        return normalize(feature_vector.reshape(1, -1), norm="l2").flatten()
    
    @classmethod
    def create(cls, modelname:str = "resnet34"):
        return create_timm_encoder(modelname)
   
@lru_cache(maxsize = None)
def create_timm_encoder(modelname:str = "resnet34"): 
    return TimmEncoder(modelname)
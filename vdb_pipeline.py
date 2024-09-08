from pymilvus import MilvusClient
from image_encoder import VitEncoder
from yolo_extractor import create_face_extractor,create_person_extractor
from pathlib import Path
import os
import shutil
from config import get_config
from tqdm import tqdm

def crop_images():
    
    crop_path = Path(get_config().crop_save_path).absolute()
    
    if os.path.exists(crop_path):
        shutil.rmtree(crop_path)
    
    image_folder = Path("images")
    face_extractor = create_face_extractor()
    person_extractor = create_person_extractor()
    image_fps = list(image_folder.glob("*.png")) + list(image_folder.glob("*.jpg"))
    face_extractor.extract_many(image_fps)
    person_extractor.extract_many(image_fps)

def recreate_collection():
    
    client = MilvusClient(uri="http://localhost", port="19530", user = "", password = "")

    if client.has_collection(collection_name="faces"):
        client.drop_collection(collection_name="faces")
        
    if client.has_collection(collection_name="persons"):
        client.drop_collection(collection_name="persons")
        
    client.create_collection(
        collection_name="faces",
        vector_field_name="vector",
        dimension=768,
        auto_id=True,
        enable_dynamic_field=True,
        metric_type="COSINE",
    )

    client.create_collection(
        collection_name="persons",
        vector_field_name="vector",
        dimension=768,
        auto_id=True,
        enable_dynamic_field=True,
        metric_type="COSINE",
    )

    print("Collections `faces` and `persons` created")
    
def insert_embeddings():
    vit = VitEncoder.create()
    client = MilvusClient(uri="http://localhost", port="19530", user = "", password = "")
    crop_path = Path(get_config().crop_save_path)

    faces_fpaths = (crop_path / "faces").absolute().rglob("*.jpg")
    persons_fpaths = (crop_path / "persons").absolute().rglob("*.jpg")
    
    faces_fpaths = [str(fp) for fp in faces_fpaths]
    persons_fpaths = [str(fp) for fp in persons_fpaths]

    for face in tqdm(faces_fpaths,desc = "Embedding faces..."):
        vector = vit(face)
        client.insert(collection_name="faces", data=[{"crop_path": face, "vector": vector}])
        
    for person in tqdm(persons_fpaths,desc = "Embedding persons..."):
        vector = vit(person)
        client.insert(collection_name="persons", data=[{"crop_path": person, "vector": vector}])
        
def run_pipeline():
    crop_images()
    recreate_collection()
    insert_embeddings()
    
if __name__ == "__main__":
    run_pipeline()
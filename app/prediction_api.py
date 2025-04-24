from fastapi import APIRouter, UploadFile, File
from typing import List
import zipfile
import io
import os
from app.model.model import predict_images

prediction_router = APIRouter()

def format_prediction(content: bytes, filename: str):
    return predict_images(content=content, image_name=filename, confidence=0.6)

@prediction_router.post("/predict/upload", summary="Upload ZIP or image(s) for prediction")
async def predict_images_from_upload(files: List[UploadFile] = File(...)):
    results = []

    for file in files:
        filename = file.filename.lower()

        if filename.endswith(".zip"):
            zip_content = await file.read()
            with zipfile.ZipFile(io.BytesIO(zip_content)) as zip_file:
                for zip_filename in zip_file.namelist():
                    if zip_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                        with zip_file.open(zip_filename) as image_file:
                            image_content = image_file.read()
                            results.append(format_prediction(image_content, zip_filename))
        elif filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_content = await file.read()
            results.append(format_prediction(image_content, file.filename))
        else:
            results.append({"error": f"Unsupported file: {file.filename}"})

    return results if len(results) > 1 else results[0]

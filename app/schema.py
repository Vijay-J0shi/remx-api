import strawberry
from strawberry.file_uploads import Upload

from numpy import ndarray

from typing import List, Dict, Tuple

from app.model.model import predict_images


@strawberry.type
class Query:

    @strawberry.field
    def prediction_output(self) -> str:
        return "Remx"


BBox = Tuple[int, int, int, int]  # (x, y, w, h) for single bounding box


@strawberry.type
class PredictionOutput:
    image: str
    coordinates: List[BBox]
    max_confidence_coordinate: BBox


@strawberry.input
class FolderInput:
    files: List[Upload]


@strawberry.type
class Mutation:

    @strawberry.mutation
    async def read_file(self, file: Upload) -> PredictionOutput:
        content = await file.read()  # 'bytes'
        return PredictionOutput(**predict_images(
            content=content, image_name=file.filename, confidence=0.6))

    @strawberry.mutation
    async def read_files(self, files: List[Upload]) -> List[PredictionOutput]:
        image_prediction_output = []
        for image in files:
            image_name = image.filename
            content = await image.read()  # 'bytes'
            image_prediction_output.append(
                PredictionOutput(**predict_images(content=content,
                                                  image_name=image.filename,
                                                  confidence=0.6)))
        return image_prediction_output

    @strawberry.mutation
    async def read_folder(self, folder: FolderInput) -> List[PredictionOutput]:
        image_prediction_output = []
        for image in folder.files:
            content = await image.read()  # 'bytes'
            image_prediction_output.append(
                PredictionOutput(**predict_images(content=content,
                                                  image_name=image.filename,
                                                  confidence=0.6)))
        return image_prediction_output

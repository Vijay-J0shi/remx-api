import numpy as np
from fastapi import  UploadFile

from pathlib import Path
from typing import Dict

from app.utils.images_predict_fn import (map_lb_original_img, bboxs_filter,
                                         nms, compute_iou, xywh2xyxy,
                                         model_ort_session,
                                         final_image_pre_process,
                                         letterboxed_result)

__version__ = "1.0.0"

BASE_DIR = Path(__file__).resolve(strict=True).parent
MODEL = f"{BASE_DIR}/remx_model_{__version__}.onnx"


def predict_images(content: UploadFile,
                   image_name: str,
                   confidence: float = 0.5,
                   MODEL=MODEL) -> Dict:

    if image_name.endswith(".jpg") or image_name.endswith(".png"):

        model = model_ort_session(MODEL)
        pre_process_image = final_image_pre_process(content,
                                                    model["input_shape"])

        outputs = model["session"].run(
            model["output_names"],
            {model["input_names"][0]: pre_process_image["input_tensor"]},
        )[0]

        bboxs_outputs = bboxs_filter(
            outputs,
            pre_process_image["input_width"],
            pre_process_image["input_height"],
            pre_process_image["image_width"],
            pre_process_image["image_height"],
        )

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        indices = nms(
            bboxs_outputs["boxes"],
            bboxs_outputs["scores"],
            iou_threshold=confidence,  # Threshold
        )

        letterboxed_output = letterboxed_result(
            boxes=bboxs_outputs["boxes"],
            indices=indices,
            scores=bboxs_outputs["scores"],
            class_ids=bboxs_outputs["class_ids"],
        )

        inverse_coordinate = map_lb_original_img(
            content, letterboxed_output["letterboxed_boxes"])

        return {
            "image":
            image_name,
            "coordinates":
            inverse_coordinate,
            # "labels": label,
            "max_confidence_coordinate":
            inverse_coordinate[letterboxed_output["max_score_index"]]
            if letterboxed_output["scores"] else
            (-1, -1, -1, -1),  # Negative for does exist
        }

import cv2
import numpy as np

from app.utils.images import (
    letterbox,
    ImgSize,
    inverse_letterbox_coordinate_transform,
)


def compute_iou(box, boxes):
    # compute xmin, ymin, xmax, ymax for both boxes
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    # compute intersection area
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    # compute union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area

    # compute iou
    iou = np.where(union_area > 0, intersection_area / union_area,
                   0).astype(float)

    return iou


def nms(boxes, scores, iou_threshold):
    """
    Non-maximum suppression (NMS)
    Select best bounding box out of a set of overlapping boxes.
    """

    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]

    keep_boxes = []
    while sorted_indices.size > 0:
        # Pick the last box
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)

        # Compute IoU of the picked box with the rest
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

        # Remove boxes with IoU over the threshold
        keep_indices = np.where(ious < iou_threshold)[0]

        # print(keep_indices.shape, sorted_indices.shape)
        sorted_indices = sorted_indices[keep_indices + 1]

    return keep_boxes


def xywh2xyxy(x: np.array):
    """
    yolov8 provide bounding box (x, y, w, h).
    Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
    """
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # x1
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # y1
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # x2
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # y2
    return y


def model_ort_session(MODEL: str):
    import onnxruntime as ort

    ort_session = ort.InferenceSession(MODEL)

    model_inputs = ort_session.get_inputs()
    input_names = [model_inputs[i].name for i in range(len(model_inputs))]
    input_shape = model_inputs[0].shape

    model_output = ort_session.get_outputs()
    output_names = [model_output[i].name for i in range(len(model_output))]

    return {
        "session": ort_session,
        "input_names": input_names,
        "input_shape": input_shape,
        "output_names": output_names,
    }


def final_image_pre_process(img_content, input_shape):

    # img_content: bytes

    # read the image from the byte stream
    img = cv2.imdecode(np.frombuffer(img_content, np.uint8),
                       cv2.IMREAD_UNCHANGED)  # return ndarray, original image

    # Converting original image into 640x640 size without losing its aspect ratio
    img_letterboxed = letterbox(np.asarray(img), ImgSize(640, 640))

    # Convert the np.ndarray to a byte stream
    img_bytes = cv2.imencode(".jpg", img_letterboxed)[1].tobytes()

    # read the image from the byte stream
    image = cv2.imdecode(np.frombuffer(img_bytes, np.uint8),
                         cv2.IMREAD_UNCHANGED)

    image_height, image_width = image.shape[:2]
    input_height, input_width = input_shape[2:]

    resized = cv2.resize(image, (input_width, input_height))

    # Scale input pixel value to 0 to 1
    input_image = resized / 255.0
    input_image = input_image.transpose(2, 0, 1)
    input_tensor = input_image[np.newaxis, :, :, :].astype(np.float32)

    return {
        "input_tensor": input_tensor,
        "image_height": image_height,
        "image_width": image_width,
        "input_height": input_height,
        "input_width": input_width,
    }


def bboxs_filter(outputs, input_width, input_height, image_width,
                 image_height):
    # Threshold
    predictions = np.squeeze(outputs).T
    conf_thresold = 0.5  # confidence score [testing phase]

    # Filter out object confidence scores below threshold
    scores = np.max(predictions[:, 4:], axis=1)
    predictions = predictions[scores > conf_thresold, :]
    scores = scores[scores > conf_thresold]

    # Get the class with the highest confidence
    class_ids = np.argmax(predictions[:, 4:], axis=1)

    # Get bounding boxes for each object
    boxes = predictions[:, :4]

    # rescale box
    input_shape = np.array(
        [input_width, input_height, input_width, input_height])
    boxes = np.divide(boxes, input_shape, dtype=np.float32)
    boxes *= np.array([image_width, image_height, image_width, image_height])
    boxes = boxes.astype(np.int32)

    return {"scores": scores, "boxes": boxes, "class_ids": class_ids}


def map_lb_original_img(img_content, letterboxed_boxes):
    # read the image from the byte stream
    img = cv2.imdecode(np.frombuffer(img_content, np.uint8),
                       cv2.IMREAD_UNCHANGED)  # return ndarray, original image

    inverse_coordinates = inverse_letterbox_coordinate_transform(
        # [(x1, y1, x2, y2)]
        letterboxed_boxes,
        ImgSize(img.shape[1], img.shape[0]),
        ImgSize(640, 640),
    )

    return inverse_coordinates


def letterboxed_result(boxes, indices, scores, class_ids, CLASSES=None):
    letterboxed_boxes = []
    new_scores = []
    labels = []

    for bbox, score, label in zip(xywh2xyxy(boxes[indices]), scores[indices],
                                  class_ids[indices]):
        bbox = bbox.round().astype(np.int32).tolist()
        letterboxed_boxes.append(tuple(bbox))
        new_scores.append(score)  # <-- Append to the new list

    return {
        "letterboxed_boxes":
        letterboxed_boxes,
        "labels":
        labels,
        "scores":
        new_scores,
        "max_score_index":
        new_scores.index(max(new_scores)) if new_scores else "None",
    }

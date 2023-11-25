import cv2
import numpy as np


class ImgSize:

    def __init__(self, width: int, height: int, channel: int = 3) -> None:
        self.height = height
        self.width = width
        self.channel = channel

    def get_tuple(self) -> tuple:
        return (self.width, self.height, self.channel)


def letterbox(img: np.ndarray,
              new_size: ImgSize,
              fill_value: int = 114) -> np.ndarray:
    # [why fill_value = 114](https://github.com/ultralytics/ultralytics/blob/796bac229eb5040159d7dff549f136f8c7e1c64e/ultralytics/data/augment.py#L587)
    aspect_ratio = min(new_size.height / img.shape[1],
                       new_size.width / img.shape[0])

    new_size_with_ar = int(img.shape[1] * aspect_ratio), int(img.shape[0] *
                                                             aspect_ratio)

    # Image resize to new_size
    resized_img = np.asarray(cv2.resize(img, new_size_with_ar))
    resized_h, resized_w, _ = resized_img.shape

    padded_img = np.full(new_size.get_tuple(), fill_value)
    center_x = new_size.width / 2
    center_y = new_size.height / 2

    x_range_start = int(center_x - (resized_w / 2))
    x_range_end = int(center_x + (resized_w / 2))

    y_range_start = int(center_y - (resized_h / 2))
    y_range_end = int(center_y + (resized_h / 2))

    padding_width = new_size.width - resized_w
    padding_height = new_size.height - resized_h

    padded_img[y_range_start:y_range_end,
               x_range_start:x_range_end, :] = resized_img

    return padded_img


from typing import List, Tuple

# type alias BBox as a tuple of four floats representing
# the coordinates of a bounding box in the format (x, y, w, h),
BBox = Tuple[float, float, float, float]  # (x, y, wh) for single bounding box


def letterbox_coordinate_transform(bboxes: List[BBox], original_size: ImgSize,
                                   letterboxed_size: ImgSize) -> List[BBox]:
    """
    The function `letterbox_coordinate_transform` takes a list of bounding boxes, the original size of
    an image, and the letterboxed size of the image, and returns a list of transformed bounding boxes
    that correspond to the letterboxed image.

    :param bboxes: The `bboxes` parameter is a list of bounding boxes. Each bounding box is represented
    as a tuple of four values: `(x1, y1, x2, y2)`. `x1` and `y1` are the coordinates of the top-left
    corner of the bounding box
    :type bboxes: List[BBox]
    :param original_size: The original_size parameter represents the size of the original image. It is
    an object of type ImgSize, which typically contains the width and height of the image
    :type original_size: ImgSize
    :param letterboxed_size: The `letterboxed_size` parameter represents the dimensions of the
    letterboxed image. It is an instance of the `ImgSize` class, which typically contains the `width`
    and `height` attributes
    :type letterboxed_size: ImgSize
    :return: a list of transformed bounding boxes in the letterboxed image dimensions.
    """

    # Calculate the aspect ratio of the original and letterboxed sizes
    aspect_ratio = min(
        letterboxed_size.height / original_size.width,
        letterboxed_size.width / original_size.height,
    )

    # Calculate the amount of padding added during the letterbox operation
    pad_w = letterboxed_size.width - (aspect_ratio * original_size.width)
    pad_h = letterboxed_size.height - (aspect_ratio * original_size.height)

    # Convert the bounding box coordinates to the letterboxed image dimensions
    letterboxed_bboxes = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        # (x1, y1) is the top-left corner of single bounding box
        map_x1 = round((x1 + pad_w / (2 * aspect_ratio)) * aspect_ratio)
        map_y1 = round((y1 + pad_h / (2 * aspect_ratio)) * aspect_ratio)

        # (x2, y2) is the bottom-right corner of single bounding box
        map_x2 = round((x2 + pad_w / (2 * aspect_ratio)) * aspect_ratio)
        map_y2 = round((y2 + pad_h / (2 * aspect_ratio)) * aspect_ratio)
        letterboxed_bboxes.append((map_x1, map_y1, map_x2, map_y2))
    return letterboxed_bboxes


def coordinate_normalize(bboxes: List[BBox], original_size: ImgSize,
                         letterboxed_size: ImgSize):
    """
    The `coordinate_normalize` function takes a list of bounding boxes, the original image size, and the
    letterboxed image size, and returns the normalized coordinates of the bounding boxes.

    :param bboxes: The `bboxes` parameter is a list of bounding boxes. Each bounding box is represented
    as a tuple of four values: `(x1, y1, x2, y2)`. `x1` and `y1` are the coordinates of the top-left
    corner of the bounding box
    :type bboxes: List[BBox]
    :param original_size: The original_size parameter represents the size of the original image before
    any letterboxing or resizing was applied. It is an object of type ImgSize, which likely contains the
    width and height of the original image
    :type original_size: ImgSize
    :param letterboxed_size: The `letterboxed_size` parameter represents the size of the image after it
    has been letterboxed. Letterboxing is a technique used to maintain the aspect ratio of an image by
    adding black bars to the top and bottom or sides of the image. The `letterboxed_size` parameter
    should be an object
    :type letterboxed_size: ImgSize
    :return: a list of normalized coordinates.
    """

    letterbox_coordinate = letterbox_coordinate_transform(
        bboxes=bboxes,
        original_size=original_size,
        letterboxed_size=letterboxed_size)

    normalized_coordinate = []
    for bbox in letterbox_coordinate:
        x1, y1, x2, y2 = bbox
        normalized_coordinate.append((
            x1 / letterboxed_size.width,
            y1 / letterboxed_size.height,
            x2 / letterboxed_size.width,
            y2 / letterboxed_size.height,
        ))

    return normalized_coordinate


def xyxy2xywh(x: np.array):
    """
    Convert bounding box (x1, y1, x2, y2) to bounding box (x, y, w, h).
    """
    y = np.copy(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y


def inverse_letterbox_coordinate_transform(
        bboxes: List[BBox], original_size: ImgSize,
        letterboxed_size: ImgSize) -> List[BBox]:
    """
    The `inverse_letterbox_coordinate_transform` function takes a list of bounding boxes, the original
    image size, and the letterboxed image size, and returns the bounding boxes transformed back to the
    original image dimensions.

    :param bboxes: The `bboxes` parameter is a list of bounding boxes. Each bounding box is represented
    as a tuple of four values: `(x1, y1, x2, y2)`. `x1` and `y1` are the coordinates of the top-left
    corner of the bounding box
    :type bboxes: List[BBox]
    :param original_size: The original_size parameter represents the dimensions of the original image
    before it was letterboxed. It is an ImgSize object that contains the width and height of the
    original image
    :type original_size: ImgSize
    :param letterboxed_size: The `letterboxed_size` parameter represents the size of the image after it
    has been letterboxed. It is an `ImgSize` object that contains the width and height of the
    letterboxed image
    :type letterboxed_size: ImgSize
    :return: a list of bounding boxes in the original image dimensions.
    """

    # Calculate the aspect ratio of the original and letterboxed sizes
    aspect_ratio = min(
        letterboxed_size.height / original_size.width,
        letterboxed_size.width / original_size.height,
    )

    # Calculate the amount of padding added during the letterbox operation
    pad_w = letterboxed_size.width - (aspect_ratio * original_size.width)
    pad_h = letterboxed_size.height - (aspect_ratio * original_size.height)

    # Convert the bounding box coordinates back to the original image dimensions
    inverse_bboxes = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        # TODO(Adam-Al-Rahman): Better method than `round`
        # (x1, y1) is the top-left corner of single bounding box
        map_x1 = round(x1 / aspect_ratio - pad_w / (2 * aspect_ratio))
        map_y1 = round(y1 / aspect_ratio - pad_h / (2 * aspect_ratio))

        # (x2, y2) is the bottom-right corner of single bounding box
        map_x2 = round(x2 / aspect_ratio - pad_w / (2 * aspect_ratio))
        map_y2 = round(y2 / aspect_ratio - pad_h / (2 * aspect_ratio))
        inverse_bboxes.append((map_x1, map_y1, map_x2, map_y2))
    return inverse_bboxes

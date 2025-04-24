import pytest
import numpy as np
import cv2
import io
from unittest.mock import patch, MagicMock
from strawberry.file_uploads import Upload
from app.utils.images_predict_fn import (
    compute_iou,
    nms,
    xywh2xyxy,
    letterboxed_result,
    model_ort_session,
    final_image_pre_process,
    bboxs_filter,
    map_lb_original_img,
)
from app.utils.images import predict_images, ImgSize

# Fixture for a synthetic image
@pytest.fixture
def sample_image(tmp_path):
    # Create a 100x100 synthetic image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :, 0] = 255  # Blue image
    img_path = tmp_path / "sample_image.jpg"
    cv2.imwrite(str(img_path), img)
    
    # Read image as bytes to simulate Upload
    with open(img_path, "rb") as f:
        img_bytes = f.read()
    
    # Mock Strawberry Upload object
    upload = MagicMock(spec=Upload)
    upload.read = MagicMock(return_value=img_bytes)
    return upload, "sample_image.jpg"

# Fixture for mocked ONNX model session
@pytest.fixture
def mock_ort_session():
    with patch("app.utils.images_predict_fn.ort") as mock_ort:
        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [
            MagicMock(name="input", shape=[1, 3, 640, 640])
        ]
        mock_session.get_outputs.return_value = [MagicMock(name="output")]
        mock_ort.InferenceSession.return_value = mock_session
        
        # Mock model output: [1, 84, 8400] (YOLOv8-like: 4 bbox + 80 classes)
        mock_output = np.random.rand(1, 84, 8400).astype(np.float32)
        mock_session.run.return_value = [mock_output]
        
        yield {
            "session": mock_session,
            "input_names": ["input"],
            "input_shape": [1, 3, 640, 640],
            "output_names": ["output"],
        }

# Test compute_iou
def test_compute_iou():
    box = np.array([50, 50, 150, 150])  # x1, y1, x2, y2
    boxes = np.array([
        [50, 50, 150, 150],  # Same box (IoU = 1.0)
        [100, 100, 200, 200],  # Partial overlap
        [200, 200, 300, 300],  # No overlap
    ])
    iou = compute_iou(box, boxes)
    assert np.isclose(iou[0], 1.0)  # Same box
    assert 0 < iou[1] < 1.0  # Partial overlap
    assert np.isclose(iou[2], 0.0)  # No overlap

# Test nms
def test_nms():
    boxes = np.array([
        [50, 50, 150, 150],  # High score box
        [60, 60, 160, 160],  # Overlapping box
        [200, 200, 300, 300],  # Non-overlapping box
    ])
    scores = np.array([0.9, 0.8, 0.7])
    indices = nms(boxes, scores, iou_threshold=0.5)
    assert len(indices) == 2  # Should keep non-overlapping boxes
    assert 0 in indices  # Highest score box
    assert 2 in indices  # Non-overlapping box

# Test xywh2xyxy
def test_xywh2xyxy():
    boxes = np.array([
        [100, 100, 50, 50],  # x, y, w, h
        [200, 200, 100, 100],
    ])
    result = xywh2xyxy(boxes)
    expected = np.array([
        [75, 75, 125, 125],  # x1, y1, x2, y2
        [150, 150, 250, 250],
    ])
    np.testing.assert_array_equal(result, expected)

# Test letterboxed_result
def test_letterboxed_result():
    boxes = np.array([
        [100, 100, 50, 50],  # x, y, w, h
        [200, 200, 100, 100],
    ])
    scores = np.array([0.9, 0.8])
    class_ids = np.array([0, 1])
    indices = [0, 1]
    result = letterboxed_result(boxes, indices, scores, class_ids)
    assert len(result["letterboxed_boxes"]) == 2
    assert result["letterboxed_boxes"][0] == (75, 75, 125, 125)
    assert result["scores"] == [0.9, 0.8]
    assert result["max_score_index"] == 0

# Mock letterbox and inverse_letterbox_coordinate_transform
@pytest.fixture
def mock_letterbox():
    with patch("app.utils.images_predict_fn.letterbox") as mock:
        # Mock letterbox to return a 640x640 image
        mock.return_value = np.zeros((640, 640, 3), dtype=np.uint8)
        yield mock

@pytest.fixture
def mock_inverse_letterbox():
    with patch("app.utils.images_predict_fn.inverse_letterbox_coordinate_transform") as mock:
        # Mock to return same coordinates (simplified)
        mock.side_effect = lambda boxes, orig_size, lb_size: boxes
        yield mock

# Test predict_images
def test_predict_images(sample_image, mock_ort_session, mock_letterbox, mock_inverse_letterbox):
    upload, image_name = sample_image
    result = predict_images(content=upload, image_name=image_name, confidence=0.5)
    
    # Assert output structure
    assert isinstance(result, dict)
    assert "image" in result
    assert "coordinates" in result
    assert "max_confidence_coordinate" in result
    assert result["image"] == image_name
    assert isinstance(result["coordinates"], list)
    assert isinstance(result["max_confidence_coordinate"], tuple)
    
    # Check if coordinates are valid tuples (x1, y1, x2, y2)
    for coord in result["coordinates"]:
        assert len(coord) == 4
        assert all(isinstance(x, (int, float)) for x in coord)

# Test predict_images with invalid file extension
def test_predict_images_invalid_extension(sample_image):
    upload, _ = sample_image
    result = predict_images(content=upload, image_name="sample.txt", confidence=0.5)
    assert result == {
        "image": "sample.txt",
        "coordinates": [],
        "max_confidence_coordinate": (-1, -1, -1, -1),
    }
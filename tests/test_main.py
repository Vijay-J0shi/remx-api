import pytest
from fastapi.testclient import TestClient
from app.main import app
from unittest.mock import patch, MagicMock
import os


MOCK_MODEL = {
    'session': MagicMock(),  
    'input_names': ['images'],
    'input_shape': [1, 3, 640, 640],
    'output_names': ['output0']
}

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def sample_image():
    file_path = os.path.join("tests", "sample_image1.jpg")
    with open(file_path, "rb") as f:
        yield ("sample_image1.jpg", f, "image/jpeg")

@pytest.fixture
def sample_zip():
    file_path = os.path.join("tests", "tst.zip")
    with open(file_path, "rb") as f:
        yield ("tst.zip", f, "application/zip")

def test_upload_image(client, sample_image):
    mock_output = [[[0.9, 0.1, 100, 100, 200, 200]]]
    with patch("app.model", MOCK_MODEL) as mock_model:
        mock_model['session'].run.return_value = [mock_output]  

        files = [("files", sample_image)]
        response = client.post("/api/predict/upload", files=files)

        assert response.status_code == 200
        response_data = response.json()
        assert isinstance(response_data, (list, dict))
      
def test_upload(client, sample_zip):
    mock_output = [[[0.9, 0.1, 100, 100, 200, 200]]]  
    with patch("app.model", MOCK_MODEL) as mock_model:
        mock_model['session'].run.return_value = [mock_output]

        files = [("files", sample_zip)]
        response = client.post("/api/predict/upload", files=files)

        assert response.status_code == 200
        response_data = response.json()
        assert isinstance(response_data, (list, dict))
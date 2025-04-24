import requests
import os

from app.model.model import model_ort_session
model = model_ort_session("app/model/remx_model_1.0.0.onnx")
print("Model loaded successfully!")

print(model)

# path = os.path.join("tests","tst.zip")
# files =[("files", open(path, "rb"))]


# response = requests.post("http://127.0.0.1:8000/api/predict/upload", files=files)

# # Check response

# print(response.json())
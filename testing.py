import requests
import os
path = os.path.join("tests","tst.zip")
files =[("files", open(path, "rb"))]


response = requests.post("http://127.0.0.1:8000/api/predict/upload", files=files)

# # Check response

print(response.json())

import requests

files =[("files", open("tests/tst.zip", "rb"))]
    # ("files", open("tests/sample_imgae2.jpg", "rb"))


response = requests.post("http://127.0.0.1:8000/api/predict/upload", files=files)

# # Check response

print(response.json())

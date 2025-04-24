from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html

from app.model.model import __version__ as model_version
from app.prediction_api import prediction_router 

app = FastAPI(title="Remx REST API", version=model_version)

app.include_router(prediction_router, prefix="/api", tags=["Prediction"])

# Static & template setup
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", include_in_schema=False)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "modelVersion": model_version,
        "endPoint": "/api/predict/upload"
    })

@app.get("/docs", include_in_schema=False)
def custom_swagger_ui():
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="Remx Swagger",
        swagger_favicon_url="/static/images/favicon.png"
    )

@app.get("/redoc", include_in_schema=False)
def custom_redoc_ui():
    return get_redoc_html(
        openapi_url="/openapi.json",
        title="Remx Redoc",
        redoc_favicon_url="/static/images/favicon.png"
    )

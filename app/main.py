import strawberry
from strawberry.fastapi import GraphQLRouter
from fastapi.templating import Jinja2Templates

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.docs import (
    get_redoc_html,
    get_swagger_ui_html,
)

from app.model.model import __version__ as model_version
from app.schema import Mutation, Query

schema = strawberry.Schema(query=Query, mutation=Mutation)
graphql_app = GraphQLRouter(schema)
app = FastAPI(title="Remx", docs_url=None, redoc_url=None)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/docs", include_in_schema=False)
def overridden_swagger():
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="Remx",
        swagger_favicon_url="/static/images/favicon.png")


@app.get("/redoc", include_in_schema=False)
def overridden_redoc():
    return get_redoc_html(openapi_url="/openapi.json",
                          title="Remx",
                          redoc_favicon_url="/static/images/favicon.png")


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html", {
            "request": request,
            "modelVersion": model_version,
            "endPoint": "https://remx.server.com/api"
        })


app.include_router(graphql_app, prefix="/api")

# Phase: Testing
# import uvicorn
# uvicorn.run(app, host="0.0.0.0", port=8000)

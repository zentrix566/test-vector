"""把告警匹配能力包成 Web 服务（FastAPI）。

体现"模型 -> 对外服务"的核心：
  - 进程启动时把模型加载进内存一次（import alert_kb 时就完成），并初始化知识库
  - 之后每个 HTTP 请求复用同一个模型，只做一次前向推理，很轻量

启动服务：
    uvicorn api:app --host 127.0.0.1 --port 8000

接口：
    GET  /         健康检查
    POST /match    body: {"alert": "告警文本"}  ->  匹配结果
交互式文档：http://127.0.0.1:8000/docs
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from alert_kb import match_alert, seed_knowledge_base


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 服务启动时确保知识库已初始化（upsert 幂等，重复启动无副作用）
    seed_knowledge_base()
    yield


# docs_url=None：关掉默认 /docs（它从国内访问不稳的 jsdelivr CDN 加载资源）
app = FastAPI(title="告警匹配知识库 API", lifespan=lifespan, docs_url=None)

# 把本地下载的 Swagger UI 静态资源挂到 /static
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/docs", include_in_schema=False)
def custom_docs():
    # 用本地静态资源渲染交互式文档，不依赖外网 CDN
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="告警匹配知识库 API - 文档",
        swagger_js_url="/static/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-ui.css",
    )


class AlertIn(BaseModel):
    alert: str


@app.get("/")
def health():
    return {"status": "ok", "service": "alert-kb"}


@app.post("/match")
def match(item: AlertIn):
    """传入一条告警，返回命中已知问题（含处置办法）或判定为新问题。"""
    return match_alert(item.alert)

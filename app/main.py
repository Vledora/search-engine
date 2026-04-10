"""FastAPI application: search API + HTML UI."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.search.service import SearchService

TEMPLATES_DIR = Path(__file__).resolve().parents[1] / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

service = SearchService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    service.load_indices()
    yield


app = FastAPI(title="Search Engine MVP", lifespan=lifespan)


@app.get("/api/search")
def api_search(
    q: str = Query(..., min_length=1),
    method: str = Query("bm25"),
    top_k: int = Query(10, ge=1, le=50),
):
    results = service.search(q, method=method, top_k=top_k)
    return {
        "query": q,
        "method": method,
        "count": len(results),
        "results": [
            {
                "title": r.title,
                "url": r.url,
                "source": r.source,
                "score": r.score,
                "snippet": r.snippet,
            }
            for r in results
        ],
    }


@app.get("/api/methods")
def api_methods():
    return {"methods": service.available_methods}


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request, "index.html")

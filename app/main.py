from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import analyze, minutes, translate


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """종료 시 코퍼스 커넥션 풀 정리.

    풀은 첫 검색 때 지연 생성한다(기동 시 DB에 붙지 않아야 DB가 늦게 떠도 서버가 뜬다).
    """
    yield
    from app.rag.corpus import close_pool

    close_pool()


app = FastAPI(
    title="Bridgenote AI",
    description="번역·문화 오해 분석·회의록 생성 (내부 호출 전용 — Spring BE가 호출)",
    version="0.0.1",
    lifespan=lifespan,
)

# 내부 호출 전제라 CORS는 원칙적으로 불필요하나, 로컬 개발 편의상 허용.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """에러 응답 규약: 400 { "message": "..." } (필수 필드 누락 등)."""
    errors = exc.errors()
    if errors:
        first = errors[0]
        loc = ".".join(str(x) for x in first.get("loc", [])[1:])
        message = f"{loc}: {first.get('msg', '')}".strip(": ")
    else:
        message = "잘못된 요청입니다."
    return JSONResponse(status_code=400, content={"message": message})


@app.get("/health", tags=["health"])
async def health():
    return {"status": "ok", "service": "bridgenote-ai"}


app.include_router(analyze.router)
app.include_router(translate.router)
app.include_router(minutes.router)

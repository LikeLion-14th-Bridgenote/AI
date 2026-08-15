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
    """라이브니스 — 프로세스가 살아있는지만. 배포 플랫폼 헬스체크용."""
    return {"status": "ok", "service": "bridgenote-ai"}


@app.get("/health/deep", tags=["health"])
async def health_deep():
    """각주 기능이 실제로 동작하는지 확인한다.

    /health는 각주가 죽어도 ok를 반환한다(fail-open 설계). 그래서 배포 후에
    "번역만 나가고 각주는 조용히 꺼진 상태"를 알아채지 못한다. 실제로 pgvector
    타입 불일치로 모든 벡터 검색이 실패하던 때에도 /health는 ok였다.

    여기서는 임베딩 → 코퍼스 검색까지 한 번 태워보고, fail-open이 몇 번
    일어났는지도 함께 보여준다. 진단용이라 문제가 있어도 200으로 응답한다.
    """
    from app.core.config import get_settings
    from app.core.embeddings import embed_one
    from app.core.health import snapshot
    from app.rag.corpus import search_by_vector

    s = get_settings()
    checks = {
        "llm_key": {"ok": bool(s.llm_api_key), "detail": s.llm_provider if s.llm_api_key else "미설정"},
        "db_url": {"ok": bool(s.supabase_db_url), "detail": "설정됨" if s.supabase_db_url else "미설정"},
    }

    try:
        vec = embed_one("헬스체크")
        checks["embedding"] = {"ok": len(vec) == s.embed_dim, "detail": f"{len(vec)}차원"}
    except Exception as e:  # noqa: BLE001
        vec = None
        checks["embedding"] = {"ok": False, "detail": f"{type(e).__name__}: {e}"}

    if vec is None:
        checks["corpus"] = {"ok": False, "detail": "임베딩 실패로 검사 못 함"}
    else:
        try:
            rows = await search_by_vector(vec, "KR", top_k=1, entry_type="risk_seed")
            checks["corpus"] = {"ok": bool(rows), "detail": f"{len(rows)}건 조회" if rows
                                else "KR risk_seed 없음 — 코퍼스 적재 확인 필요"}
        except Exception as e:  # noqa: BLE001
            checks["corpus"] = {"ok": False, "detail": f"{type(e).__name__}: {e}"}

    return {
        "status": "ok" if all(c["ok"] for c in checks.values()) else "degraded",
        "service": "bridgenote-ai",
        "checks": checks,
        "fail_open": snapshot(),  # 비어 있으면 아직 삼킨 실패 없음
    }


app.include_router(analyze.router)
app.include_router(translate.router)
app.include_router(minutes.router)

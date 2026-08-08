from fastapi import APIRouter

from app.schemas.analyze import AnalyzeRequest, AnalyzeResponse
from app.services.analyze.service import analyze as analyze_service

router = APIRouter(prefix="/ai", tags=["analyze"])


@router.post("/analyze", response_model=AnalyzeResponse, response_model_exclude_none=True)
async def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    """게이트 판정 → 통과 시 각주·리라이트·번역, 미통과 시 번역만."""
    return await analyze_service(req)

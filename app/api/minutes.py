from fastapi import APIRouter

from app.schemas.minutes import MinutesRequest, MinutesResponse
from app.services.minutes.service import generate_minutes

router = APIRouter(prefix="/ai", tags=["minutes"])


@router.post("/minutes", response_model=MinutesResponse)
async def minutes(req: MinutesRequest) -> MinutesResponse:
    """회의록 생성(배치). 세션 로그 전체 → 결정/논의/액션 분리, 언어·직무별 생성."""
    return await generate_minutes(req)

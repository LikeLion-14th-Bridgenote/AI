from fastapi import APIRouter

from app.schemas.translate import TranslateRequest, TranslateResponse
from app.services.translate.service import translate as translate_service

router = APIRouter(prefix="/ai", tags=["translate"])


@router.post("/translate", response_model=TranslateResponse)
async def translate(req: TranslateRequest) -> TranslateResponse:
    """게이트 미통과(각주 불필요) 발화의 번역만 단독 처리."""
    return await translate_service(req)

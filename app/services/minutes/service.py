from app.schemas.minutes import MinutesRequest, MinutesResponse


async def generate_minutes(req: MinutesRequest) -> MinutesResponse:
    """/ai/minutes (스텁, 배치). 세션 로그 → 결정/논의/액션 분리, 언어·직무별 생성."""
    # TODO: LLM으로 언어·직무별 회의록 생성.
    return MinutesResponse(meeting_id=req.meeting_id, minutes=[])

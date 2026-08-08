# Bridgenote AI

Bridgenote의 **AI 서버 (FastAPI)**. 번역·문화 오해 분석·회의록 생성을 담당한다.

이 서버의 엔드포인트는 프론트가 직접 호출하지 않는다. 메인 백엔드(`bridgenote-BE`, Spring)가 확정 발화를 받아 이 서버를 내부 HTTP로 호출하고, 결과를 받아 WebSocket으로 클라이언트에 전달한다.

## Tech Stack

| 항목 | 값 |
| --- | --- |
| Language | Python 3.11+ |
| Framework | FastAPI |
| Server | Uvicorn |
| LLM | Claude 또는 OpenAI |
| STT | Deepgram (Agora 미사용 확정) |
| RAG | Supabase pgvector (culture_corpus 코퍼스) |
| Docs | FastAPI 자동 문서 (`/docs`) |

## Prerequisites

- Python 3.11+
- pip / venv
- LLM API 키, Deepgram 키 (`.env`)
- Supabase DB 접근 (pgvector 코퍼스 조회)

## Project Structure

```
app
 ├─ api        # FastAPI 라우터 (엔드포인트 정의)
 ├─ core       # 설정, 환경변수, LLM/STT 클라이언트 초기화
 ├─ schemas    # 요청/응답 Pydantic 모델
 ├─ services
 │   ├─ analyze    # 문화 오해 판정(게이트) + 각주 + 리라이트
 │   ├─ translate  # 다국어 번역
 │   └─ minutes    # 회의록 생성(결정/논의/액션)
 ├─ rag        # pgvector 코퍼스 검색 (culture_corpus)
 └─ prompts    # LLM 프롬프트 템플릿 (Culture Map 축 기반)
tests
```

## Ownership

- 전체 AI 서버 – 원종윤

## 문화 판단 로직 (학술 근거)

- **오해 감지 게이트**: 임베딩 유사도(pgvector) + 문화 거리 가중으로 1차 필터. 모든 문장을 무겁게 처리하지 않기 위한 단계.
- **문화 각주**: 게이트 통과분만 LLM이 Erin Meyer의 Culture Map 8축(직설성·피드백·대립·위계 등) 기준으로 판단. 화자 의도 / 수신자 오해 소지 / 권장 대응 3요소 구조화 출력.
- **근거 이론**: Edward Hall 고맥락/저맥락 이론 (한국·베트남=고맥락, 서구=저맥락), Erin Meyer Culture Map.

## Environment

- 기본 포트: `8000`
- `.env` 예시는 `.env.example` 참고 (실제 값 커밋 금지)

## API (내부 호출용)

메인 백엔드(Spring)가 STT 확정 발화(`is_final`)를 받아 호출한다. 프론트는 직접 호출하지 않는다.

| Method | Endpoint | 설명 |
| --- | --- | --- |
| POST | `/ai/analyze` | 게이트 판정 → 통과 시 각주·리라이트·번역까지 함께 반환 |
| POST | `/ai/translate` | 게이트 미통과(각주 불필요) 발화의 번역만 단독 처리 |
| POST | `/ai/minutes` | 회의록 생성(배치) |

### 처리 흐름

1. `is_final` 발화를 `/ai/analyze`가 임베딩 게이트로 오해 소지 판정
2. 오해 소지 있음(`has_risk: true`) → 각주 + 리라이트 + 참가자 언어별 번역을 한 번에 반환
3. 오해 소지 없음(`has_risk: false`) → 번역만 반환 (각주 필드 없음)
4. 각주가 불필요한 발화의 순수 번역은 `/ai/translate`로 단독 처리 가능

> ⚠️ 번역의 B/C 경계는 확정 필요 — 문화 리라이트와 한 LLM 호출로 묶으면 C(FastAPI)가 맡는 것이 효율적.

### `/ai/analyze` 요청

```json
{
  "utterance_id": "uuid",
  "meeting_id": "uuid",
  "source_text": "네, 검토해보겠습니다",
  "source_lang": "ko",
  "meeting_context": "법률 자문 미팅",
  "speaker": { "culture": "KR", "job_role": "pm" },
  "listeners": [
    { "participant_id": "uuid", "lang": "en", "culture": "US", "job_role": "dev" },
    { "participant_id": "uuid", "lang": "vi", "culture": "VN", "job_role": "design" }
  ],
  "context": [ { "text": "금요일까지 가능할까요?", "lang": "ko" } ]
}
```

### `/ai/analyze` 응답 — 오해 소지 있음 (게이트 통과)

```json
{
  "utterance_id": "uuid",
  "has_risk": true,
  "risk_level": "High",
  "note_type": "간접적 피드백",
  "speaker_intent": "신중한 유보(완곡한 거절 가능성)",
  "listener_misread": "US 청자는 '승인'으로 오독 가능",
  "advice": "구체적 기한을 되물어 확인",
  "rewrite_text": "I'll need to review this—can we confirm the deadline?",
  "translations": [
    { "participant_id": "uuid", "lang": "en", "text": "Yes, I'll review it." },
    { "participant_id": "uuid", "lang": "vi", "text": "Vâng, tôi sẽ xem xét." }
  ]
}
```

### `/ai/analyze` 응답 — 오해 소지 없음 (게이트 미통과, 번역만)

```json
{
  "utterance_id": "uuid",
  "has_risk": false,
  "translations": [
    { "participant_id": "uuid", "lang": "en", "text": "..." }
  ]
}
```

> `risk_level`은 High / Med / Low (`cultural_note` 테이블과 일치).

### `/ai/translate` 요청·응답

```json
// 요청
{
  "utterance_id": "uuid",
  "source_text": "금요일까지 가능할까요?",
  "source_lang": "ko",
  "targets": [
    { "participant_id": "uuid", "lang": "en" },
    { "participant_id": "uuid", "lang": "vi" }
  ]
}
```

```json
// 응답 200
{
  "utterance_id": "uuid",
  "translations": [
    { "participant_id": "uuid", "lang": "en", "text": "Is it possible by Friday?" },
    { "participant_id": "uuid", "lang": "vi", "text": "Có thể xong trước thứ Sáu không?" }
  ]
}
```

## 관련 테이블 (Supabase)

- `cultural_note` — 각주 저장 (risk_level·note_type·speaker_intent·listener_misread·advice·rewrite_text)
- `culture_corpus` — RAG용 문화 규칙 코퍼스 (id, rule_text, source_lang, target_lang, embedding VECTOR)

## How to Run (Local)

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # 값 채우기
uvicorn app.main:app --reload --port 8000
```

문서: http://localhost:8000/docs

## Git Workflow

- 기본 브랜치: `main`
- 작업 브랜치 규칙: `feature/<기능>-<이름>`
  - 예) `feature/analyze-jongyoon`, `feature/translate-jongyoon`

```bash
git checkout -b feature/analyze-jongyoon
git add .
git commit -m "[ADD] 문화 오해 판정 프롬프트 추가"
git push -u origin feature/analyze-jongyoon
```

### PR 규칙

- `main`에 직접 push 금지 → 반드시 브랜치를 파서 PR로만 병합 (초기 세팅 커밋만 예외)
- 민감 파일(`.env`, API 키)은 반드시 `.gitignore`에 추가 (이미 포함)
- PR 제목 예시:
  - `feat(analyze): implement culture gate prompt`
  - `fix(translate): context ordering bug`

## Commit Convention

| 태그 | 의미 |
| --- | --- |
| `[INIT]` | 초기 세팅 |
| `[ADD]` | 기능 추가 |
| `[FIX]` | 버그 수정 |
| `[REFACTOR]` | 리팩토링 |
| `[HOTFIX]` | 긴급 수정 |

## 관련 문서

- 기능 명세서 (노션)
- API 명세서 (노션)
- 메인 백엔드: `bridgenote-BE` 레포

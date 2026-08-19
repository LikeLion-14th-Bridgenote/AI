# Bridgenote AI

Bridgenote의 **AI 서버 (FastAPI)**. 번역·문화 오해 분석·회의록 생성을 담당한다.

이 서버의 엔드포인트는 프론트가 직접 호출하지 않는다. 메인 백엔드(`bridgenote-BE`, Spring)가 확정 발화를 받아 이 서버를 내부 HTTP로 호출하고, 결과를 받아 WebSocket으로 클라이언트에 전달한다.

## Tech Stack

| 항목 | 값 |
| --- | --- |
| Language | Python 3.11+ |
| Framework | FastAPI |
| Server | Uvicorn |
| LLM | provider 스위치 — `openai` \| `gemini` \| `claude` (기본 openai) |
| 임베딩 | 로컬 fastembed `paraphrase-multilingual-MiniLM-L12-v2` (384차원) — API 호출 없음 |
| RAG | Supabase pgvector (`culture_corpus` 코퍼스 검색) |
| STT | 이 서버는 담당하지 않음 — Deepgram 스트리밍은 Spring BE `realtime` 모듈 |
| Docs | FastAPI 자동 문서 (`/docs`) |

LLM 모델은 `LLM_MODEL`로 지정하고, 비우면 provider 기본값(openai=`gpt-4o-mini`, gemini=`gemini-2.0-flash`, claude=`claude-sonnet-4-6`)을 쓴다. `.env.example`은 `gpt-4o`로 지정해 두었다.

## Prerequisites

- Python 3.11+
- pip / venv
- LLM API 키 (`.env`의 `LLM_API_KEY`)
- Supabase DB 접근 (`SUPABASE_DB_URL` — pgvector 코퍼스 조회)

`STT_API_KEY`는 스키마에만 남아 있는 자리표시자다. 실시간 오디오 스트리밍은 Spring BE가 처리하므로 이 서버 구동에는 필요하지 않다.

## Project Structure

```
app
 ├─ api        # FastAPI 라우터 (엔드포인트 정의)
 ├─ core       # 설정, 임베딩, LLM 클라이언트, fail-open 기록
 ├─ schemas    # 요청/응답 Pydantic 모델
 ├─ services
 │   ├─ analyze    # 문화 오해 판정(게이트) + 각주 + 리라이트
 │   ├─ translate  # 다국어 번역
 │   └─ minutes    # 회의록 생성(결정/논의/액션)
 ├─ rag        # pgvector 코퍼스 검색 (culture_corpus) + 문화 거리
 └─ prompts    # LLM 프롬프트 템플릿 (Culture Map 축 기반)
data
 ├─ seed       # culture_corpus 시드 (rule / risk_seed / neutral_seed)
 └─ eval       # 게이트 평가셋 + 실제 STT 전사 샘플
scripts        # 코퍼스 적재, 게이트 평가, 누수 검사, 스모크 테스트
tests
```

## Ownership

- 전체 AI 서버 – 원종윤

## 문화 판단 로직 (학술 근거)

- **오해 감지 게이트**: 발화를 임베딩해 **화자 문화의 위험표현 시드**(실제 완곡 표현)와의 유사도를 재고, 정상표현 시드와 비교한 뒤 화자↔청자 문화 거리로 문턱을 조정한다. 모든 문장을 무겁게 처리하지 않기 위한 1차 필터.
- **문화 각주**: 게이트 통과분만 LLM이 Erin Meyer의 Culture Map 8축(직설성·피드백·대립·위계 등) 기준으로 판단. 화자 의도 / 수신자 오해 소지 / 권장 대응 3요소 구조화 출력.
- **근거 이론**: Edward Hall 고맥락/저맥락 이론 (한국·베트남=고맥락, 서구=저맥락), Erin Meyer Culture Map.
- **fail-open**: 임베딩·코퍼스 조회가 실패하면 각주만 생략하고 번역 등 나머지 응답은 정상 반환한다(500을 내지 않는다). 삼킨 실패 횟수는 `/health/deep`의 `fail_open`에서 확인한다.
- 게이트 튜닝값과 그 근거는 `app/core/config.py` 주석, 재현은 `scripts/eval_gate.py` 참고.

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
| GET | `/health` | 라이브니스 — 프로세스 생존만 확인 (배포 플랫폼용) |
| GET | `/health/deep` | 각주 기능이 실제로 도는지 진단 (LLM 키·DB·임베딩·코퍼스 조회) |

`/health`는 각주가 죽어도 `ok`를 반환한다(fail-open). 배포 후 "번역만 나가고 각주는 조용히 꺼진 상태"를 확인하려면 `/health/deep`을 봐야 한다.

### 공통 규약

- 발화 식별자 필드명은 전부 **`sentence_id`**다 (`utterance_id` 아님)
- 화자·청자의 직무 필드명은 **`job`**이다 (`job_role` 아님). 회의록 응답의 `job_role`은 별개 필드
- 응답에서 `null` 필드는 제외된다 (`response_model_exclude_none=True`). 게이트 미통과 시 각주 필드들은 키 자체가 없다
- 검증 실패(필수 필드 누락 등)는 **400**과 `{ "message": "..." }` 형태로 응답한다

### 처리 흐름

1. `is_final` 발화를 `/ai/analyze`가 임베딩 게이트로 오해 소지 판정
2. 오해 소지 있음(`has_risk: true`) → 각주 + 리라이트 + 참가자 언어별 번역을 한 번에 반환
3. 오해 소지 없음(`has_risk: false`) → 번역만 반환 (각주 필드 없음)
4. 각주가 불필요한 발화의 순수 번역은 `/ai/translate`로 단독 처리 가능

### `/ai/analyze` 요청

```json
{
  "sentence_id": "uuid",
  "meeting_id": "uuid",
  "source_text": "네, 검토해보겠습니다",
  "source_lang": "ko",
  "meeting_context": "법률 자문 미팅",
  "speaker": { "culture": "KR", "job": "pm" },
  "listeners": [
    { "participant_id": "uuid", "lang": "en", "culture": "US", "job": "dev" },
    { "participant_id": "uuid", "lang": "vi", "culture": "VN", "job": "design" }
  ],
  "context": [ { "text": "금요일까지 가능할까요?", "lang": "ko" } ]
}
```

`meeting_context`와 `context`는 선택 항목이다. `context`는 직전 발화 몇 건으로, 생략하면 빈 배열로 처리된다.

### `/ai/analyze` 응답 — 오해 소지 있음 (게이트 통과)

```json
{
  "sentence_id": "uuid",
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
  "sentence_id": "uuid",
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
  "sentence_id": "uuid",
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
  "sentence_id": "uuid",
  "translations": [
    { "participant_id": "uuid", "lang": "en", "text": "Is it possible by Friday?" },
    { "participant_id": "uuid", "lang": "vi", "text": "Có thể xong trước thứ Sáu không?" }
  ]
}
```

### `/ai/minutes` 요청·응답

세션 로그 전체를 넘기면 언어·직무 조합별로 회의록 섹션을 만들어 돌려준다.

```json
// 요청
{
  "meeting_id": "uuid",
  "languages": ["ko", "en"],
  "job_roles": ["pm", "dev"],
  "utterances": [
    { "sentence_id": "u1", "speaker": "지훈", "lang": "ko", "text": "로그인 API는 금요일까지 붙이겠습니다" },
    { "sentence_id": "u2", "speaker": "Minh", "lang": "vi", "text": "Tôi sẽ lo phần giao diện đăng nhập" }
  ]
}
```

```json
// 응답 200
{
  "meeting_id": "uuid",
  "minutes": [
    {
      "language": "ko",
      "job_role": "pm",
      "decisions": ["로그인 API 마감을 금요일로 확정"],
      "discussions": ["로그인 화면 담당 분배"],
      "action_items": [
        { "task": "로그인 API 연동", "owner": "지훈", "deadline": "금요일" }
      ]
    }
  ]
}
```

`speaker`, `job_role`, `owner`, `deadline`은 없으면 생략된다.

## 관련 테이블 (Supabase)

- `cultural_note` — 각주 저장 (risk_level·note_type·speaker_intent·listener_misread·advice·rewrite_text)
- `culture_corpus` — RAG용 문화 코퍼스 (`id`, `rule_text`, `culture`, `note_type`, `source`, `entry_type`, `embedding vector`)
  - `entry_type`은 `rule`(문화 규칙) / `risk_seed`(오해 유발 완곡표현) / `neutral_seed`(정상 표현) 세 가지
  - 적재는 `python scripts/ingest_corpus.py`

## How to Run (Local)

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # 값 채우기
uvicorn app.main:app --reload --port 8000
```

문서: http://localhost:8000/docs

첫 실행 때 임베딩 모델(약 240MB)을 내려받으므로 조금 걸린다. 이후에는 로컬 캐시를 쓴다.

### 게이트 평가

```bash
python scripts/eval_gate.py          # 평가셋으로 정밀도/재현율/F1 + 문턱 스윕
python scripts/check_eval_leakage.py # 평가셋 문장이 시드에 섞였는지 검사
```

## Git Workflow

- 기본 브랜치: `main`
- 작업 브랜치 규칙: `feat/AI/<주제>`
  - 예) `feat/AI/eval-observed-cases`

```bash
git checkout -b feat/AI/analyze-prompt
git add app/prompts/culture_map.py   # 변경 파일만 지정 (git add . 금지)
git commit -m "feat(analyze): 문화 오해 판정 프롬프트 추가"
git push -u origin feat/AI/analyze-prompt
```

### PR 규칙

- `main`에 직접 push 금지 → 반드시 브랜치를 파서 PR로만 병합 (초기 세팅 커밋만 예외)
- 민감 파일(`.env`, API 키)은 반드시 `.gitignore`에 추가 (이미 포함)
- PR 본문에는 무엇을 고쳤는지보다 **어떤 문제 때문에 고쳤는지**를 먼저 쓴다

## Commit Convention

Conventional Commits를 쓴다.

| 태그 | 의미 |
| --- | --- |
| `feat:` | 기능 추가 |
| `fix:` | 버그 수정 |
| `docs:` | 문서 수정 |
| `refactor:` | 리팩토링 |
| `test:` | 테스트 추가·수정 |

범위를 붙이면 더 좋다 — `feat(analyze):`, `fix(eval):`, `docs(config):`

## 관련 문서

- 기능 명세서 (노션)
- API 명세서 (노션)
- 메인 백엔드: `bridgenote-BE` 레포

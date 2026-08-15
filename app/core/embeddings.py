"""로컬 다국어 임베딩 (fastembed / ONNX).

게이트는 발화마다 호출되므로 API 왕복 없이 로컬에서 임베딩한다.
모델(paraphrase-multilingual-MiniLM-L12-v2)은 한/영/베/중 등 50+ 언어를 커버.
첫 호출 시 모델을 1회 다운로드(실측 241MB)하고 이후 캐시된다.

ONNX 추론은 동기 CPU 작업이라 async 안에서 그대로 부르면 그 시간 동안 이벤트 루프가
멈춘다(= 다른 발화 처리가 대기). embed_one은 스레드로 넘겨 실행한다.
"""

from functools import lru_cache
from typing import List

import anyio.to_thread

from app.core.config import get_settings


@lru_cache(maxsize=1)
def _model():
    # fastembed는 무거우므로 import·로딩을 첫 사용 시점까지 지연.
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=get_settings().embed_model)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """여러 문장을 임베딩 (적재용 배치 — 스크립트에서 동기로 쓴다)."""
    return [v.tolist() for v in _model().embed(list(texts))]


@lru_cache(maxsize=256)
def _embed_cached(text: str) -> tuple:
    return tuple(embed_texts([text])[0])


def embed_one_sync(text: str) -> List[float]:
    """동기 버전 — 스크립트·테스트용. 서비스 코드는 embed_one을 쓴다."""
    return list(_embed_cached(text))


async def embed_one(text: str) -> List[float]:
    """한 문장을 임베딩 (게이트·검색용).

    한 발화가 게이트와 각주 RAG 양쪽에서 임베딩되는데 결과가 같으므로 캐시한다.
    캐시에 없으면 ONNX 추론이 돌아가므로 스레드로 넘겨 이벤트 루프를 막지 않는다.
    """
    return await anyio.to_thread.run_sync(embed_one_sync, text)

"""로컬 다국어 임베딩 (fastembed / ONNX).

게이트는 발화마다 호출되므로 API 왕복 없이 로컬에서 임베딩한다.
모델(paraphrase-multilingual-MiniLM-L12-v2)은 한/영/베/중 등 50+ 언어를 커버.
첫 호출 시 모델을 1회 다운로드(실측 241MB)하고 이후 캐시된다.
"""

from functools import lru_cache
from typing import List

from app.core.config import get_settings


@lru_cache(maxsize=1)
def _model():
    # fastembed는 무거우므로 import·로딩을 첫 사용 시점까지 지연.
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=get_settings().embed_model)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """여러 문장을 임베딩 (적재용 배치)."""
    return [v.tolist() for v in _model().embed(list(texts))]


@lru_cache(maxsize=256)
def _embed_cached(text: str) -> tuple:
    return tuple(embed_texts([text])[0])


def embed_one(text: str) -> List[float]:
    """한 문장을 임베딩 (게이트·검색용).

    한 발화가 게이트(gate.py)와 각주 RAG(analyze/service.py) 양쪽에서 임베딩되는데,
    임베딩은 텍스트만의 함수라 결과가 같다. 캐시해서 요청당 1회로 줄인다.
    호출부가 리스트를 변형하지 않도록 tuple로 들고 매번 새 리스트를 돌려준다.
    """
    return list(_embed_cached(text))

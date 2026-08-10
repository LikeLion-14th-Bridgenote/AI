"""로컬 다국어 임베딩 (fastembed / ONNX).

게이트는 발화마다 호출되므로 API 왕복 없이 로컬에서 임베딩한다.
모델(paraphrase-multilingual-MiniLM-L12-v2)은 한/영/베/중 등 50+ 언어를 커버.
첫 호출 시 모델을 1회 다운로드(약 0.22GB)하고 이후 캐시된다.
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


def embed_one(text: str) -> List[float]:
    """한 문장을 임베딩 (게이트·검색용)."""
    return embed_texts([text])[0]

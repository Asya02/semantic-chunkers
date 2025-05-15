import os
from asyncio import sleep as asleep
from time import sleep
from typing import Any, List, Optional

from dotenv import find_dotenv, load_dotenv
from langchain_gigachat.embeddings import GigaChatEmbeddings
from pydantic.v1 import PrivateAttr
from semantic_router.encoders import BaseEncoder
from semantic_router.schema import EncoderInfo
from semantic_router.utils.logger import logger

load_dotenv(find_dotenv())

model_configs = {
    "EmbeddingsGigaR": EncoderInfo(
        name="EmbeddingsGigaR",
        token_limit=8192,
        threshold=0.3,
    )
}


class GigaChatEncoder(BaseEncoder):
    type: str = "gigachat"
    token_limit: int = 8192
    max_retries: int = 3
    _embedder: Any = PrivateAttr()

    def __init__(
        self,
        name: Optional[str] = None,
        score_threshold: Optional[float] = None,
        max_retries: int = 3,
    ):
        if name is None:
            name = "EmbeddingsGigaR"
        if score_threshold is None and name in model_configs:
            score_threshold = model_configs[name].threshold
        elif score_threshold is None:
            logger.warning(
                f"Score threshold not set for model: {name}. Using default value."
            )
            score_threshold = 0.3

        super().__init__(name=name, score_threshold=score_threshold)

        self.max_retries = max_retries
        if name in model_configs:
            self.token_limit = model_configs[name].token_limit

        self._embedder = GigaChatEmbeddings(
            verify_ssl_certs=False,
            model=name
        )

    def __call__(self, docs: List[str], truncate: bool = True) -> List[List[float]]:
        if truncate:
            docs = [self._truncate(doc) for doc in docs]

        for j in range(self.max_retries + 1):
            try:
                embeddings = [self._embedder.embed_query(doc) for doc in docs]
                return embeddings
            except Exception as e:
                logger.error(f"GigaChat embedding failed. Error: {e}")
                if j < self.max_retries:
                    sleep(2 ** j)
                    logger.warning(f"Retrying in {2 ** j} seconds...")
                else:
                    raise

        raise ValueError("No embeddings returned.")

    async def acall(self, docs: List[str], truncate: bool = True) -> List[List[float]]:
        if truncate:
            docs = [self._truncate(doc) for doc in docs]

        for j in range(self.max_retries + 1):
            try:
                embeddings = [self._embedder.embed_query(doc) for doc in docs]
                return embeddings
            except Exception as e:
                logger.error(f"GigaChat embedding failed. Error: {e}")
                if j < self.max_retries:
                    await asleep(2 ** j)
                    logger.warning(f"Retrying in {2 ** j} seconds...")
                else:
                    raise

        raise ValueError("No embeddings returned.")

    def _truncate(self, text: str) -> str:
        if len(text) > self.token_limit * 4:
            logger.warning(
                f"Document exceeds token limit. Truncating..."
            )
            return text[:self.token_limit * 4]
        return text

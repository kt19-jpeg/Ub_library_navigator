import hashlib

from chromadb.api.types import Documents, EmbeddingFunction


class FakeEmbeddingFunction(EmbeddingFunction[Documents]):
    """Deterministic, offline stand-in for a real embedding model: same text always
    gives the same vector, and chromadb never makes a network call in tests."""

    def __call__(self, input: Documents) -> list[list[float]]:
        vectors = []
        for text in input:
            digest = hashlib.sha256(text.encode("utf-8")).digest()[:8]
            vectors.append([b / 255 for b in digest])
        return vectors

    @staticmethod
    def name() -> str:
        return "fake"

    def get_config(self) -> dict:
        return {}

    @staticmethod
    def build_from_config(config: dict) -> "FakeEmbeddingFunction":
        return FakeEmbeddingFunction()

    @staticmethod
    def is_legacy() -> bool:
        return False

# 📄 src/utils/embedder_utils.py
from typing import List
from sentence_transformers import SentenceTransformer
from chromadb import Documents, EmbeddingFunction, Embeddings as EMD
from langchain.embeddings.base import Embeddings


class SentenceTransformerEmbedder(EmbeddingFunction):
    def __init__(self, model_path: str):
        self.model = SentenceTransformer(model_path)
        super().__init__()

    def __call__(self, input: Documents) -> EMD:
        return self.model.encode(input).tolist()

    def encode_documents(self, documents: List[str]) -> List[List[float]]:
        return self.model.encode(documents).tolist()

    def encode_queries(self, queries: List[str]) -> List[List[float]]:
        return self.model.encode(queries).tolist()


class LangchainSentenceTransformer(Embeddings):
    def __init__(self, model_path):
        self.model = SentenceTransformer(model_path)

    def embed_documents(self, texts):
        """Embeds a list of texts and returns list of vectors"""
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text):
        """Embeds a single query and returns vector"""
        return self.model.encode(text, convert_to_numpy=True).tolist()

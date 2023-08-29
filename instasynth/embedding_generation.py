import hashlib
import pickle
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, ClassVar, List

import tiktoken
from openai.embeddings_utils import get_embedding

from .config import logger


@dataclass
class EmbeddingStorage:
    storage_path: Path
    embedding_file_name: str = "embeddings.pkl"
    _embeddings_file: Path = field(init=False)
    _hash_file: Path = field(init=False)
    _hash_embeddings: Dict[str, Any] = field(
        init=False, default_factory=dict, repr=False
    )
    _hash_map: Dict[str, str] = field(init=False, default_factory=dict, repr=False)

    def __post_init__(self):
        self._embeddings_file = self.storage_path / self.embedding_file_name
        self._hash_file = self.storage_path / f".hash_{self.embedding_file_name}"
        # Attempt to load from the pickle file during initialization
        self.load_from_disk()

    def _generate_hash(self, text: str) -> str:
        hash = hashlib.sha256(text.encode()).hexdigest()
        self._hash_map[hash] = text
        return hash

    def store_embedding(self, text: str, embedding: Any) -> None:
        text_hash = self._generate_hash(text)
        self._hash_embeddings[text_hash] = embedding
        self._save_to_disk()

    def _save_to_disk(self) -> None:
        with self._embeddings_file.open("wb") as file, self._hash_file.open(
            "wb"
        ) as hash_file:
            pickle.dump(self._hash_embeddings, file)
            pickle.dump(self._hash_map, hash_file)

    def load_from_disk(self) -> None:
        try:
            with self._embeddings_file.open("rb") as file, self._hash_file.open(
                "rb"
            ) as hash_file:
                self._hash_embeddings = pickle.load(file)
                self._hash_map = pickle.load(hash_file)

        except FileNotFoundError:
            self._hash_embeddings = {}
            self._hash_map = {}

    def check_if_embedded(self, text: str) -> bool:
        text_hash = self._generate_hash(text)
        return text_hash in self._hash_embeddings

    def get_embedding(self, text: str) -> Any:
        text_hash = self._generate_hash(text)
        return self._hash_embeddings.get(text_hash, None)


@dataclass
class EmbeddingGenerator:
    embedding_storage: EmbeddingStorage
    texts: List[str]
    embeddings: Dict[str, Any] = field(default_factory=dict, repr=False)
    embedding_model = "text-embedding-ada-002"
    embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
    encoding = tiktoken.get_encoding(embedding_encoding)
    max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191

    def _generate_embedding(self, text: str) -> Any:
        if len(self.encoding.encode(text)) <= self.max_tokens:
            return np.array(get_embedding(text, self.embedding_model))
        return None

    def generate_and_store(self) -> None:
        for i, text in enumerate(self.texts):
            if i % 100 == 0:
                logger.info(f"Generating embedding for text {i+1}/{len(self.texts)}")
            if not self.embedding_storage.check_if_embedded(text):
                embedding = self._generate_embedding(text)
                self.embeddings[text] = embedding
                self.embedding_storage.store_embedding(text, embedding)

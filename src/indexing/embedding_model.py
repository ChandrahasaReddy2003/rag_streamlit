from typing import List, Union

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer


class EmbeddingModel:
    """
    Offline embedding fallback.

    This avoids Hugging Face downloads and works on corporate laptops where
    external model downloads may fail because of SSL/proxy restrictions.
    It creates deterministic local vectors that can be stored in FAISS.

    Later you can replace this class with OpenAI/Azure/Gemini embeddings
    without changing the retrieval pipeline.
    """

    def __init__(self, model_name: str = None):
        self.model_name = model_name or "local-hashing-vectorizer"
        self.vectorizer = HashingVectorizer(
            n_features=4096,
            alternate_sign=False,
            norm="l2",
            ngram_range=(1, 2),
            lowercase=True,
        )

    def encode(
        self,
        texts: Union[str, List[str]],
        show_progress_bar: bool = False,
        **kwargs,
    ) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        vectors = self.vectorizer.transform(texts)
        return vectors.toarray().astype(np.float32)

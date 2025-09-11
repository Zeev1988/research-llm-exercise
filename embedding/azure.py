import os
from typing import List, Tuple, Optional
import numpy as np
from dotenv import load_dotenv
from azure.identity import get_bearer_token_provider, EnvironmentCredential
from openai import AzureOpenAI


class AzureClientBase:
    """Base Azure client that holds a shared AzureOpenAI client."""

    def __init__(self, client: Optional[AzureOpenAI] = None) -> None:
        load_dotenv(override=True)
        if client is not None:
            self.client = client
        else:
            token_provider = get_bearer_token_provider(
                EnvironmentCredential(), "https://cognitiveservices.azure.com/.default"
            )
            self.client = AzureOpenAI(
                api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                azure_ad_token_provider=token_provider,
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            )


class AzureEmbeddingsClient(AzureClientBase):
    """Embeddings client built on top of AzureClientBase."""

    def __init__(self, model_env_var: str = "AZURE_OPENAI_MODEL_ADA2", client: Optional[AzureOpenAI] = None) -> None:
        super().__init__(client=client)
        self.model_name = os.getenv(model_env_var)
        if not self.model_name:
            raise RuntimeError(f"Missing environment variable {model_env_var}")

    def embed_texts(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """Embed a list of texts using Azure OpenAI in batches without progress UI."""
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        all_vectors: List[List[float]] = []
        for batch in self._batched(texts, batch_size):
            vectors = self._embed_text(batch)
            all_vectors.extend(vectors)

        return self._normalize(all_vectors)

    def _batched(self, items: List[str], batch_size: int):
        for i in range(0, len(items), batch_size):
            yield items[i : i + batch_size]

    def _embed_text(self, batch: List[str]) -> List[List[float]]:
        try:
            resp = self.client.embeddings.create(model=self.model_name, input=batch)
            return [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]
        except Exception:
            return []

    def _normalize(self, vectors: List[List[float]]) -> np.ndarray:
        if not vectors:
            return np.zeros((0, 0), dtype=np.float32)
        arr = np.array(vectors, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms


class AzureChatClient(AzureClientBase):
    """Chat completions client built on top of AzureClientBase."""

    def __init__(self, model_env_var: str = "AZURE_OPENAI_MODEL_GPT4o", client: Optional[AzureOpenAI] = None) -> None:
        super().__init__(client=client)
        self.model_name = os.getenv(model_env_var)
        if not self.model_name:
            raise RuntimeError(f"Missing environment variable {model_env_var}")

    def chat(self, messages: List[dict]) -> str:
        resp = self.client.chat.completions.create(model=self.model_name, messages=messages)
        return resp.choices[0].message.content if resp.choices and resp.choices[0].message else ""

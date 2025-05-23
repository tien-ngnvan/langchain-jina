import base64
import json
from os.path import exists
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests
from langchain_core.embeddings import Embeddings
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
from pydantic import BaseModel, ConfigDict, SecretStr, model_validator

JINA_API_URL: str = "https://api.jina.ai/v1/embeddings"


def is_local(url: str) -> bool:
    """Check if a URL is a local file.

    Args:
        url (str): The URL to check.

    Returns:
        bool: True if the URL is a local file, False otherwise.
    """
    url_parsed = urlparse(url)
    if url_parsed.scheme in ("file", ""):  # Possibly a local file
        return exists(url_parsed.path)
    return False


def get_bytes_str(file_path: str) -> str:
    """Get the bytes string of a file.

    Args:
        file_path (str): The path to the file.

    Returns:
        str: The bytes string of the file.
    """
    with open(file_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


class LateChunkEmbeddings(BaseModel, Embeddings):
    session: Any  #: :meta private:
    model_name: str = "jina-embeddings-v3"
    jina_api_key: Optional[SecretStr] = None

    model_config = ConfigDict(protected_namespaces=())
    dimensions: int = 1024
    embedding_type: float = "float"

    model_config = ConfigDict(protected_namespaces=())

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """Validate that auth token exists in environment."""
        try:
            jina_api_key = convert_to_secret_str(
                get_from_dict_or_env(values, "jina_api_key", "JINA_API_KEY")
            )
        except ValueError as original_exc:
            try:
                jina_api_key = convert_to_secret_str(
                    get_from_dict_or_env(values, "jina_auth_token", "JINA_AUTH_TOKEN")
                )
            except ValueError as exc:
                raise original_exc from exc
        session = requests.Session()
        session.headers.update(
            {
                "Authorization": f"Bearer {jina_api_key.get_secret_value()}",
                "Accept-Encoding": "identity",
                "Content-type": "application/json",
            }
        )
        values["session"] = session
        return values

    def _embed(self, input: Any, task: str, late_chunking: bool) -> List[List[float]]:
        # Call Jina AI Embedding API
        data = {
            "input": input,
            "model": self.model_name,
            "late_chunking": late_chunking,
            "dimensions": self.dimensions,
            "task": task,
            "embedding_type": self.embedding_type
        }

        resp = self.session.post(  # type: ignore
            JINA_API_URL, data=json.dumps(data)
        ).json()

        if "data" not in resp:
            raise RuntimeError(resp["detail"])

        embeddings = resp["data"]

        return [result["embedding"] for result in embeddings]

    def embed_documents(
        self, texts: List[str], task="retrieval.passage", late_chunking=True
    ) -> List[List[float]]:
        """Call out to Jina's embedding endpoint.
        Args:
            texts: The list of texts to embed.
            task: Task-Specific Embedding. `retrieval.passage` used for 
                passage embeddings in asymmetric retrieval tasks
            late_chunking: Apply latechunking or not. Default: True
        Returns:
            List of embeddings, one for each text.
        """

        return self._embed(texts, task=task, late_chunking=late_chunking)

    def embed_query(
        self, text: str, task="retrieval.query", late_chunking=False
    ) -> List[float]:
        """Call out to Jina's embedding endpoint.
        Args:
            text: The text to embed.
            task: Task-Specific Embedding. `retrieval.query' used for 
                    query embeddings in asymmetric retrieval tasks
            late_chunking: Apply latechunking or not. Default: False
        Returns:
            Embeddings for the text.
        """

        return self._embed([text], task=task, late_chunking=late_chunking)[0]

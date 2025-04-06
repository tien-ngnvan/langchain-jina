from importlib import metadata

from langchain_jina.vectorestores.qdrant import LateChunkQdrant
from langchain_jina.embeddings import LateChunkEmbeddings

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "__version__",
    "LateChunkQdrant",
    "LateChunkEmbeddings",
]

import os
import pytest
from langchain_jina import LateChunkEmbeddings

DOCUMENTS = [
    "Berlin is the capital and largest city of Germany, by both area and population.",
    "With 3.66 million inhabitants, it has the highest population within "
    "its city limits of any city in the European Union.",
    "The city is also one of the states of Germany, being the third smallest state in the country by area.",
]

@pytest.fixture(scope="module")
def jina_embeddings():
    api_key = os.environ.get("JINA_API_KEY", "")
    if not api_key:
        pytest.skip("JINA_API_KEY environment variable not set. Skipping Jina embeddings tests.")
    return LateChunkEmbeddings(jina_api_key=api_key, model_name="jina-embeddings-v3")


class TestLateChunkEmbeddings:
    """Tests for LateChunkEmbeddings."""

    def test_generate_embed_documents_with_latechunking(self, jina_embeddings):
        """Test embedding documents with late chunking."""
        embeddings = jina_embeddings.embed_documents(texts=DOCUMENTS, late_chunking=True)
        assert len(embeddings) == len(DOCUMENTS)
        for embedding in embeddings:
            assert isinstance(embedding, list)
            assert len(embedding) > 0

    def test_generate_embed_documents_without_latechunking(self, jina_embeddings):
        """Test embedding documents without late chunking."""
        embeddings = jina_embeddings.embed_documents(texts=DOCUMENTS, late_chunking=False)
        assert len(embeddings) == len(DOCUMENTS)
        for embedding in embeddings:
            assert isinstance(embedding, list)
            assert len(embedding) > 0

    def test_generate_embed_query(self, jina_embeddings):
        """Test embedding a single query."""
        embedding = jina_embeddings.embed_query(text=DOCUMENTS[0])
        assert isinstance(embedding, list)
        assert len(embedding) > 0
import os
import pytest
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_jina import LateChunkEmbeddings, LateChunkQdrant
from langchain_core.documents import Document
from qdrant_client import QdrantClient


@pytest.fixture
def setup_vectorstore():
    """Fixture to set up the vectorstore."""
    # Prepare the document
    doc = Document(
        page_content=(
            "Berlin is the capital and largest city of Germany, by both area and population. "
            "With 3.66 million inhabitants, it has the highest population within its city "
            "limits of any city in the European Union. The city is also one of "
            "the states of Germany, being the third smallest state in the country by area."
        ),
        metadata={},
    )

    # Set up the text embeddings
    text_embeddings = LateChunkEmbeddings(
        jina_api_key=os.environ.get("JINA_API_KEY"),
        model_name="jina-embeddings-v3",
    )

    # Set up the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=0,
        length_function=len,
        is_separator_regex=False,
    )

    # Initialize the Qdrant client and vectorstore
    client = QdrantClient(":memory:")
    vectorstore = LateChunkQdrant(
        client,
        collection_name="test",
        embeddings=text_embeddings,
        text_splitter=text_splitter,
    )

    # Load the document into the vectorstore
    vectorstore = vectorstore.from_documents(
        documents=[doc],
        embedding=text_embeddings,
        text_splitter=text_splitter,
        collection_name="test",
        location=":memory:",
    )

    return vectorstore


def test_search_similarity_with_score(setup_vectorstore):
    """Test to search similarity with score in the vectorstore."""
    vectorstore = setup_vectorstore
    results = vectorstore.similarity_search_with_score(
        "What is Berlin", k=3
    )
    assert len(results) > 0
    print("\nResponse: ", results)

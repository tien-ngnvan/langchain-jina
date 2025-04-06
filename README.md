# langchain-jina
This package contains the LangChain integration with Late Chunking


## Installation
`pip install -U langchain-jina`

## Environment Variable

Export your logins:
`export JINA_API_KEY="jina_*`

## Usage
### 1. Get Embedings
Here is an example usage of these classes:
```python
from langchain_jina import LateChunkEmbeddings

text_embeddings = LateChunkEmbeddings(
    jina_api_key=os.environ.get("JINA_API_KEY"),
    model_name="jina-embeddings-v3"
)

text = [
    "Berlin is the capital and largest city of Germany, by both area and population.",
    "With 3.66 million inhabitants, it has the highest population within its city limits of any city in the European Union.",
    "The city is also one of the states of Germany, being the third smallest state in the country by area.",
]

# with late chunking
doc_result = text_embeddings.embed_documents(text, late_chunking=True)
print("With late_chunking")
for doc in doc_result:
    print(doc)
```

### 2. Build with Vectorstore
First of all, we need the context length entire input text limit with the model context length. So, we using `tokenizer` from [transformers](https://huggingface.co/docs/transformers/en/index) to check it.
```
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v3")
```

Next, when the `tokenizer` is loaded, we can combine it with any `text_splitter` LangChain. The example below giving the instruction of handle the same method of authors.
```
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0,
    length_function=len,
    is_separator_regex=False,
)

text_splitter.tokenizer = tokenizer 
```

We create vectorstore embeding, here we use `LateChunkQdrant`
```python
from qdrant_client import QdrantClient
from langchain_community.docstore.document import Document
from langchain_jina import LateChunkQdrant


client = QdrantClient()

vectorstore = LateChunkQdrant(
    client, 
    collection_name="demo",
    embeddings=text_embeddings, 
    text_splitter=text_splitter
)

# load documents
with open("./state_of_the_union.txt") as f:
        state_of_the_union = f.read()

documents  = [
    Document(
        page_content=state_of_the_union, 
        metadata={"source": "state_of_the_union.txt"}
    ),
]

vectorstore = vectorstore.from_documents(
    documents=documents, 
    embedding=text_embeddings,
    text_splitter=text_splitter,
    path="test_db", 
    collection_name="demo"
)
```

Finally, we can combine with any purpose
```python
query = "What did the president say about ketanji brown jackson?" 
results = vectorstore.similarity_search(query, k=3)

for res in results:
    print(f"* {res.page_content} [{res.metadata}]")
```

## License
This project is licensed under the [MIT License](./LICENSE)
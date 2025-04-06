from __future__ import annotations

import functools
import uuid
import warnings
from itertools import islice
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

from langchain_core._api.deprecation import deprecated
from langchain_core.embeddings import Embeddings
from langchain_core.runnables.config import run_in_executor
from langchain_text_splitters import TextSplitter

from langchain_community.vectorstores import Qdrant

if TYPE_CHECKING:
    from qdrant_client import grpc  # noqa
    from qdrant_client.conversions import common_types
    from qdrant_client.http import models as rest

    DictFilter = Dict[str, Union[str, int, bool, dict, list]]
    MetadataFilter = Union[DictFilter, common_types.Filter]


class QdrantException(Exception):
    """`Qdrant` related exceptions."""


def sync_call_fallback(method: Callable) -> Callable:
    """
    Decorator to call the synchronous method of the class if the async method is not
    implemented. This decorator might be only used for the methods that are defined
    as async in the class.
    """

    @functools.wraps(method)
    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return await method(self, *args, **kwargs)
        except NotImplementedError:
            # If the async method is not implemented, call the synchronous method
            # by removing the first letter from the method name. For example,
            # if the async method is called ``aaad_texts``, the synchronous method
            # will be called ``aad_texts``.
            return await run_in_executor(
                None, getattr(self, method.__name__[1:]), *args, **kwargs
            )

    return wrapper


@deprecated(since="0.0.37", removal="1.0", alternative_import="langchain_qdrant.Qdrant")
class LateChunkQdrant(Qdrant):
    CONTENT_KEY: str = "page_content"
    METADATA_KEY: str = "metadata"
    VECTOR_NAME = None

    def __init__(
        self,
        client: Any,
        collection_name: str,
        text_splitter: TextSplitter,
        embeddings: Optional[Embeddings] = None,
        content_payload_key: str = CONTENT_KEY,
        metadata_payload_key: str = METADATA_KEY,
        distance_strategy: str = "COSINE",
        vector_name: Optional[str] = VECTOR_NAME,
        async_client: Optional[Any] = None,
        embedding_function: Optional[Callable] = None,  # deprecated
    ):
        super().__init__(
            client=client,
            collection_name=collection_name,
            embeddings=embeddings,
            content_payload_key=content_payload_key,
            metadata_payload_key=metadata_payload_key,
            distance_strategy=distance_strategy,
            vector_name=vector_name,
            async_client=async_client,
            embedding_function=embedding_function
        )

        self.text_splitter = text_splitter

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[Sequence[str]] = None,
        batch_size: int = 64,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            ids:
                Optional list of ids to associate with the texts. Ids have to be
                uuid-like strings.
            batch_size:
                How many vectors upload per-request.
                Default: 64

        Returns:
            List of ids from adding the texts into the vectorstore.
        """

        context_length = kwargs.get('context_length', 8194)
        step = batch_size // 2
        added_ids = []
        for subtext, submetadatas, subembed in self.process_text_in_batches(
            texts, metadatas, step, context_length
        ):
            for batch_ids, points in self._generate_rest_batches(
                subtext, subembed, submetadatas, ids, batch_size
            ):
                self.client.upsert(
                    collection_name=self.collection_name, points=points, **kwargs
                )
                added_ids.extend(batch_ids)

        return added_ids

    @sync_call_fallback
    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[Sequence[str]] = None,
        batch_size: int = 64,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            ids:
                Optional list of ids to associate with the texts. Ids have to be
                uuid-like strings.
            batch_size:
                How many vectors upload per-request.
                Default: 64

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        from qdrant_client.local.async_qdrant_local import AsyncQdrantLocal
        if self.async_client is None or isinstance(
            self.async_client._client, AsyncQdrantLocal
        ):
            raise NotImplementedError(
                "QdrantLocal cannot interoperate with sync and async clients"
            )

        context_length = kwargs.get('context_length', 8194)
        step = batch_size // 2
        added_ids = []
        async for subtext, submetadatas, subembed in self.process_text_in_batches(
            texts, metadatas, step, context_length
        ):
            async for batch_ids, points in self._agenerate_rest_batches(
                subtext, subembed, submetadatas, ids, batch_size
            ):
                await self.async_client.upsert(
                    collection_name=self.collection_name, points=points, **kwargs
                )
                added_ids.extend(batch_ids)

        return added_ids

    @classmethod
    def from_texts(
        cls: Type[Qdrant],
        texts: List[str],
        embedding: Embeddings,
        text_splitter: TextSplitter,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[Sequence[str]] = None,
        location: Optional[str] = None,
        url: Optional[str] = None,
        port: Optional[int] = 6333,
        grpc_port: int = 6334,
        prefer_grpc: bool = False,
        https: Optional[bool] = None,
        api_key: Optional[str] = None,
        prefix: Optional[str] = None,
        timeout: Optional[float] = None,
        host: Optional[str] = None,
        path: Optional[str] = None,
        collection_name: Optional[str] = None,
        distance_func: str = "Cosine",
        content_payload_key: str = CONTENT_KEY,
        metadata_payload_key: str = METADATA_KEY,
        vector_name: Optional[str] = VECTOR_NAME,
        batch_size: int = 64,
        shard_number: Optional[int] = None,
        replication_factor: Optional[int] = None,
        write_consistency_factor: Optional[int] = None,
        on_disk_payload: Optional[bool] = None,
        hnsw_config: Optional[common_types.HnswConfigDiff] = None,
        optimizers_config: Optional[common_types.OptimizersConfigDiff] = None,
        wal_config: Optional[common_types.WalConfigDiff] = None,
        quantization_config: Optional[common_types.QuantizationConfig] = None,
        init_from: Optional[common_types.InitFrom] = None,
        on_disk: Optional[bool] = None,
        force_recreate: bool = False,
        **kwargs: Any,
    ) -> Qdrant:
        """Construct LateChunkQdrant wrapper from a list of texts.

        Args:
            texts: A list of texts to be indexed in Qdrant.
            embedding: A subclass of `Embeddings`, responsible for text vectorization.
            metadatas:
                An optional list of metadata. If provided it has to be of the same
                length as a list of texts.
            text_splitter:
                Split the text up into small, semantically meaningful chunks 
                (often sentences).
            ids:
                Optional list of ids to associate with the texts. Ids have to be
                uuid-like strings.
            location:
                If `:memory:` - use in-memory Qdrant instance.
                If `str` - use it as a `url` parameter.
                If `None` - fallback to relying on `host` and `port` parameters.
            url: either host or str of "Optional[scheme], host, Optional[port],
                Optional[prefix]". Default: `None`
            port: Port of the REST API interface. Default: 6333
            grpc_port: Port of the gRPC interface. Default: 6334
            prefer_grpc:
                If true - use gPRC interface whenever possible in custom methods.
                Default: False
            https: If true - use HTTPS(SSL) protocol. Default: None
            api_key: API key for authentication in Qdrant Cloud. Default: None
            prefix:
                If not None - add prefix to the REST URL path.
                Example: service/v1 will result in
                    http://localhost:6333/service/v1/{qdrant-endpoint} for REST API.
                Default: None
            timeout:
                Timeout for REST and gRPC API requests.
                Default: 5.0 seconds for REST and unlimited for gRPC
            host:
                Host name of Qdrant service. If url and host are None, set to
                'localhost'. Default: None
            path:
                Path in which the vectors will be stored while using local mode.
                Default: None
            collection_name:
                Name of the Qdrant collection to be used. If not provided,
                it will be created randomly. Default: None
            distance_func:
                Distance function. One of: "Cosine" / "Euclid" / "Dot".
                Default: "Cosine"
            content_payload_key:
                A payload key used to store the content of the document.
                Default: "page_content"
            metadata_payload_key:
                A payload key used to store the metadata of the document.
                Default: "metadata"
            vector_name:
                Name of the vector to be used internally in Qdrant.
                Default: None
            batch_size:
                How many vectors upload per-request.
                Default: 64
            shard_number: Number of shards in collection. Default is 1, minimum is 1.
            replication_factor:
                Replication factor for collection. Default is 1, minimum is 1.
                Defines how many copies of each shard will be created.
                Have effect only in distributed mode.
            write_consistency_factor:
                Write consistency factor for collection. Default is 1, minimum is 1.
                Defines how many replicas should apply the operation for us to consider
                it successful. Increasing this number will make the collection more
                resilient to inconsistencies, but will also make it fail if not enough
                replicas are available.
                Does not have any performance impact.
                Have effect only in distributed mode.
            on_disk_payload:
                If true - point`s payload will not be stored in memory.
                It will be read from the disk every time it is requested.
                This setting saves RAM by (slightly) increasing the response time.
                Note: those payload values that are involved in filtering and are
                indexed - remain in RAM.
            hnsw_config: Params for HNSW index
            optimizers_config: Params for optimizer
            wal_config: Params for Write-Ahead-Log
            quantization_config:
                Params for quantization, if None - quantization will be disabled
            init_from:
                Use data stored in another collection to initialize this collection
            force_recreate:
                Force recreating the collection
            **kwargs:
                Additional arguments passed directly into REST client initialization

        This is a user-friendly interface that:
        1. Creates embeddings, one for each text
        2. Initializes the Qdrant database as an in-memory docstore by default
           (and overridable to a remote docstore)
        3. Adds the text embeddings to the Qdrant database

        This is intended to be a quick way to get started.

        Example:
            .. code-block:: python

                from langchain_community.vectorstores import LateChunkQdrant
                from langchain_community.embeddings import JinaLateChunkEmbeddings
                from langchain_text_splitters import CharacterTextSplitter
                embeddings = JinaLateChunkEmbeddings()
                text_splitter = CharacterTextSplitter()
                qdrant = LateChunkQdrant.from_texts(
                    texts, embeddings, text_splitter, "localhost"
                )
        """
        qdrant = cls.construct_instance(
            texts,
            embedding,
            text_splitter,
            location,
            url,
            port,
            grpc_port,
            prefer_grpc,
            https,
            api_key,
            prefix,
            timeout,
            host,
            path,
            collection_name,
            distance_func,
            content_payload_key,
            metadata_payload_key,
            vector_name,
            shard_number,
            replication_factor,
            write_consistency_factor,
            on_disk_payload,
            hnsw_config,
            optimizers_config,
            wal_config,
            quantization_config,
            init_from,
            on_disk,
            force_recreate,
            **kwargs,
        )
        qdrant.add_texts(texts, metadatas, ids, batch_size)

        return qdrant

    @classmethod
    @sync_call_fallback
    async def afrom_texts(
        cls: Type[Qdrant],
        texts: List[str],
        embedding: Embeddings,
        text_splitter: TextSplitter,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[Sequence[str]] = None,
        location: Optional[str] = None,
        url: Optional[str] = None,
        port: Optional[int] = 6333,
        grpc_port: int = 6334,
        prefer_grpc: bool = False,
        https: Optional[bool] = None,
        api_key: Optional[str] = None,
        prefix: Optional[str] = None,
        timeout: Optional[float] = None,
        host: Optional[str] = None,
        path: Optional[str] = None,
        collection_name: Optional[str] = None,
        distance_func: str = "Cosine",
        content_payload_key: str = CONTENT_KEY,
        metadata_payload_key: str = METADATA_KEY,
        vector_name: Optional[str] = VECTOR_NAME,
        batch_size: int = 64,
        shard_number: Optional[int] = None,
        replication_factor: Optional[int] = None,
        write_consistency_factor: Optional[int] = None,
        on_disk_payload: Optional[bool] = None,
        hnsw_config: Optional[common_types.HnswConfigDiff] = None,
        optimizers_config: Optional[common_types.OptimizersConfigDiff] = None,
        wal_config: Optional[common_types.WalConfigDiff] = None,
        quantization_config: Optional[common_types.QuantizationConfig] = None,
        init_from: Optional[common_types.InitFrom] = None,
        on_disk: Optional[bool] = None,
        force_recreate: bool = False,
        **kwargs: Any,
    ) -> Qdrant:
        """Construct Qdrant wrapper from a list of texts.

        Args:
            texts: A list of texts to be indexed in Qdrant.
            embedding: A subclass of `Embeddings`, responsible for text vectorization.
            metadatas:
                An optional list of metadata. If provided it has to be of the same
                length as a list of texts.
            text_splitter:
                Split the text up into small, semantically meaningful chunks 
                (often sentences).
            ids:
                Optional list of ids to associate with the texts. Ids have to be
                uuid-like strings.
            location:
                If `:memory:` - use in-memory Qdrant instance.
                If `str` - use it as a `url` parameter.
                If `None` - fallback to relying on `host` and `port` parameters.
            url: either host or str of "Optional[scheme], host, Optional[port],
                Optional[prefix]". Default: `None`
            port: Port of the REST API interface. Default: 6333
            grpc_port: Port of the gRPC interface. Default: 6334
            prefer_grpc:
                If true - use gPRC interface whenever possible in custom methods.
                Default: False
            https: If true - use HTTPS(SSL) protocol. Default: None
            api_key: API key for authentication in Qdrant Cloud. Default: None
            prefix:
                If not None - add prefix to the REST URL path.
                Example: service/v1 will result in
                    http://localhost:6333/service/v1/{qdrant-endpoint} for REST API.
                Default: None
            timeout:
                Timeout for REST and gRPC API requests.
                Default: 5.0 seconds for REST and unlimited for gRPC
            host:
                Host name of Qdrant service. If url and host are None, set to
                'localhost'. Default: None
            path:
                Path in which the vectors will be stored while using local mode.
                Default: None
            collection_name:
                Name of the Qdrant collection to be used. If not provided,
                it will be created randomly. Default: None
            distance_func:
                Distance function. One of: "Cosine" / "Euclid" / "Dot".
                Default: "Cosine"
            content_payload_key:
                A payload key used to store the content of the document.
                Default: "page_content"
            metadata_payload_key:
                A payload key used to store the metadata of the document.
                Default: "metadata"
            vector_name:
                Name of the vector to be used internally in Qdrant.
                Default: None
            batch_size:
                How many vectors upload per-request.
                Default: 64
            shard_number: Number of shards in collection. Default is 1, minimum is 1.
            replication_factor:
                Replication factor for collection. Default is 1, minimum is 1.
                Defines how many copies of each shard will be created.
                Have effect only in distributed mode.
            write_consistency_factor:
                Write consistency factor for collection. Default is 1, minimum is 1.
                Defines how many replicas should apply the operation for us to consider
                it successful. Increasing this number will make the collection more
                resilient to inconsistencies, but will also make it fail if not enough
                replicas are available.
                Does not have any performance impact.
                Have effect only in distributed mode.
            on_disk_payload:
                If true - point`s payload will not be stored in memory.
                It will be read from the disk every time it is requested.
                This setting saves RAM by (slightly) increasing the response time.
                Note: those payload values that are involved in filtering and are
                indexed - remain in RAM.
            hnsw_config: Params for HNSW index
            optimizers_config: Params for optimizer
            wal_config: Params for Write-Ahead-Log
            quantization_config:
                Params for quantization, if None - quantization will be disabled
            init_from:
                Use data stored in another collection to initialize this collection
            force_recreate:
                Force recreating the collection
            **kwargs:
                Additional arguments passed directly into REST client initialization

        This is a user-friendly interface that:
        1. Creates embeddings, one for each text
        2. Initializes the Qdrant database as an in-memory docstore by default
           (and overridable to a remote docstore)
        3. Adds the text embeddings to the Qdrant database

        This is intended to be a quick way to get started.

        Example:
            .. code-block:: python
                            
                from langchain_community.vectorstores import LateChunkQdrant
                from langchain_community.embeddings import JinaLateChunkEmbeddings
                from langchain_text_splitters import CharacterTextSplitter
                embeddings = JinaLateChunkEmbeddings()
                text_splitter = CharacterTextSplitter()
                qdrant = await LateChunkQdrant.afrom_texts(
                    texts, embeddings, text_splitter, "localhost"
                )
        """
        qdrant = await cls.aconstruct_instance(
            texts,
            embedding,
            text_splitter,
            location,
            url,
            port,
            grpc_port,
            prefer_grpc,
            https,
            api_key,
            prefix,
            timeout,
            host,
            path,
            collection_name,
            distance_func,
            content_payload_key,
            metadata_payload_key,
            vector_name,
            shard_number,
            replication_factor,
            write_consistency_factor,
            on_disk_payload,
            hnsw_config,
            optimizers_config,
            wal_config,
            quantization_config,
            init_from,
            on_disk,
            force_recreate,
            **kwargs,
        )
        await qdrant.aadd_texts(texts, metadatas, ids, batch_size)
        return qdrant

    @classmethod
    def construct_instance(
        cls: Type[Qdrant],
        texts: List[str],
        embedding: Embeddings,
        text_splitter: TextSplitter,
        location: Optional[str] = None,
        url: Optional[str] = None,
        port: Optional[int] = 6333,
        grpc_port: int = 6334,
        prefer_grpc: bool = False,
        https: Optional[bool] = None,
        api_key: Optional[str] = None,
        prefix: Optional[str] = None,
        timeout: Optional[float] = None,
        host: Optional[str] = None,
        path: Optional[str] = None,
        collection_name: Optional[str] = None,
        distance_func: str = "Cosine",
        content_payload_key: str = CONTENT_KEY,
        metadata_payload_key: str = METADATA_KEY,
        vector_name: Optional[str] = VECTOR_NAME,
        shard_number: Optional[int] = None,
        replication_factor: Optional[int] = None,
        write_consistency_factor: Optional[int] = None,
        on_disk_payload: Optional[bool] = None,
        hnsw_config: Optional[common_types.HnswConfigDiff] = None,
        optimizers_config: Optional[common_types.OptimizersConfigDiff] = None,
        wal_config: Optional[common_types.WalConfigDiff] = None,
        quantization_config: Optional[common_types.QuantizationConfig] = None,
        init_from: Optional[common_types.InitFrom] = None,
        on_disk: Optional[bool] = None,
        force_recreate: bool = False,
        **kwargs: Any,
    ) -> Qdrant:
        try:
            import qdrant_client  # noqa
        except ImportError:
            raise ImportError(
                "Could not import qdrant-client python package. "
                "Please install it with `pip install qdrant-client`."
            )
        from grpc import RpcError
        from qdrant_client.http import models as rest
        from qdrant_client.http.exceptions import UnexpectedResponse

        # Just do a single quick embedding to get vector size
        partial_embeddings = embedding.embed_documents(["Random sentence init"])
        vector_size = len(partial_embeddings[0])
        collection_name = collection_name or uuid.uuid4().hex
        distance_func = distance_func.upper()
        client, async_client = cls._generate_clients(
            location=location,
            url=url,
            port=port,
            grpc_port=grpc_port,
            prefer_grpc=prefer_grpc,
            https=https,
            api_key=api_key,
            prefix=prefix,
            timeout=timeout,
            host=host,
            path=path,
            **kwargs,
        )
        try:
            # Skip any validation in case of forced collection recreate.
            if force_recreate:
                raise ValueError

            # Get the vector configuration of the existing collection and vector, if it
            # was specified. If the old configuration does not match the current one,
            # an exception is being thrown.
            collection_info = client.get_collection(collection_name=collection_name)
            current_vector_config = collection_info.config.params.vectors
            if isinstance(current_vector_config, dict) and vector_name is not None:
                if vector_name not in current_vector_config:
                    raise QdrantException(
                        f"Existing Qdrant collection {collection_name} does not "
                        f"contain vector named {vector_name}. Did you mean one of the "
                        f"existing vectors: {', '.join(current_vector_config.keys())}? "
                        f"If you want to recreate the collection, set `force_recreate` "
                        f"parameter to `True`."
                    )
                # type: ignore[assignment]
                current_vector_config = current_vector_config.get(vector_name)
            elif isinstance(current_vector_config, dict) and vector_name is None:
                raise QdrantException(
                    f"Existing Qdrant collection {collection_name} uses named vectors. "
                    f"If you want to reuse it, please set `vector_name` to any of the "
                    f"existing named vectors: "
                    f"{', '.join(current_vector_config.keys())}."
                    f"If you want to recreate the collection, set `force_recreate` "
                    f"parameter to `True`."
                )
            elif (
                not isinstance(current_vector_config, dict) and vector_name is not None
            ):
                raise QdrantException(
                    f"Existing Qdrant collection {collection_name} doesn't use named "
                    f"vectors. If you want to reuse it, please set `vector_name` to "
                    f"`None`. If you want to recreate the collection, set "
                    f"`force_recreate` parameter to `True`."
                )

            # Check if the vector configuration has the same dimensionality.
            if current_vector_config.size != vector_size:  # type: ignore[union-attr]
                raise QdrantException(
                    f"Existing Qdrant collection is configured for vectors with "
                    f"{current_vector_config.size} "  # type: ignore[union-attr]
                    f"dimensions. Selected embeddings are {vector_size}-dimensional. "
                    f"If you want to recreate the collection, set `force_recreate` "
                    f"parameter to `True`."
                )

            current_distance_func = (
                current_vector_config.distance.name.upper()  # type: ignore[union-attr]
            )
            if current_distance_func != distance_func:
                raise QdrantException(
                    f"Existing Qdrant collection is configured for "
                    f"{current_distance_func} similarity, but requested "
                    f"{distance_func}. Please set `distance_func` parameter to "
                    f"`{current_distance_func}` if you want to reuse it. "
                    f"If you want to recreate the collection, set `force_recreate` "
                    f"parameter to `True`."
                )
        except (UnexpectedResponse, RpcError, ValueError):
            vectors_config = rest.VectorParams(
                size=vector_size,
                distance=rest.Distance[distance_func],
                on_disk=on_disk,
            )

            # If vector name was provided, we're going to use the named vectors feature
            # with just a single vector.
            if vector_name is not None:
                vectors_config = {  # type: ignore[assignment]
                    vector_name: vectors_config,
                }

            client.recreate_collection(
                collection_name=collection_name,
                vectors_config=vectors_config,
                shard_number=shard_number,
                replication_factor=replication_factor,
                write_consistency_factor=write_consistency_factor,
                on_disk_payload=on_disk_payload,
                hnsw_config=hnsw_config,
                optimizers_config=optimizers_config,
                wal_config=wal_config,
                quantization_config=quantization_config,
                init_from=init_from,
                timeout=timeout,  # type: ignore[arg-type]
            )
        qdrant = cls(
            client=client,
            collection_name=collection_name,
            embeddings=embedding,
            text_splitter=text_splitter,
            content_payload_key=content_payload_key,
            metadata_payload_key=metadata_payload_key,
            distance_strategy=distance_func,
            vector_name=vector_name,
            async_client=async_client,
        )
        return qdrant

    @classmethod
    async def aconstruct_instance(
        cls: Type[Qdrant],
        texts: List[str],
        embedding: Embeddings,
        text_splitter: TextSplitter,
        location: Optional[str] = None,
        url: Optional[str] = None,
        port: Optional[int] = 6333,
        grpc_port: int = 6334,
        prefer_grpc: bool = False,
        https: Optional[bool] = None,
        api_key: Optional[str] = None,
        prefix: Optional[str] = None,
        timeout: Optional[float] = None,
        host: Optional[str] = None,
        path: Optional[str] = None,
        collection_name: Optional[str] = None,
        distance_func: str = "Cosine",
        content_payload_key: str = CONTENT_KEY,
        metadata_payload_key: str = METADATA_KEY,
        vector_name: Optional[str] = VECTOR_NAME,
        shard_number: Optional[int] = None,
        replication_factor: Optional[int] = None,
        write_consistency_factor: Optional[int] = None,
        on_disk_payload: Optional[bool] = None,
        hnsw_config: Optional[common_types.HnswConfigDiff] = None,
        optimizers_config: Optional[common_types.OptimizersConfigDiff] = None,
        wal_config: Optional[common_types.WalConfigDiff] = None,
        quantization_config: Optional[common_types.QuantizationConfig] = None,
        init_from: Optional[common_types.InitFrom] = None,
        on_disk: Optional[bool] = None,
        force_recreate: bool = False,
        **kwargs: Any,
    ) -> Qdrant:
        try:
            import qdrant_client  # noqa
        except ImportError:
            raise ImportError(
                "Could not import qdrant-client python package. "
                "Please install it with `pip install qdrant-client`."
            )
        from grpc import RpcError
        from qdrant_client.http import models as rest
        from qdrant_client.http.exceptions import UnexpectedResponse

        # Just do a single quick embedding to get vector size
        partial_embeddings = await embedding.aembed_documents(texts[:1])
        vector_size = len(partial_embeddings[0])
        collection_name = collection_name or uuid.uuid4().hex
        distance_func = distance_func.upper()
        client, async_client = cls._generate_clients(
            location=location,
            url=url,
            port=port,
            grpc_port=grpc_port,
            prefer_grpc=prefer_grpc,
            https=https,
            api_key=api_key,
            prefix=prefix,
            timeout=timeout,
            host=host,
            path=path,
            **kwargs,
        )
        try:
            # Skip any validation in case of forced collection recreate.
            if force_recreate:
                raise ValueError

            # Get the vector configuration of the existing collection and vector, if it
            # was specified. If the old configuration does not match the current one,
            # an exception is being thrown.
            collection_info = client.get_collection(collection_name=collection_name)
            current_vector_config = collection_info.config.params.vectors
            if isinstance(current_vector_config, dict) and vector_name is not None:
                if vector_name not in current_vector_config:
                    raise QdrantException(
                        f"Existing Qdrant collection {collection_name} does not "
                        f"contain vector named {vector_name}. Did you mean one of the "
                        f"existing vectors: {', '.join(current_vector_config.keys())}? "
                        f"If you want to recreate the collection, set `force_recreate` "
                        f"parameter to `True`."
                    )
                # type: ignore[assignment]
                current_vector_config = current_vector_config.get(vector_name)
            elif isinstance(current_vector_config, dict) and vector_name is None:
                raise QdrantException(
                    f"Existing Qdrant collection {collection_name} uses named vectors. "
                    f"If you want to reuse it, please set `vector_name` to any of the "
                    f"existing named vectors: "
                    f"{', '.join(current_vector_config.keys())}."
                    f"If you want to recreate the collection, set `force_recreate` "
                    f"parameter to `True`."
                )
            elif (
                not isinstance(current_vector_config, dict) and vector_name is not None
            ):
                raise QdrantException(
                    f"Existing Qdrant collection {collection_name} doesn't use named "
                    f"vectors. If you want to reuse it, please set `vector_name` to "
                    f"`None`. If you want to recreate the collection, set "
                    f"`force_recreate` parameter to `True`."
                )

            # Check if the vector configuration has the same dimensionality.
            if current_vector_config.size != vector_size:  # type: ignore[union-attr]
                raise QdrantException(
                    f"Existing Qdrant collection is configured for vectors with "
                    f"{current_vector_config.size} "  # type: ignore[union-attr]
                    f"dimensions. Selected embeddings are {vector_size}-dimensional. "
                    f"If you want to recreate the collection, set `force_recreate` "
                    f"parameter to `True`."
                )

            current_distance_func = (
                current_vector_config.distance.name.upper()  # type: ignore[union-attr]
            )
            if current_distance_func != distance_func:
                raise QdrantException(
                    f"Existing Qdrant collection is configured for "
                    f"{current_vector_config.distance} "  # type: ignore[union-attr]
                    f"similarity. Please set `distance_func` parameter to "
                    f"`{distance_func}` if you want to reuse it. If you want to "
                    f"recreate the collection, set `force_recreate` parameter to "
                    f"`True`."
                )
        except (UnexpectedResponse, RpcError, ValueError):
            vectors_config = rest.VectorParams(
                size=vector_size,
                distance=rest.Distance[distance_func],
                on_disk=on_disk,
            )

            # If vector name was provided, we're going to use the named vectors feature
            # with just a single vector.
            if vector_name is not None:
                vectors_config = {  # type: ignore[assignment]
                    vector_name: vectors_config,
                }

            client.recreate_collection(
                collection_name=collection_name,
                vectors_config=vectors_config,
                shard_number=shard_number,
                replication_factor=replication_factor,
                write_consistency_factor=write_consistency_factor,
                on_disk_payload=on_disk_payload,
                hnsw_config=hnsw_config,
                optimizers_config=optimizers_config,
                wal_config=wal_config,
                quantization_config=quantization_config,
                init_from=init_from,
                timeout=timeout,  # type: ignore[arg-type]
            )
        qdrant = cls(
            client=client,
            collection_name=collection_name,
            embeddings=embedding,
            text_splitter=text_splitter,
            content_payload_key=content_payload_key,
            metadata_payload_key=metadata_payload_key,
            distance_strategy=distance_func,
            vector_name=vector_name,
            async_client=async_client,
        )
        return qdrant

    def _generate_rest_batches(
        self,
        texts: Iterable[str],
        embeded: Iterable[float],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[Sequence[str]] = None,
        batch_size: int = 64,
    ) -> Generator[Tuple[List[str], List[rest.PointStruct]], None, None]:
        from qdrant_client.http import models as rest

        texts_iterator = iter(texts)
        embeded_iterator = iter(embeded)
        metadatas_iterator = iter(metadatas or [])
        ids_iterator = iter(ids or [uuid.uuid4().hex for _ in iter(texts)])
        while batch_texts := list(islice(texts_iterator, batch_size)):

            # Take the corresponding metadata and id for each text in a batch
            batch_metadatas = list(islice(metadatas_iterator, batch_size)) or None
            batch_ids = list(islice(ids_iterator, batch_size))

            # Get the embeddings for all the texts in a batch
            batch_embeddings = list(islice(embeded_iterator, batch_size))

            points = [
                rest.PointStruct(
                    id=point_id,
                    vector=vector
                    if self.vector_name is None
                    else {self.vector_name: vector},
                    payload=payload,
                )

                for point_id, vector, payload in zip(
                    batch_ids,
                    batch_embeddings,
                    self._build_payloads(
                        batch_texts,
                        batch_metadatas,
                        self.content_payload_key,
                        self.metadata_payload_key,
                    ),
                )
            ]

            yield batch_ids, points

    async def _agenerate_rest_batches(
        self,
        texts: Iterable[str],
        embeded: Iterable[float],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[Sequence[str]] = None,
        batch_size: int = 64,
    ) -> AsyncGenerator[Tuple[List[str], List[rest.PointStruct]], None]:
        from qdrant_client.http import models as rest

        texts_iterator = iter(texts)
        embeded_iterator = iter(embeded)
        metadatas_iterator = iter(metadatas or [])
        ids_iterator = iter(ids or [uuid.uuid4().hex for _ in iter(texts)])
        while batch_texts := list(islice(texts_iterator, batch_size)):
            # Take the corresponding metadata and id for each text in a batch
            batch_metadatas = list(islice(metadatas_iterator, batch_size)) or None
            batch_ids = list(islice(ids_iterator, batch_size))

            # Generate the embeddings for all the texts in a batch
            batch_embeddings = list(islice(embeded_iterator, batch_size))

            points = [
                rest.PointStruct(
                    id=point_id,
                    vector=vector
                    if self.vector_name is None
                    else {self.vector_name: vector},
                    payload=payload,
                )
                for point_id, vector, payload in zip(
                    batch_ids,
                    batch_embeddings,
                    self._build_payloads(
                        batch_texts,
                        batch_metadatas,
                        self.content_payload_key,
                        self.metadata_payload_key,
                    ),
                )
            ]

            yield batch_ids, points

    def process_text_in_batches(
        self,
        texts: list[str],
        metadatas:List[dict],
        batch_size: int = 3,
        context_length: int = 8192
    ):
        n_texts = len(texts)

        if not hasattr(self.text_splitter, 'tokenizer'):
            warnings.warn(
                """
                Warning: The 'tokenizer' attribute is not implemented.
                It is recommended to implement for token length calculations.
                Example:
                    from transformer import Autokenizer
                    tokenizer = AutoTokenizer.from_pretrained(
                        'jinaai/jina-embeddings-v3', trust_remote=True
                    )
                    
                    text_spliter.tokenizer = tokenizer
                """,
                UserWarning
            )

        for start in range(0, n_texts, batch_size):
            end = min(start + batch_size, n_texts)
            subtexts, subembeds, submetadatas = [], [], []

            for i in range(start, end):
                text = texts[i]

                subtext = [
                    doc.page_content
                    for doc in self.text_splitter.create_documents([text])
                ]

                # Check token length and adjust if it exceeds context_length
                size = len(subtext)
                if hasattr(self.text_splitter, 'tokenizer'):
                    size = len(self.text_splitter.tokenizer(text)['input_ids'])

                if size > context_length:
                    # Split into smaller chunks if text exceeds context_length
                    bz = len(subtext) // 4
                    for k in range(0, len(subtext), bz):
                        subembeds.extend(self._embed_texts(subtext[k: k+bz]))
                else:
                    subembeds.extend(self._embed_texts(subtext))

                # Extend metadatas and subtexts accordingly
                submetadatas.extend([metadatas[i]] * len(subtext))
                subtexts.extend(subtext)

            assert len(subtexts) == len(subembeds) == len(submetadatas)

            yield subtexts, submetadatas, subembeds
